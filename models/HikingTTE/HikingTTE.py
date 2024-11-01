import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(project_root)
from src.common import utils
import models.HikingTTE.base as base

EPS = 10

class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size = 128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim = 1)

        hidden = F.leaky_relu(self.input2hid(inputs))   

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = torch.abs(pred - label) / label
        return {'label': label, 'pred': pred}, loss.mean()
    

class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))
        out = self.hid2out(hidden)
        return out

    def eval_on_batch(self, pred_n, lens, label_n, time_diff_mean, time_diff_std, loss_fn = 'MAPE'):
        label_n = nn.utils.rnn.pack_padded_sequence(label_n, lens, batch_first = True,enforce_sorted=False)[0]
        label_n = label_n.view(-1, 1)


        label = label_n * time_diff_std + time_diff_mean
        pred = pred_n * time_diff_std + time_diff_mean

        if loss_fn == 'MAPE':
            loss = torch.abs(pred - label) / (label + EPS)
        elif loss_fn == 'MSE':
            loss = (pred - label) ** 2
        elif loss_fn == 'MAE':
            loss = torch.abs(pred - label)
        
        return loss.mean()
    

class Net(nn.Module):
    def __init__(self, num_final_fcs = 3, final_fc_size = 128, alpha = 0.3, attribute_names=None):
        super(Net, self).__init__()
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.alpha = alpha

        if attribute_names is None:
            raise ValueError("Attribute names must be provided")
        else:
            self.attribute_names = attribute_names

        self.build()
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(param.data)

    def build(self):
        self.attr_net = base.Attr.Net(attribute_names=self.attribute_names)
        self.lstm_attention = base.LSTMAttention.Net(attr_size = self.attr_net.out_size())

        self.entire_estimate = EntireEstimator(input_size =  self.lstm_attention.out_size() + self.attr_net.out_size(), num_final_fcs = self.num_final_fcs, hidden_size = self.final_fc_size)

        self.local_estimate = LocalEstimator(input_size = self.lstm_attention.out_size())


    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr)
        sptm_s, sptm_l, sptm_t = self.lstm_attention(traj, attr_t, config)

        entire_out = self.entire_estimate(attr_t, sptm_t)

        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def eval_on_batch(self, attr_n, traj_n, config, local_loss_fn = 'MAPE'):
        if self.training:
            entire_out, (local_out_n, local_length) = self(attr_n, traj_n, config)
        else:
            entire_out = self(attr_n, traj_n, config)

        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr_n['travel_time'], config['travel_time']["mean"], config['travel_time']['std'])

        if self.training:
            time_diff_mean, time_diff_std = config['time_diff(s)']['mean'], config['time_diff(s)']['std']

            local_label_n = traj_n['time_diff_n']

            local_loss = self.local_estimate.eval_on_batch(local_out_n, local_length, local_label_n, time_diff_mean, time_diff_std, local_loss_fn)

            return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss
        else:
            return pred_dict, entire_loss
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, attr_size):
        super(Net, self).__init__()
        self.process_coords = nn.Linear(8, 16)
        self.rnn = nn.LSTM(input_size = 16 + attr_size, \
                                hidden_size = 128, \
                                num_layers = 2, \
                                batch_first = True
                                )
        self.attr2atten = nn.Linear(attr_size, 128)

    def out_size(self):
        return 128


    def attent_pooling(self, hiddens, lens, attr_t):
        attent = torch.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)

        alpha = torch.bmm(hiddens, attent)
        alpha = torch.exp(alpha) 

        alpha = alpha / torch.sum(alpha, dim = 1, keepdim = True)  

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = torch.bmm(hiddens, alpha)
        hiddens = torch.squeeze(hiddens)

        return hiddens
    
    def forward(self, traj, attr_t, config):

        latitude_diff_n = torch.unsqueeze(traj['latitude_diff_n'], dim = 2)
        longitude_diff_n = torch.unsqueeze(traj['longitude_diff_n'], dim = 2)
        elevation_diff_n = torch.unsqueeze(traj['elevation_diff_n'], dim = 2)
        terrain_slope_n = torch.unsqueeze(traj['terrain_slope_n'], dim = 2)
        slope_n = torch.unsqueeze(traj['slope_n'], dim = 2)
        distance_diff_n = torch.unsqueeze(traj['distance_diff_n'], dim = 2)
        elevation_n = torch.unsqueeze(traj['elevation_n'], dim = 2)

        pred_velocity_n = torch.unsqueeze(traj['pred_velocity_n'], dim = 2)

        locs = torch.cat((latitude_diff_n, longitude_diff_n, elevation_diff_n, terrain_slope_n, slope_n, distance_diff_n, elevation_n,pred_velocity_n), dim = 2)

        locs = torch.tanh(self.process_coords(locs))
        attr_t = torch.unsqueeze(attr_t, dim = 1)
        expand_attr_t = attr_t.expand(locs.size()[:2] + (attr_t.size()[-1], ))
        locs = torch.cat((locs, expand_attr_t), dim = 2)
        lens = (traj['latitude_diff_n'] != -1000000).sum(dim=1).tolist()

        packed_inputs = nn.utils.rnn.pack_padded_sequence(locs, lens, batch_first = True, enforce_sorted=False)
        packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)

        return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)
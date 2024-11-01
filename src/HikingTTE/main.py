import os
import torch
import sys
import datetime
import argparse
import json
import wandb
import numpy as np
import scipy.optimize as opt

sys.path.append('..')
import common.utils as utils
from common.logger import Logger
from common import data_loader_split

sys.path.append('../..')
from models.HikingTTE import HikingTTE

def batch_to_device(batch, device):
    latitude_diff_n = batch['lists']['latitude_diff'].to(device)
    longitude_diff_n = batch['lists']['longitude_diff'].to(device)
    elevation_diff_n = batch['lists']['elevation_diff(m)'].to(device)
    distance_diff_n = batch['lists']['distance_diff(m)'].to(device)

    slope_n = batch['lists']['slope(degrees)'].to(device)
    terrain_slope_n = batch['lists']['Terrain_slope(degrees)'].to(device)

    elevation_n = batch['lists']['elevation'].to(device)

    time_diff_n = batch['target_lists']['time_diff(s)'].to(device)

    traj_n = {
        "latitude_diff_n": latitude_diff_n,
        "longitude_diff_n": longitude_diff_n,
        "elevation_diff_n": elevation_diff_n,
        "terrain_slope_n": terrain_slope_n,
        "slope_n": slope_n,
        "distance_diff_n": distance_diff_n,
        "elevation_n": elevation_n,
        "time_diff_n": time_diff_n,
    }

    max_elevation_n = batch['numerical']['max_elevation'].to(device)
    min_elevation_n = batch['numerical']['min_elevation'].to(device)
    uphill_n = batch['numerical']['uphill'].to(device)
    downhill_n = batch['numerical']['downhill'].to(device)
    length_2d_n = batch['numerical']['length_2d'].to(device)
    travel_time_n = batch['target']['travel_time'].to(device)

    attr_n = {
        "uphill": uphill_n,
        "downhill": downhill_n,
        "max_elevation": max_elevation_n,
        "min_elevation": min_elevation_n,
        "length_2d": length_2d_n,
        "travel_time": travel_time_n
    }
    return traj_n, attr_n



def lorentz(slope, a, b, c, d, e):
    return c * (1 / (np.pi * b * (1 + ((slope - a) / b) ** 2))) + d + e * slope

def fit_lorentz(valid_slope,valid_velocity,lorentz_coef,lorentz_variable):
    fixed_params = {key: value for key, value in lorentz_coef.items() if key not in lorentz_variable} 
    initial_values = [lorentz_coef[var] for var in lorentz_variable]

    def error_func(params):
        params_dict = {lorentz_variable[i]: params[i] for i in range(len(params))}
        all_params = {**fixed_params, **params_dict}
        return lorentz(valid_slope, all_params['a'], all_params['b'], all_params['c'], all_params['d'], all_params['e']) - valid_velocity

    result = opt.least_squares(error_func, x0=initial_values)
    return result.x

def front_batch_to_lorentz(front_batch, config, lorentz_coef, lorentz_variable):
    velocity_mean = config['velocity(m/s)']['mean']
    velocity_std = config['velocity(m/s)']['std']
    slope_degree_mean = config['slope(degrees)']['mean']
    slope_degree_std = config['slope(degrees)']['std']

    slope_n = front_batch['lists']['slope(degrees)']
    velocity_n = front_batch['lists']['velocity(m/s)']

    padding_mask = (slope_n != -1000000)
    valid_count = padding_mask.sum(dim=1)

    slope = slope_degree_mean + slope_degree_std * slope_n
    velocity = velocity_mean + velocity_std * velocity_n

    slope_with_zero_padded = slope * padding_mask
    velocity_with_zero_padded = velocity * padding_mask

    batch_size = slope_with_zero_padded.size(0)

    fitted_params_list = []
    for batch_idx in range(batch_size):
        valid_slope_batch = slope_with_zero_padded[batch_idx, 1:valid_count[batch_idx]]
        valid_velocity_batch = velocity_with_zero_padded[batch_idx, 1:valid_count[batch_idx]]
        result = fit_lorentz(valid_slope_batch.numpy(), valid_velocity_batch.numpy(), lorentz_coef, lorentz_variable)
        

        fitted_params = lorentz_coef.copy()
        for i, var in enumerate(lorentz_variable):
            fitted_params[var] = result[i]

        fitted_params_list.append(torch.tensor([fitted_params['a'], fitted_params['b'], fitted_params['c'], fitted_params['d'], fitted_params['e']], dtype=torch.float32))
        

    fitted_params_tensor = torch.stack(fitted_params_list)
    return fitted_params_tensor



def train(model, dataloader, optimizer, device, epoch, config,lorentz_coef, lorentz_variable, local_loss_fn):
    total_batches = len(dataloader)
    model.train()
    total_loss = 0
    total_samples = 0

    slope_list = [-30, -20, -10, 0, 10, 20, 30]

    for i, (front_batch, zero_shift_back_batch) in enumerate(dataloader):
        optimizer.zero_grad()

        fitted_params_tensor = front_batch_to_lorentz(front_batch, config,lorentz_coef, lorentz_variable)
        velocity_mean = config['velocity(m/s)']['mean']
        velocity_std = config['velocity(m/s)']['std']

        zero_shift_back_traj_n, zero_shift_back_attr_n = batch_to_device(zero_shift_back_batch, device)

        for slope in slope_list:
            a = fitted_params_tensor[:, 0]
            b = fitted_params_tensor[:, 1]
            c = fitted_params_tensor[:, 2]
            d = fitted_params_tensor[:, 3]
            e = fitted_params_tensor[:, 4]

            speed_at_slope = c * (1 / (np.pi * b * (1 + ((slope - a) / b) ** 2))) + d + e * slope
            speed_at_slope_n = (speed_at_slope - velocity_mean) / velocity_std
            zero_shift_back_attr_n[str(slope)] = speed_at_slope_n.to(device)

        slope_n = zero_shift_back_batch['lists']['slope(degrees)'].to(device)

        slope_mean = config['slope(degrees)']['mean']
        slope_std = config['slope(degrees)']['std']

        slope = slope_mean + slope_std * slope_n

        a = fitted_params_tensor[:, 0].unsqueeze(1).expand_as(slope).to(device)
        b = fitted_params_tensor[:, 1].unsqueeze(1).expand_as(slope).to(device)
        c = fitted_params_tensor[:, 2].unsqueeze(1).expand_as(slope).to(device)
        d = fitted_params_tensor[:, 3].unsqueeze(1).expand_as(slope).to(device)
        e = fitted_params_tensor[:, 4].unsqueeze(1).expand_as(slope).to(device)

        pred_velocity = lorentz(slope, a, b, c, d, e)
        pred_velocity_n = (pred_velocity - velocity_mean) / velocity_std

        zero_shift_back_traj_n['pred_velocity_n'] = pred_velocity_n

        back_latitude_diff_n = zero_shift_back_batch['lists']['latitude_diff'].to(device)

        batch_loss_ave = 0
        _, batch_loss_ave = model.eval_on_batch(zero_shift_back_attr_n, zero_shift_back_traj_n, config, local_loss_fn)

        batch_loss_ave.backward()
        optimizer.step()

        batch_size = back_latitude_diff_n.size(0)
        total_loss += batch_loss_ave.item() * batch_size
        total_samples += batch_size

        progress = ((i + 1) / total_batches) * 100
        print(f'\rEpoch {epoch+1}, Batch {i+1}/{total_batches}, Progress: {progress:.2f}%', end='')

    average_loss = total_loss / total_samples
    return average_loss

def evaluate(model, dataloader, device, data_dir, config,lorentz_coef, lorentz_variable):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mape = 0
    total_mse = 0
    total_samples = 0

    slope_list = [-30, -20, -10, 0, 10, 20, 30]

    with torch.no_grad():
        for front_batch, zero_shift_back_batch in dataloader:

            fitted_params_tensor = front_batch_to_lorentz(front_batch, config,lorentz_coef, lorentz_variable)

            velocity_mean = config['velocity(m/s)']['mean']
            velocity_std = config['velocity(m/s)']['std']

            zero_shift_back_traj_n, zero_shift_back_attr_n = batch_to_device(zero_shift_back_batch, device)

            for slope in slope_list:
                a = fitted_params_tensor[:, 0]
                b = fitted_params_tensor[:, 1]
                c = fitted_params_tensor[:, 2]
                d = fitted_params_tensor[:, 3]
                e = fitted_params_tensor[:, 4]

                speed_at_slope = c * (1 / (np.pi * b * (1 + ((slope - a) / b) ** 2))) + d + e * slope
                speed_at_slope_n = (speed_at_slope - velocity_mean) / velocity_std
                zero_shift_back_attr_n[str(slope)] = speed_at_slope_n.to(device)

            slope_n = zero_shift_back_batch['lists']['slope(degrees)'].to(device)
            slope_mean = config['slope(degrees)']['mean']
            slope_std = config['slope(degrees)']['std']
            slope = slope_mean + slope_std * slope_n

            a = fitted_params_tensor[:, 0].unsqueeze(1).expand_as(slope).to(device)
            b = fitted_params_tensor[:, 1].unsqueeze(1).expand_as(slope).to(device)
            c = fitted_params_tensor[:, 2].unsqueeze(1).expand_as(slope).to(device)
            d = fitted_params_tensor[:, 3].unsqueeze(1).expand_as(slope).to(device)
            e = fitted_params_tensor[:, 4].unsqueeze(1).expand_as(slope).to(device)

            pred_velocity = lorentz(slope, a, b, c, d, e)
            pred_velocity_n = (pred_velocity - velocity_mean) / velocity_std

            zero_shift_back_traj_n['pred_velocity_n'] = pred_velocity_n

            back_latitude_diff_n = zero_shift_back_batch['lists']['latitude_diff'].to(device)     

            loss = 0
            pred_dict, loss = model.eval_on_batch(zero_shift_back_attr_n, zero_shift_back_traj_n, config)
            predictions = pred_dict['pred']
            batch_size = back_latitude_diff_n.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            back_travel_time_n = zero_shift_back_batch['target']['travel_time'].to(device) 

            unnorm_back_travel_times = utils.unnormalize(back_travel_time_n.cpu(), 'travel_time', data_dir)
            predictions = predictions.cpu()
            predictions = predictions.squeeze()

            mae = torch.sum(torch.abs(predictions - unnorm_back_travel_times))
            mape = torch.sum(torch.abs(predictions - unnorm_back_travel_times) / unnorm_back_travel_times)
            mse = torch.sum((predictions - unnorm_back_travel_times) ** 2)

            total_mae += mae.item()
            total_mape += mape.item()
            total_mse += mse.item()

    average_loss = total_loss / total_samples
    average_mae = total_mae / total_samples
    average_mape = total_mape / total_samples
    average_mse = total_mse / total_samples

    return average_loss, average_mae, average_mape, average_mse



def main():
    parser = argparse.ArgumentParser()

    project_dir = os.path.join(os.path.dirname(__file__), '../..')
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_dir, 'Datasets/processed_dataset'), help='Directory of the dataset. default = Datasets/processed_dataset')

    parser.add_argument('--model_name', type=str, default='HikingTTE', help='model name.default=HikingTTE')

    parser.add_argument('--alpha', type = float, default = 0.5, help = 'Weight coefficient for multi-task learning,default=0.5')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size,default=64')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs,default=10')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate,default=0.001')

    parser.add_argument('--task', type=str, default='train', help='train,resume or test,default=train')

    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model weights,default=10')

    parser.add_argument('--local_loss_fn', type=str, default='MAPE', help='Specify the loss function for LocalEstimator,default=MAPE')

    parser.add_argument('--project_name', type=str, default='test_project', help='wandb project name,default=test_TTE_project')
    parser.add_argument('--exp_name', type=str, default='HikingTTE', help='experiment name,default=HikingTTE')

    parser.add_argument('--split_ratio', type=float, default=0.5, help='Ratio to split into front and back segment,default=0.5')

    parser.add_argument('--lorentz_coef', type=str, default='{"a": -1.4579, "b": 22.0787, "c": 76.3271, "d": 0.0525, "e": -3.2002e-4}', help='Parameters of the Lorentz function.')
    parser.add_argument('--lorentz_variable', type=str, default='["c"]', help='Variable(s) of the Lorentz function,default=["c"]')

    parser.add_argument('--wandb', action='store_true', help='Whether to use Wandb')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='Flag for not using Wandb')
    parser.set_defaults(wandb=True)

    args = parser.parse_args()

    args.lorentz_coef = json.loads(args.lorentz_coef)
    args.lorentz_variable = json.loads(args.lorentz_variable)

    result_dir = os.path.join(project_dir, "result", args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_file_dir = os.path.join(args.data_dir, 'config.json')
    with open(config_file_dir, 'r') as f:
        config_file = json.load(f)

    attribute_names = ['length_2d', 'uphill', 'downhill', 'max_elevation', 'min_elevation']

    slope_list = [-30, -20, -10, 0, 10, 20, 30]
    attribute_names += [str(slope) for slope in slope_list]

    model = HikingTTE.Net(alpha=args.alpha,attribute_names=attribute_names)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task in ['train','resume']:
        train_file = os.path.join(args.data_dir, 'train.jsonl')
        valid_file = os.path.join(args.data_dir, 'valid.jsonl')
        train_dataloader = data_loader_split.get_splitted_zero_shift_dataloader(train_file, args.batch_size, data_dir=args.data_dir, split_ratio=args.split_ratio, shuffle=True)
        valid_dataloader = data_loader_split.get_splitted_zero_shift_dataloader(valid_file, args.batch_size, data_dir=args.data_dir, split_ratio=args.split_ratio, shuffle=False)

        if args.task == "train":
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir = os.path.join(result_dir, f"{current_time}_{args.exp_name}") 
            os.makedirs(log_dir, exist_ok=True)
            weights_dir = os.path.join(log_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)

            if args.wandb:
                wandb.init(project=args.project_name, name=args.exp_name)
                run_id = wandb.run.id
                with open(os.path.join(log_dir, 'wandb.json'), 'w') as f:
                    json.dump({'run_id': run_id, 'project_name': args.project_name}, f)

            logger = Logger(os.path.join(log_dir, 'training.log'))

            logger.log(str(model))
            logger.log(f'alpha: {args.alpha}')
            logger.log(f'attribute names: {attribute_names}')
            logger.log(f'local_loss_fn: {args.local_loss_fn}')
            logger.log(str(optimizer))

            hyperparameters = {
                'alpha': args.alpha,
                'local_loss_fn': args.local_loss_fn,
                'batch_size': args.batch_size,
                'data_dir': args.data_dir,
                'lr': args.lr,
            }

            with open(os.path.join(log_dir, 'hyperparameters.json'), 'w') as f:
                json.dump(hyperparameters, f)

            start_epoch = 0

        else: #args.task == "resume"
            selected_dir = utils.list_and_choose_directory(result_dir)
            weights_dir = os.path.join(selected_dir, 'weights')
            selected_model_file = utils.select_latest_model_file(weights_dir)

            start_epoch = selected_model_file.split('_')[-1].split('.')[0]

            model.load_state_dict(torch.load(selected_model_file))
            model.to(device)
            model.train()

            if args.wandb:
                with open(os.path.join(selected_dir, 'wandb.json'), 'r') as f:
                    wandb_info = json.load(f)
                    run_id = wandb_info['run_id']
                    project_name = wandb_info['project_name']
                wandb.init(project=project_name, id=run_id, resume=True)

            logger = Logger(os.path.join(selected_dir, 'training.log'), mode='a')
            with open(os.path.join(selected_dir, 'hyperparameters.json'), 'r') as f:
                saved_hyperparameters = json.load(f)

            if any(saved_hyperparameters[key] != getattr(args, key) for key in saved_hyperparameters):
                print("Warning: The current hyperparameters do not match those used in training.")

        for epoch in range(int(start_epoch),args.num_epochs):
            train_average_loss = train(model, train_dataloader,  optimizer, device , epoch, config_file, args.lorentz_coef,args.lorentz_variable,args.local_loss_fn)
            valid_average_loss,_,valid_MAPE,_ = evaluate(model, valid_dataloader, device, args.data_dir,config_file, args.lorentz_coef,args.lorentz_variable)

            print(f'Epoch {epoch + 1}/{args.num_epochs}, Train Average Loss: {train_average_loss}, Valid Average Loss: {valid_average_loss}, Valid_MAPE = {valid_MAPE}')
            logger.log(f'Epoch {epoch + 1}/{args.num_epochs}, Train Average Loss: {train_average_loss}, Valid Average Loss: {valid_average_loss}, Valid_MAPE = {valid_MAPE}')
            if args.wandb:
                wandb.log({'Train Average Loss': train_average_loss, 'Valid Average Loss': valid_average_loss,'valid_MAPE':valid_MAPE}, step=epoch + 1, commit=True)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.num_epochs - 1:
                weight_path = os.path.join(weights_dir, f'model_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), weight_path)
                print(f'Model weights saved to {weight_path}')
        if args.wandb:
            wandb.finish()

    elif args.task == "test":
        selected_dir = utils.list_and_choose_directory(result_dir)
        weights_dir = os.path.join(selected_dir, 'weights')
        selected_model_file = utils.select_and_choose_model_file(weights_dir)
        epoch_number = selected_model_file.split('_')[-1].split('.')[0]

        model.load_state_dict(torch.load(selected_model_file))
        model.to(device)
        model.eval()

        test_file = os.path.join(args.data_dir, 'test.jsonl')
        test_dataloader = data_loader_split.get_splitted_zero_shift_dataloader(test_file, args.batch_size, data_dir=args.data_dir, split_ratio=args.split_ratio, shuffle=False)
        average_loss, average_mae, average_mape, average_mse = evaluate(model, test_dataloader, device, args.data_dir, config_file, args.lorentz_coef,args.lorentz_variable)
        average_mae = average_mae/3600
        average_mse = average_mse/(3600*3600)

        results_path = os.path.join(selected_dir, 'evaluation_results.txt')
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(results_path, 'a') as file:
            file.write(f'Evaluation at {current_time} for Epoch {epoch_number} - Loss: {average_loss}, MAE: {average_mae}, MAPE: {average_mape}, MSE: {average_mse}\n')

        print(f'Evaluation results saved to {results_path}')
        print(f'Evaluation results - Loss: {average_loss}, MAE: {average_mae}, MAPE: {average_mape}, MSE: {average_mse}')


if __name__ == '__main__':
    main()
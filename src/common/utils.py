import torch
import json
import os
import datetime


def read_config(data_dir):
    return json.load(open(os.path.join(data_dir, 'config.json'), 'r'))

def normalize(x, key, data_dir):
    config = read_config(data_dir)
    mean = config[key]['mean']
    std = config[key]['std']
    return (x - mean) / std

def unnormalize(x, key, data_dir):
    config = read_config(data_dir)
    mean = config[key]['mean']
    std = config[key]['std']
    return x * std + mean

def list_and_choose_directory(base_dir):
    all_subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not all_subdirs:
        raise FileNotFoundError("No subdirectories found.")

    sorted_subdirs = sorted(all_subdirs, key=os.path.getmtime, reverse=True)

    print("Available directories:")
    for idx, subdir in enumerate(sorted_subdirs, 1):
        print(f"{idx}. {subdir} - Last Modified: {datetime.datetime.fromtimestamp(os.path.getmtime(subdir)).strftime('%Y-%m-%d %H:%M:%S')}")

    choice = int(input("Enter the number of the directory to use: "))
    if choice < 1 or choice > len(sorted_subdirs):
        raise ValueError("Invalid choice. Please select a valid number from the list.")

    selected_dir = sorted_subdirs[choice - 1]
    return selected_dir

def select_and_choose_model_file(weights_directory):
    all_weights = [os.path.join(weights_directory, f) for f in os.listdir(weights_directory) if f.endswith('.pth')]
    if not all_weights:
        raise FileNotFoundError("No model weight files found.")

    sorted_weights = sorted(all_weights)

    print("Available model files:")
    for idx, weight_file in enumerate(sorted_weights, 1):
        print(f"{idx}. {weight_file}")

    choice = int(input("Enter the number of the model file to load: "))
    if choice < 1 or choice > len(sorted_weights):
        raise ValueError("Invalid choice. Please select a valid number from the list.")

    selected_weight_file = sorted_weights[choice - 1]
    return selected_weight_file

def select_latest_model_file(weights_directory):
    all_weights = [os.path.join(weights_directory, f) for f in os.listdir(weights_directory) if f.endswith('.pth')]
    if not all_weights:
        raise FileNotFoundError("No model weight files found.")

    latest_weight_file = max(all_weights, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return latest_weight_file


def setup_experiment_directory(base_dir, project_name):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(base_dir, project_name, current_time)
    os.makedirs(log_dir, exist_ok=True)

    weights_dir = os.path.join(log_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    wandb_dir = os.path.join(log_dir, 'wandb_ids')
    os.makedirs(wandb_dir, exist_ok=True)

    return {
        'log_dir': log_dir,
        'weights_dir': weights_dir,
        'wandb_dir': wandb_dir
    }


def create_log_directories(base_dir, save_name = ""):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(base_dir, f"{current_time}_{save_name}")
    os.makedirs(log_dir, exist_ok=True)

    weights_dir = os.path.join(log_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    wandb_dir = os.path.join(log_dir, 'wandb_ids')
    os.makedirs(wandb_dir, exist_ok=True)

    return log_dir, weights_dir, wandb_dir, current_time
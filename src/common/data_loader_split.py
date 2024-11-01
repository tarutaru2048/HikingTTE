import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from . import utils

import numpy as np
import ujson as json


class MySplittedZeroShiftDataset(Dataset):
    def __init__(self, input_file, split_ratio=0.3):
        with open(input_file, "r") as f:
            self.tracks = f.readlines()
            self.tracks = list(map(lambda x: json.loads(x), self.tracks))
            self.lengths = list(map(lambda x: len(x['latitude']), self.tracks))
        self.split_ratio = split_ratio

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        track = self.tracks[idx]
        split_index = int(len(track['latitude']) * self.split_ratio)
        front_track = {key: val[:split_index] if isinstance(val, list) else val for key, val in track.items()}
        zero_shift_back_track = {key: val[split_index:] if isinstance(val, list) else val for key, val in track.items()}

        if isinstance(track['elapsed_time(s)'], list):
            back_elapsed_times = np.array(track['elapsed_time(s)'][split_index:])
            back_elapsed_times -= back_elapsed_times[0]
            zero_shift_back_track['elapsed_time(s)'] = back_elapsed_times.tolist()
            zero_shift_back_track['travel_time'] = back_elapsed_times[-1]
            front_track['travel_time'] = track['elapsed_time(s)'][split_index]
        
        return front_track, zero_shift_back_track

def process_batch(batch, data_dir):

    try:
        with open('data_attributes.json', 'r') as f:
            data_attributes = json.load(f)
            category_attrs = data_attributes['category']
            time_attrs = data_attributes['time']
            numerical_attrs = data_attributes['numerical']
            list_attrs = data_attributes['lists']
            target_attrs = data_attributes['target']
            list_target_attrs = data_attributes['target_lists']
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    category, time, numerical, lists, target, list_target = {}, {}, {}, {}, {}, {}

    lengths = list(map(lambda x: len(x['latitude']), batch))

    for key in category_attrs:
        x = torch.LongTensor([d[key] for d in batch])
        category[key] = x

    for key in time_attrs:
        x = torch.LongTensor([d[key] for d in batch])
        time[key] = x

    for key in numerical_attrs:
        x = torch.FloatTensor([d[key] for d in batch])
        x = utils.normalize(x, key, data_dir)
        numerical[key] = x

    for key in list_attrs:
        normalized_tensors = [torch.tensor([utils.normalize(val, key, data_dir) for val in d[key]]) for d in batch]
        padding_value = -1000000
        lists[key] = pad_sequence(normalized_tensors, padding_value=padding_value, batch_first=True)

    for key in target_attrs:
        x = torch.FloatTensor([d[key] for d in batch])
        x = utils.normalize(x, key, data_dir)
        target[key] = x

    for key in list_target_attrs:
        normalized_tensors = [torch.tensor([utils.normalize(val, key, data_dir) for val in d[key]]) for d in batch]
        padding_value = -1000000
        list_target[key] = pad_sequence(normalized_tensors, padding_value=padding_value, batch_first=True)

    return {
        'category': category,
        'time': time,
        'numerical': numerical,
        'lists': lists,
        'target': target,
        'target_lists': list_target
    }


def create_collate_fn(data_dir, split_ratio=0.3):
    def collate_fn(batch):
        front_batch = [item[0] for item in batch]
        back_batch = [item[1] for item in batch]
        front_collated = process_batch(front_batch, data_dir)
        back_collated = process_batch(back_batch, data_dir)
        return front_collated, back_collated
    
    return collate_fn

def get_splitted_zero_shift_dataloader(input_file, batch_size, data_dir, split_ratio=0.3, shuffle=True):
    splitted_zero_shift_dataset = MySplittedZeroShiftDataset(input_file, split_ratio=split_ratio)
    splitted_zero_shift_dataloader = DataLoader(splitted_zero_shift_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=create_collate_fn(data_dir))
    return splitted_zero_shift_dataloader
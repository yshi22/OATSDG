import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from omegaconf import open_dict
import torch
import copy

APPLIANCES_HOUSE = {
    'dishwasher':['5','6','7','9','13','16','18','20'],
    'fridge':['2','4','5','6','9','12','15','18'],
    'washingmachine':['2','5','7','8','9','15','16','17'],
    'microwave':['4','5','6','10','12', '17', '18', '19']
}

class refit_dataset(Dataset):
    def __init__(self, domain_idx, dataLoader, device, datas = None, isLoad = False):
        if isLoad:
            self.samples = datas[0].to(device)
            self.labels =  datas[1].to(device)
            self.domain =  datas[2].to(device)
        else:
            samples, labels, target_seq = dataLoader.get_data()
            self.samples = torch.from_numpy(samples).view([-1, 1, dataLoader.window_size]).float().to(device)
            self.labels = torch.from_numpy(labels).view(-1).float().to(device)
            self.domain = torch.from_numpy(np.full(self.labels.shape[0], int(domain_idx), dtype=int)).long().to(device)

    def __getitem__(self, index):
        return  self.samples[index], \
                self.labels[index], \
                self.domain[index]

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def load_data(directory, domain_idx, device):
        samples = np.load(f"{directory}_samples.npy")
        labels = np.load(f"{directory}_labels.npy")

        samples = torch.from_numpy(samples).float()
        labels = torch.from_numpy(labels).float()
        domain = torch.from_numpy(np.full(labels.shape[0], int(domain_idx), dtype=int)).long()
        return refit_dataset(None, None, device, (samples, labels, domain), True)
        



def load_refit_Loaders(args, target_domain):
    source_domain_list = copy.deepcopy(APPLIANCES_HOUSE[args.appliance])
    source_domain_list.remove(target_domain)
    with open_dict(args):
        args.source_domain_list = source_domain_list

    source_loaders, eval_loaders = [], []
    for domain_idx, loc in enumerate(source_domain_list):
        train_set = refit_dataset.load_data(f'{args.data_dir}/{args.appliance}/{args.appliance}_{loc}', domain_idx, args.device)
        source_loaders.append(train_set)
    
    source_loaders = [DataLoader(
                            ConcatDataset(copy.deepcopy(source_loaders)), batch_size=args.batch_size, shuffle=True)]
    
    target_set = refit_dataset.load_data(f'{args.data_dir}/{args.appliance}/{args.appliance}_{target_domain}_test', -1, args.device)
    target_loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=False)
    
    with open(f'{args.data_dir}/{args.appliance}/stats.pkl', 'rb') as f:
        stats = pickle.load(f)
        with open_dict(args):
            args.stats = stats
            args.mean = stats[-2]
            args.std = stats[-1]
    return source_loaders, target_loader




def calculate_mean_std(datasets):
    all_samples = torch.cat([dataset.samples for dataset in datasets])
    mean_samples = all_samples.mean()
    std_samples = all_samples.std()

    return mean_samples, std_samples, mean_samples, std_samples

def z_score_normalize(dataset, mean_samples, std_samples, _, __):
    dataset.samples = (dataset.samples - mean_samples) / std_samples
    dataset.target_seq = (dataset.target_seq - mean_samples) / std_samples
    dataset.labels = (dataset.labels - mean_samples) / std_samples
    return dataset
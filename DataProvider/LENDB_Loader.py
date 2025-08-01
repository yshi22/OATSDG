from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import numpy as np
import torch
import pickle

class LENDB_dataPrep:
    def __init__(self, args):
        train_path = args.data_dir
        with open(train_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        self.train_datasets = []
        for idx, (station, station_data) in enumerate(loaded_data.items()):
            data = np.array(station_data['inputs'])
            label = np.array(station_data['labels'])
            id = [idx] * len(data)
            self.train_datasets.append(LENDB_Dataset(id, data, label, args.device))
    
    def get_lendb_Loaders(self, args, target_domain):
        train_datasets = self.train_datasets
        source_loaders = []
        for idx, train_set in enumerate(train_datasets):
            if idx == target_domain:
                target_loader = DataLoader(train_set , batch_size=args.batch_size, shuffle=False)
            else:
                source_loaders.append(train_set)
        source_loaders = [DataLoader(
                                ConcatDataset(source_loaders), batch_size=args.batch_size, shuffle=True)]

        return source_loaders, target_loader


class LENDB_Dataset(Dataset):
    def __init__(self, id, data, label, device):
        data_size = len(data)
        self.data = torch.tensor(data).float().view([data_size, 3, -1]).to(device)
        self.label = torch.tensor(label).float().to(device)
        self.id = torch.tensor(id).long().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.id[index]

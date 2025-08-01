from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch
import pickle

class PRSA_dataPrep:
    def __init__(self, args):
        with open(args.data_dir, 'rb') as f:
            loaded_data = pickle.load(f)
        
        self.train_datasets = []
        for idx, (station, station_data) in enumerate(loaded_data.items()):
            data, label = station_data
            id = [idx] * len(data)
            self.train_datasets.append(PRSA_Dataset(id, data, label, args.device))

        
    def get_dataset(self):
        return self.train_datasets
    
    def get_Loaders(self, args, target_domain):
        train_datasets = self.get_dataset()
        source_loaders = []
        for idx, train_set in enumerate(train_datasets):
            if idx == target_domain:
                target_loader = DataLoader(train_set , batch_size=args.batch_size, shuffle=False)
            else:
                source_loaders.append(train_set)
        source_loaders = [DataLoader(
                                ConcatDataset(source_loaders), batch_size=args.batch_size, shuffle=True)]

        return source_loaders, target_loader


class PRSA_Dataset(Dataset):
    def __init__(self, id, data, label, device):
        data_size = len(data)
        self.data = torch.tensor(data).float().view([data_size, 5, -1]).to(device)
        self.label = torch.tensor(label).float().to(device)
        self.id = torch.tensor(id).long().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.id[index]
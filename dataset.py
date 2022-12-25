import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MyDataset(Dataset):

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def __init__(self, n, n_samples,  min_value, max_value, device='cuda:0'):

      self.device = device

      self.input = self.create_dataset(n, n_samples,  min_value, max_value)
      values = self.input.values
      # transformed = np.vstack(values).astype(np.float)
      self.input = torch.from_numpy(values).float().to(self.device)

      print("--------------------------------------")
      print(f'[INFO] Input dataset shape : {self.input.shape}')
      print(f'[INFO] Total number of observations in the dataset : {len(self.input)}')
      print("--------------------------------------")

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    # number of rows in the dataset
    def __len__(self):
        return len(self.input)

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    # get an observation by index
    def __getitem__(self, idx):

       return self.input[idx, :]

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#
         
    def create_dataset(self, n, n_samples, min_value, max_value):
        
        data_list = []
        for _ in range(n_samples):

            x = np.random.randint(low=min_value, high=max_value, size=n*2, dtype=int)
            
            dictionnary = {}
            for j in range(len(x)):
                dictionnary[j]=x[j]
            data_list.append(dictionnary)

        columns = [i for i in range(0,2*n)]
        df = pd.DataFrame(data_list,columns=columns)
        return df

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def split_train_validation(self, dataset, batch_size=8, validation_split=0.15, shuffle=True, num_workers=0):

        n_samples = len(dataset)
        indices = list(range(n_samples))

        if shuffle:
            np.random.shuffle(indices)
        
        index_split = int(np.floor(n_samples*validation_split))

        train_indices, validation_indices = indices[index_split:], indices[:index_split]

        # Init random sampler
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)

        # Init dataloader
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        validation_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)

        print(f"[INFO] using {len(train_indices)} observations for training.")
        print(f"[INFO] using {len(validation_indices)} observations for validation.")

        return train_dataloader, validation_dataloader
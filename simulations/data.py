import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from multiprocessing import  Pool
import time

class P2PDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.size(0)

    def add_data(self, feature, label):
        self.feature = torch.cat((self.feature, feature), 0)
        self.label = torch.cat((self.label, label), 0)


class P2PEnvironment(object):
    def __init__(self, config, seed):
        self.config = config
        self.n = config['points_per_iter']
        self.m = config['feature_dim']
        self.d = config['label_dim']
        self.eps = config['epsilon']
        self.eta = config['eta']
        self.KA = config['KA_norm']
        np.random.seed(seed)
        self.A = (2 * np.random.rand(self.d, self.m) - 1)
        self.A = self.A/np.linalg.norm(self.A, 1) * np.random.rand() * self.KA
        self.mu = 2 * np.random.rand(self.d) - 1
        self.mu = self.mu/np.linalg.norm(self.mu, 2) * np.random.rand()
        self.get_new_data(0)
        self.dataset = P2PDataset(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)

    def get_new_data(self, epoch_id):
        self._x = 2 * np.random.rand(self.n, self.m) - 1
        self._x = self._x / (np.tile(np.linalg.norm(self._x, 2, 1)/np.random.rand(self.n), (self.m, 1)).T)
        self._y = self.A.dot(self._x.T).T + np.random.normal(0, self.eps, self.d)
        return self._x, self._y

    def add_to_data_loader(self):
        self.dataset.add_data(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))

    def get_reward(self, action):
        ro = np.sum(self._y * action, 1)
        rb = action.dot(self.mu) + np.random.normal(0, self.eta, self.n)
        return ro, rb

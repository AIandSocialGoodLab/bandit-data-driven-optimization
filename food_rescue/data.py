import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from multiprocessing import  Pool
from sklearn import preprocessing
from utils import getTimeIntervalSigned, getNeighborGrid, getDate, calc_straight_dist_vec
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
        self.reward_scale = config['reward_scale']
        np.random.seed(seed)
        self.A = (2 * np.random.rand(self.d, self.m) - 1)
        self.A = self.A/np.linalg.norm(self.A, 1) * np.random.rand() * self.KA
        self.mu = 2 * np.random.rand(self.d) - 1
        self.mu = self.mu/np.linalg.norm(self.mu, 2) * np.random.rand()
        self.iter = 0
        if self.config['problem'] == 'toy':
            self.get_new_data(0)
            self.dataset = P2PDataset(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))
        elif self.config['problem'] == 'food_rescue':
            data_path = 'data/'
            self.u = pd.read_csv(data_path+'u.csv')
            self.r = pd.read_csv(data_path+'r.csv')
            self.r_time = pd.read_csv(data_path+'r_time.csv')
            self.rv_suc = pd.read_csv(data_path+'rv_suc.csv')
            self.rv_call = pd.read_csv(data_path+'rv_call.csv')
            self.rv_notify = pd.read_csv(data_path+'rv_notify_' + str(int(config['neg_example_time_cutoff'])) + '.csv')
            boh = self.u['created_at'].min()
            self.u = self.u.loc[:, ~self.u.columns.str.match('Unnamed')]
            self.r = self.r.loc[:, ~self.r.columns.str.match('Unnamed')]
            self.r_time = self.r_time.loc[:, ~self.r_time.columns.str.match('Unnamed')]
            self.rv_suc = self.rv_suc.loc[:, ~self.rv_suc.columns.str.match('Unnamed')]
            self.rv_call = self.rv_call.loc[:, ~self.rv_call.columns.str.match('Unnamed')]
            self.rv_notify = self.rv_notify.loc[:, ~self.rv_notify.columns.str.match('Unnamed')]
            self.features = config['features']
            self.rv_suc = self.rv_suc[self.rv_suc['user_id'].isin(self.u['user_id'])]

            self.rv_init, self.rv_main = np.split(self.rv_suc, [300])
            self.rv_init = self.rv_init.sample(n=config['init_num_data'], random_state=1)
            self.init_timestamp = self.rv_main.iloc[0]['timestamp']

            self.rv_main = self.rv_main.sample(frac=1, random_state=1)
            self.rv_valid, self.rv_main = np.split(self.rv_main, [config['valid_num_data']])

            self.u_list = self.u['user_id'].astype(int)
            self.scaler = preprocessing.StandardScaler()

            self.init_train_loader()
            self.init_valid_loader()

    def init_train_loader(self):
        rv_all = pd.concat([self.rv_init] * self.config['pos_sample_multiplier']
        + [self.rv_call[self.rv_call['rescue_id'].isin(self.rv_init['rescue_id'])]]
        + [self.rv_notify[self.rv_notify['rescue_id'].isin(self.rv_init['rescue_id'])].sample(n=len(self.rv_init) * self.config['neg_sample_multiplier'], replace=True, random_state=self.iter, axis=0)]
        , axis=0, ignore_index=True)
        rv_all = pd.merge(rv_all, self.r, how='left', left_on='rescue_id', right_on='rescue_id')
        rv_all = pd.merge(rv_all, self.u, how='left', left_on='user_id', right_on='user_id')
        rv_all = rv_all.dropna()

        rv_all = self.rv_feature_engineering(rv_all)
        rv_all = rv_all.drop(['rescue_id', 'user_id', 'timestamp_x', 'timestamp_y', 'created_at', 'donor_grid', 'recipient_grid'], axis=1)
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('user_grid_')]
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('donor_grid_')]
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('recipient_grid_')]
        rv_all = rv_all[self.features]
        rv_all_label = rv_all['claimed']
        rv_all = rv_all.drop(['claimed'], axis=1)
        self.column_order = rv_all.columns
        rv_all = self._normalize(rv_all, fit=True)
        self.dataset = P2PDataset(feature=torch.Tensor(rv_all.values), label=torch.Tensor(rv_all_label.values))

    def init_valid_loader(self):
        rv_all = pd.concat([self.rv_valid] * self.config['pos_sample_multiplier']
        + [self.rv_call[self.rv_call['rescue_id'].isin(self.rv_valid['rescue_id'])]]
        + [self.rv_notify[self.rv_notify['rescue_id'].isin(self.rv_valid['rescue_id'])].sample(n=len(self.rv_valid) * self.config['neg_sample_multiplier'], replace=True, random_state=self.iter, axis=0)]
        , axis=0, ignore_index=True)
        rv_all = pd.merge(rv_all, self.r, how='left', left_on='rescue_id', right_on='rescue_id')
        rv_all = pd.merge(rv_all, self.u, how='left', left_on='user_id', right_on='user_id')
        rv_all = rv_all.dropna()

        rv_all = self.rv_feature_engineering(rv_all)
        rv_all = rv_all.drop(['rescue_id', 'user_id', 'timestamp_x', 'timestamp_y', 'created_at', 'donor_grid', 'recipient_grid'], axis=1)
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('user_grid_')]
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('donor_grid_')]
        rv_all = rv_all.loc[:, ~rv_all.columns.str.match('recipient_grid_')]
        rv_all = rv_all[self.features]
        rv_all_label = rv_all['claimed']
        rv_all = rv_all.drop(['claimed'], axis=1)
        assert all(rv_all.columns == self.column_order)
        rv_all = self._normalize(rv_all, fit=False)
        self.valid_dataset = P2PDataset(feature=torch.Tensor(rv_all.values), label=torch.Tensor(rv_all_label.values))



    def get_new_data_fr(self):
        if self.iter >= len(self.rv_main):
            raise ValueError('end of data')
        self._x_claim = int(self.rv_main.iloc[self.iter]['user_id'])
        self._x_rescue = int(self.rv_main.iloc[self.iter]['rescue_id'])
        self._x = pd.DataFrame(np.tile(self.rv_main.iloc[self.iter].values, (self.u_list.size, 1)), columns=self.rv_main.columns)
        self._x['user_id'] = self.u_list
        self._x['claimed'] = 0
        self._x.loc[self._x['user_id'] == self._x_claim, 'claimed'] = 1
        self._x = pd.merge(self._x, self.r, how='left', left_on='rescue_id', right_on='rescue_id')
        self._x = self._x.drop(['user_id'], axis=1)
        self._x = pd.concat([self._x, self.u], axis=1)
        self._x = self.rv_feature_engineering(self._x)
        self._x = self._x.drop(['rescue_id', 'user_id', 'timestamp_x', 'timestamp_y', 'created_at', 'donor_grid', 'recipient_grid'], axis=1)
        self._x = self._x.loc[:, ~self._x.columns.str.match('user_grid_')]
        self._x = self._x.loc[:, ~self._x.columns.str.match('donor_grid_')]
        self._x = self._x.loc[:, ~self._x.columns.str.match('recipient_grid_')]

        self._x = self._x[self.features]
        self._y = self._x['claimed']
        self._x = self._x.drop(['claimed'], axis=1)
        assert all(self._x.columns == self.column_order)
        self._x = self._normalize(self._x, fit=False)


        if len(self.rv_notify[self.rv_notify['rescue_id'] == self._x_rescue]) > 0:
            self._x_train = pd.concat([self.rv_main.iloc[self.iter:self.iter+1]] * self.config['pos_sample_multiplier']
            + [self.rv_call[self.rv_call['rescue_id'] == self._x_rescue]]
            + [self.rv_notify[self.rv_notify['rescue_id'] == self._x_rescue].sample(
            n=self.config['neg_sample_multiplier'], replace=True,
            random_state=self.iter, axis=0)]
            , axis=0, ignore_index=True)
        else:
            self._x_train = pd.concat([self.rv_main.iloc[self.iter:self.iter+1]] * self.config['pos_sample_multiplier']
            + [self.rv_call[self.rv_call['rescue_id'] == self._x_rescue]]
            , axis=0, ignore_index=True)
        self._x_train = pd.merge(self._x_train, self.r, how='left', left_on='rescue_id', right_on='rescue_id')
        self._x_train = pd.merge(self._x_train, self.u, how='left', left_on='user_id', right_on='user_id')
        self._x_train = self._x_train.dropna()

        self._x_train = self.rv_feature_engineering(self._x_train)
        self._x_train = self._x_train.drop(['rescue_id', 'user_id', 'timestamp_x', 'timestamp_y', 'created_at', 'donor_grid', 'recipient_grid'], axis=1)
        self._x_train = self._x_train.loc[:, ~self._x_train.columns.str.match('user_grid_')]
        self._x_train = self._x_train.loc[:, ~self._x_train.columns.str.match('donor_grid_')]
        self._x_train = self._x_train.loc[:, ~self._x_train.columns.str.match('recipient_grid_')]
        self._x_train = self._x_train[self.features]
        self._y_train = self._x_train['claimed']
        self._x_train = self._x_train.drop(['claimed'], axis=1)
        assert all(self._x_train.columns == self.column_order)
        self._x_train = self._normalize(self._x_train, fit=False)

        return self._x, self._y

    def add_to_data_loader_fr(self):
        self.iter += 1
        self.dataset.add_data(feature=torch.Tensor(self._x_train.values), label=torch.Tensor(self._y_train.values))

    def rv_feature_engineering(self, df):
        df['dist_donor'] = calc_straight_dist_vec(df['donor_lat'], df['donor_lon'], df['latitude'], df['longitude'])
        df['reg2rescue'] = df['timestamp_x'] - df['created_at']
        df['prev_rescue_in_donor_grid'] = df.lookup(df.index, ['donor_grid_prev_'+str(int(i)) for i in df['donor_grid']])
        df['prev_rescue_in_recipient_grid'] = df.lookup(df.index, ['recipient_grid_prev_'+str(int(i)) for i in df['recipient_grid']])
        df['user_donor_same_grid'] = df.lookup(df.index, ['user_grid_'+str(int(i)) for i in df['donor_grid']])
        df['user_recipient_same_grid'] = df.lookup(df.index, ['user_grid_'+str(int(i)) for i in df['recipient_grid']])
        df['available_at_rescue'] = df.lookup(df.index, ['user_avail_'+str(int(i))+'_'+str(int(j)) for (i,j) in zip(df['day_of_week'], df['time_of_day'])])
        return df

    def _normalize(self, df, fit):
        if fit:
            self.scaler.fit(df.values)
        df = pd.DataFrame(self.scaler.transform(df.values), columns=df.columns)
        return df

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=min(len(self.dataset.feature), self.config['batch_size']), shuffle=True)

    def get_new_data(self, epoch_id):
        self._x = 2 * np.random.rand(self.n, self.m) - 1
        self._x = self._x / (np.tile(np.linalg.norm(self._x, 2, 1)/np.random.rand(self.n), (self.m, 1)).T)
        self._y = self.A.dot(self._x.T).T + np.random.normal(0, self.eps, self.d)
        return self._x, self._y


    def add_to_data_loader(self):
        self.dataset.add_data(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))


    def get_reward(self, action):
        ro = -action.dot(self._y)/self.reward_scale
        rb = action.dot(self.mu) + np.random.normal(0, self.eta, self.n)
        return ro, rb

import pandas as pd
import numpy as np
import torch
import random
from mlp import P2PEngine
from data import P2PEnvironment
import matplotlib.pyplot as plt
from time import time
import seaborn as sns




mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'learning_alg': 'OLS',
              'num_trial': 10,
              'points_per_iter': 20,
              'feature_dim': 20,
              'label_dim': 5,
              'epsilon': 1e-1,
              'eta': 1e-4,
              'KA_norm': 10,
              'delta': 1e-1,
              'num_epoch': 21,
              'learn_iter': 500,
              'batch_size': 32,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'l2_regularization': 1e-4
              }


config = mlp_config

regret_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_p2p_o = np.zeros((config['num_trial'], config['num_epoch']-1))
regret_p2p_b = np.zeros((config['num_trial'], config['num_epoch']-1))
time_p2p = np.zeros((config['num_trial'], config['num_epoch']-1))
time_bandit = np.zeros((config['num_trial'], config['num_epoch']-1))

for trial_idx in range(config['num_trial']):
    # DataLoader for training
    env = P2PEnvironment(config, seed=trial_idx * 10)
    engine_p2p = P2PEngine(env, config, pure_bandit=False)
    engine_bandit = P2PEngine(env, config, pure_bandit=True)
    start_time = time()
    for epoch in range(1, config['num_epoch']):
        data_loader = env.get_data_loader()
        test_feature, test_label = env.get_new_data(epoch)
        action_p2p, time_p2p[trial_idx, epoch-1] = engine_p2p.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
        action_bandit, time_bandit[trial_idx, epoch-1] = engine_bandit.p2p_an_epoch(data_loader, test_feature, epoch_id=epoch)
        best_action = engine_p2p.p2p_known_mu(test_label)
        ro_p2p, rb_p2p = env.get_reward(action_p2p)
        ro_bandit, rb_bandit = env.get_reward(action_bandit)
        engine_p2p.update_bandit(ro_p2p, rb_p2p)
        engine_bandit.update_bandit(ro_bandit, rb_bandit)
        env.add_to_data_loader()
        ro_best, rb_best = env.get_reward(best_action)
        regret_p2p[trial_idx, epoch-1] = ro_p2p.sum() + rb_p2p.sum() - ro_best.sum() - rb_best.sum()
        regret_p2p_o[trial_idx, epoch-1] = ro_p2p.sum() - ro_best.sum()
        regret_p2p_b[trial_idx, epoch-1] = rb_p2p.sum() - rb_best.sum()
        regret_bandit[trial_idx, epoch-1] = ro_bandit.sum() + rb_bandit.sum() - ro_best.sum() - rb_best.sum()
        if epoch == config['num_epoch'] - 1:
            end_time = time()
            print('Epoch {} starts !'.format(epoch))
            print('-' * 80)
            np.save('runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_p2p.npy'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],trial_idx),
            regret_p2p)
            np.save('runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_bandit.npy'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],trial_idx),
            regret_bandit)
            np.save('runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_p2p_o.npy'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],trial_idx),
            regret_p2p_o)
            np.save('runs/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}trial{}regret_p2p_b.npy'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
            config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],trial_idx),
            regret_p2p_b)



nbc_palette = ['#e16428', '#b42846', '#008cc3', '#00a846']
sns.palplot(sns.color_palette(nbc_palette))
fig, ax = plt.subplots()
regret_p2p_mean = (np.nancumsum(regret_p2p[:,:-1], axis=1)/range(1, config['num_epoch']-1)).mean(axis=0)
regret_bandit_mean = (np.nancumsum(regret_bandit[:,:-1], axis=1)/range(1, config['num_epoch']-1)).mean(axis=0)
regret_p2p_o_mean = (np.nancumsum(regret_p2p_o[:,:-1], axis=1)/range(1, config['num_epoch']-1)).mean(axis=0)
regret_p2p_b_mean = (np.nancumsum(regret_p2p_b[:,:-1], axis=1)/range(1, config['num_epoch']-1)).mean(axis=0)
regret_p2p_std = (np.nancumsum(regret_p2p[:,:-1], axis=1)/range(1, config['num_epoch']-1)).std(axis=0)
regret_bandit_std = (np.nancumsum(regret_bandit[:,:-1], axis=1)/range(1, config['num_epoch']-1)).std(axis=0)
regret_p2p_o_std = (np.nancumsum(regret_p2p_o[:,:-1], axis=1)/range(1, config['num_epoch']-1)).std(axis=0)
regret_p2p_b_std = (np.nancumsum(regret_p2p_b[:,:-1], axis=1)/range(1, config['num_epoch']-1)).std(axis=0)
results_mean = np.stack([regret_p2p_mean, regret_bandit_mean, regret_p2p_o_mean, regret_p2p_b_mean])
results_std = np.stack([regret_p2p_std, regret_bandit_std, regret_p2p_o_std, regret_p2p_b_std])
labels = ['PROOF', 'Vanilla OFU', 'PROOF (Optimization)', 'PROOF (Bandit)']
with sns.axes_style("darkgrid"):
    epochs = list(range(101))
    for i in range(4):
        ax.plot(range(1, config['num_epoch']-1), results_mean[i,:], label=labels[i], c=nbc_palette[i])
        ax.fill_between(range(1, config['num_epoch']-1), results_mean[i,:]-results_std[i,:], results_mean[i,:]+results_std[i,:] ,alpha=0.3, facecolor=nbc_palette[i])
    ax.legend()
fig.savefig('figures/n{}m{}d{}ka{}eta{}eps{}iter{}alg{}FINAL{}.png'.format(config['points_per_iter'], config['feature_dim'], config['label_dim'],
config['KA_norm'],config['eta'],config['epsilon'],config['learn_iter'],config['learning_alg'],config['num_trial']))

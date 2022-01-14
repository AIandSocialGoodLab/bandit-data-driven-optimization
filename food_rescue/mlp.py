import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LinearRegression
from utils import use_optimizer
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from time import time
from multiprocessing import  Pool

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, x):
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = torch.nn.ReLU()(x)
        logits = self.affine_output(x)
        rating = self.logistic(logits)
        return rating


class P2PEngine(object):
    def __init__(self, env, config, pure_bandit):
        self.config = config
        self.pure_bandit = pure_bandit
        if config['learning_alg'] == 'NN':
            self.learn_iter = config['learn_iter']
        elif config['learning_alg'] == 'OLS':
            self.predictor = LinearRegression(fit_intercept=False, n_jobs=-1)
        else:
            assert 1 == 0
        self.n = config['points_per_iter']
        self.m = config['feature_dim']
        self.d = config['label_dim']
        self.delta = config['delta']
        self.w = np.zeros((self.n, self.d))
        self.A = np.tile(np.eye(self.d), (self.n, 1, 1))
        self.mu_hat = np.zeros((self.n, self.d))
        self.rw = np.zeros((self.n, self.d))
        self.beta = max(128 * self.d * np.log(1) * np.log(1/self.delta), np.square(8/3 * np.log(1/self.delta)))
        self.mu = env.mu
        self.w_known = np.zeros((self.n, self.d))
        self.valid_dataset = env.valid_dataset


    def p2p_an_epoch(self, data_loader, test_feature, epoch_id):
        self.beta = max(128 * self.d * np.log(epoch_id) * np.log(epoch_id*epoch_id/self.delta),
        np.square(8/3 * np.log(epoch_id*epoch_id/self.delta)))
        self.epoch_id = epoch_id
        start_time = time()
        if not self.pure_bandit:
            self.learn(data_loader, epoch_id)
        self.optimize(test_feature)
        end_time = time()
        return self.w, end_time - start_time

    def learn(self, data_loader, epoch_id):
        if self.config['learning_alg'] == 'NN':
            self.predictor = MLP(self.config)
            self.opt = use_optimizer(self.predictor, self.config)
            self.crit = torch.nn.BCELoss()
            total_loss= np.zeros(self.learn_iter)
            valid_loss= np.zeros(self.learn_iter)
            min_valid_loss_mavg = float('inf')
            for iter in range(self.learn_iter):
                ex_seen = 0
                self.predictor.train()
                for batch_id, batch in enumerate(data_loader):
                    x, y = batch[0], batch[1]
                    loss = self.train_single_batch(x, y)
                    total_loss[iter] += loss
                    ex_seen += len(x)
                    if ex_seen > len(data_loader.dataset.feature)/self.config['pos_sample_multiplier']:
                        break
                self.predictor.eval()
                valid_loss[iter] = self.eval_single_batch(self.valid_dataset.feature, self.valid_dataset.label)
                if iter > 3:
                    valid_loss_mavg = np.mean(valid_loss[iter-3:iter+1])
                    if valid_loss_mavg < min_valid_loss_mavg:
                        min_valid_loss_mavg = valid_loss_mavg
                    elif iter > 4:
                        break
        elif self.config['learning_alg'] == 'OLS':
            self.predictor.fit(data_loader.dataset.feature, data_loader.dataset.label)

    def optimize(self, test_feature):
        n_cores = 4
        lst = np.arange(self.n)
        n_lst = int(len(lst)/n_cores)
        test_features_split = [test_feature[i:i + n_lst, :] for i in range(0, len(lst), n_lst)]
        pool = Pool(len(test_features_split))
        w_list = pool.map(self.helper_optimize, test_features_split)
        pool.close()
        pool.join()
        self.w = np.concatenate(w_list, axis=0)


    def helper_optimize(self, test_feature):
        solver = pyomo.opt.SolverFactory('ipopt')
        solver.options['print_level'] = 5
        solver.options['max_cpu_time'] = int(100)
        solver.options['warm_start_init_point'] = 'yes'
        w = -np.inf * np.ones((test_feature.shape[0], self.d))
        for i in range(test_feature.shape[0]):
            x = test_feature[i,:]
            if self.pure_bandit:
                c_hat = np.zeros(self.d)
            else:
                if self.config['learning_alg'] == 'NN':
                    self.predictor.eval()
                    c_hat = self.predictor(torch.Tensor(x)).detach().numpy()
                elif self.config['learning_alg'] == 'OLS':
                    c_hat = self.predictor.predict(x.reshape(1, -1)).squeeze()

            w[i,:] = self.solve_optimization(solver, c_hat)
        return w

    def solve_optimization(self, solver, c_hat):
        model = ConcreteModel()
        model.dSet = Set(initialize=range(self.d))
        model.w = Var(model.dSet)
        model.nu = Var(model.dSet)
        for j in range(self.d):
            model.w[j].value = (2*np.random.rand() - 1)/np.sqrt(self.d)
            model.nu[j].value = (2*np.random.rand() - 1)/np.sqrt(self.d)

        model.obj = Objective(expr=sum((c_hat[j] + model.nu[j]) * model.w[j] for j in range(self.d)), sense=minimize)
        model.w_constraint = Constraint(expr = sum(model.w[j] * model.w[j] for j in range(self.d)) <= 1)
        expr1 = sum(self.A[i,j,k] * (model.nu[j] - self.mu_hat[i,j]) *
        (model.nu[k] - self.mu_hat[i,k]) for j in range(self.d) for k in range(self.d))
        model.nu_constraint = Constraint(expr= expr1 <= 1)
        try:
            result = solver.solve(model, tee = False, keepfiles = False)
        except ValueError:
            w = -self.mu_hat[i,:]/np.linalg.norm(self.mu_hat[i,:], 2)
        if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
            self.feasible = True
            w = [value(model.w[j]) for j in range(self.d)]
        else:
            self.feasible = False
            print('Encountered an attribute error')
        return w

    def update_bandit(self, ro, rb):
        if self.pure_bandit:
            rb = ro + rb
        for i in range(self.n):
            self.rw[i,:] += rb[i] * self.w[i,:]
            self.A[i,:,:] = self.A[i,:,:] + np.outer(self.w[i,:], self.w[i,:])
            self.mu_hat[i,:] = np.matmul(np.linalg.inv(self.A[i,:,:]), self.rw[i,:])


    def p2p_known_mu(self, test_label):
        for i in range(self.n):
            lp = gp.Model("lp" + str(i))
            c = test_label[i,:].squeeze()
            w = lp.addMVar(shape=self.d, lb=-GRB.INFINITY, name="w")
            lp.setObjective((c + self.mu) @ w, GRB.MINIMIZE)
            lp.addConstr(w @ w <= 1, name="norm")
            lp.Params.OutputFlag = 0
            try:
                lp.optimize()
                self.w_known[i,:] = w.X
            except gp.GurobiError as e:
                pass
            except AttributeError:
                print('Encountered an attribute error')
        return self.w_known


    def train_single_batch(self, x, y):
        self.opt.zero_grad()
        pred = self.predictor(x)
        loss = self.crit(pred.squeeze(), y.squeeze())
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss


    def eval_single_batch(self, x, y):
        self.opt.zero_grad()
        pred = self.predictor(x)
        loss = self.crit(pred.squeeze(), y.squeeze())
        loss = loss.item()
        return loss

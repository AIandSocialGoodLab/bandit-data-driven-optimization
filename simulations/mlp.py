import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LinearRegression
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
from time import time
from multiprocessing import  Pool


class P2PEngine(object):
    def __init__(self, env, config, pure_bandit):
        self.config = config
        self.pure_bandit = pure_bandit
        self.predictor = LinearRegression(fit_intercept=False, n_jobs=-1)
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


    def p2p_an_epoch(self, data_loader, test_feature, epoch_id):
        self.beta = max(128 * self.d * np.log(epoch_id) * np.log(epoch_id*epoch_id/self.delta),
        np.square(8/3 * np.log(epoch_id*epoch_id/self.delta)))
        start_time = time()
        if not self.pure_bandit:
            self.learn(data_loader, epoch_id)
        self.optimize(test_feature)
        end_time = time()
        return self.w, end_time - start_time

    def learn(self, data_loader, epoch_id):
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
                c_hat = self.predictor.predict(x.reshape(1, -1)).squeeze()

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
                w[i,:] = -self.mu_hat[i,:]/np.linalg.norm(self.mu_hat[i,:], 2)
                continue
            if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
                self.feasible = True
                w[i,:] = [value(model.w[j]) for j in range(self.d)]
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

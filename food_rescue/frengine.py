from mlp import P2PEngine
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import sqrtm
import numpy as np
import torch

class FREngine(P2PEngine):
    def __init__(self, env, config, pure_bandit):
        super(FREngine, self).__init__(env, config, pure_bandit)

    def optimize(self, test_feature):
        if self.pure_bandit:
            c_hat = np.zeros(self.d)
        else:
            self.predictor.eval()
            c_hat = self.predictor(torch.Tensor(test_feature.values)).detach().numpy()

        c_hat = c_hat.squeeze()
        mu_hat = self.mu_hat.squeeze()

        w = cp.Variable(self.d, boolean=True)
        nu = cp.Variable(self.d)
        z = cp.Variable(self.d) # auxiliary variable
        objective = cp.Minimize(-c_hat @ w.T / (self.config['reward_scale'] * self.config['p2p_opt_scale']) + sum(z))
        constr1 = sum(w) <= self.config['top_k']
        constr2 = cp.quad_form(nu - mu_hat, self.A[0,:,:])<= max(5 - self.epoch_id, np.power(5.0, -(self.epoch_id - 5) / 5))
        constrz = [z >= -w, z <= w, z >= nu - 1 + w, z <= nu + 1 - w]
        lp = cp.Problem(objective, [constr1, constr2] + constrz)
        try:
            lp.solve(solver=cp.GUROBI, verbose=False, TimeLimit=10, MIPGap=1e-2)
            self.w = np.reshape(w.value, (1, -1))
        except (KeyError, cp.SolverError) as e:
            argmax = np.argpartition(c_hat - mu_hat, -self.config['top_k'])[-self.config['top_k']:]
            self.w = np.zeros(self.d)
            self.w[argmax] = 1
            self.w = np.reshape(self.w, (1, -1))

    def p2p_known_mu(self, test_label):
        lp = gp.Model("lp")
        w = cp.Variable(test_label.shape, boolean=True)
        objective = cp.Minimize((-test_label.values / self.config['reward_scale'] + self.mu) @ w)
        constr = sum(w) <= self.config['top_k']
        lp = cp.Problem(objective, [constr])
        lp.solve(solver=cp.GUROBI, verbose=False, TimeLimit=10)
        self.w_known = np.reshape(w.value, (1, -1))
        return self.w_known

#!/usr/bin/env python3

import sympy as sy
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


class HGDSolver:
    def __init__(self, mu=1, Ab_path='./../../data/'):
        #load A and b matrices, cast as sympy matrices
        A = np.loadtxt(Ab_path + 'A.txt')
        b = np.loadtxt(Ab_path + 'b.txt')
        self.A = sy.Matrix(A)
        self.b = sy.Matrix(b)
        self.mu = mu
        self.construct_H()
        self.construct_grad_H()

    def construct_H(self):
        #initialize variables (currently assuming only a single contact!)
        gamma1, gamma2, gamma3, s1, s2, lam1, lam2 = sy.symbols('gamma_1, gamma_2, gamma_3, s_1, s_2, lambda_1, lambda_2')
        gamma = sy.Matrix([gamma1, gamma2, gamma3])
        self.variables = [gamma1, gamma2, gamma3, s1, s2, lam1, lam2]

        #construct Hamiltonian using sympy,
        f = sy.Rational(1, 2)*gamma.T@self.A@gamma + self.b.T@gamma
        L = f[0] + lam1*(self.mu**2*gamma1**2 - gamma2**2 - gamma3**2 - s1**2) + lam2*(self.mu*gamma1 - s2**2)
        grad_L = [sy.diff(L, self.variables[i]) for i in range(len(self.variables))]
        self.H = sy.Rational(1, 2)*sum([entry**2 for entry in grad_L])
        self.H_fun = sy.lambdify([self.variables], self.H)

    def construct_grad_H(self):
        #get the Jacobian of H, then create function version
        self.grad_H = sy.Matrix([sy.diff(self.H, self.variables[i]) for i in range(len(self.variables))]).T
        def grad_H_fun(x):
            jac = sy.lambdify([self.variables], self.grad_H)
            return jac(x).reshape(7)
        self.grad_H_fun = grad_H_fun

    def solve(self, x0=None):
        if x0 is None:
            gamma_n0 = np.random.uniform(0, 1)
            scale = np.random.uniform(-1, 1, 2)
            gamma0 = np.array([gamma_n0, scale[0]*np.sqrt(gamma_n0**2/2), scale[0]*np.sqrt(gamma_n0**2/2)])
            s0 = np.random.uniform(0, 1, 2)
            lam0 = -np.random.uniform(0, 1, 2)
            x0 = np.concatenate((gamma0, s0, lam0))
        sol = minimize(self.H_fun, x0, method='BFGS', jac=self.grad_H_fun)
        return sol


if __name__ == '__main__':
    solver = HGDSolver()
    print(solver.solve())

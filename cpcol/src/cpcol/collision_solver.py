import numpy as np
import sympy as sy
from scipy.optimize import minimize


def apgd(A, b, x, project, computeResidual, config):
    maxIter = config['maxIter']
    residualTol = config['residualTol']
    stepSizeTol = config['stepSizeTol']
    logFileName = config['logFileName']

    with open(logFileName, 'w') as fp:
        fp.write('{:>9s} {:>11s} {:>12s} {:>12s} {:>12s}\n'.format(
                'iteration', 'matVecCount', 'tk', 'gradientNorm', 'residual'))

        matVecCount = 0

        tk = 1.0
        xkm2 = np.copy(x)
        xkm1 = np.copy(x)

        for k in range(1, maxIter + 1):
            alpha = (k - 2.0) / (k + 1.0)
            v = xkm1 + alpha * (xkm1 - xkm2)

            Av = A @ v
            matVecCount += 1

            f = np.dot(v, 0.5 * Av + b)
            g = Av + b

            while tk >= stepSizeTol:
                xk = project(v - tk * g)
                dx = xk - v

                Axk = A @ xk
                matVecCount += 1

                fk = np.dot(xk, 0.5 * Axk + b)

                if fk <= f + np.dot(g, dx) + 0.5 * np.linalg.norm(dx)**2 / tk:
                    break

                tk *= 0.9

            if tk < stepSizeTol:
                raise RuntimeError('APGD step size is too small')

            gk = Axk + b
            residual = computeResidual(xk, gk)

            fp.write('{:>9d} {:>11d} {:>12.6e} {:>12.6e} {:>12.6e}\n'.format(
                k, matVecCount, tk, np.linalg.norm(gk), residual))

            xkm2, xkm1 = xkm1, xk

            if residual < residualTol:
                break

    return xkm1


def apgdLcp(A, b, config):
    def project(x):
        return np.maximum(x, np.zeros_like(x))

    def computeResidual(x, g):
        return max(0.0, np.abs(x * g).max(), -x.min(), -g.min())

    return apgd(A, b, np.ones_like(b), project, computeResidual, config)


def apgdCcp(A, b, mu, config):
    assert b.shape[0] % 3 == 0

    n = b.shape[0] // 3

    def project(x):
        for i in range(n):
            xn = x[3 * i]

            if xn < 1.0e-15:
                x[3 * i] = 0.0
                x[3 * i + 1] = 0.0
                x[3 * i + 2] = 0.0
                continue

            xt = np.hypot(x[3 * i + 1], x[3 * i + 2])

            if xt > mu * xn:
                xn = (mu * xt + xn) / (mu**2 + 1)
                c = mu * xn / xt

                x[3 * i] = xn
                x[3 * i + 1] *= c
                x[3 * i + 2] *= c

        return x

    def computeResidual(x, g):
        residual = 0.0

        for i in range(n):
            xn = x[3 * i]
            x1 = x[3 * i + 1]
            x2 = x[3 * i + 2]

            gn = g[3 * i]
            g1 = g[3 * i + 1]
            g2 = g[3 * i + 2]

            residual = max(residual,
                           np.hypot(x1, x2) - xn * mu,
                           np.hypot(g1, g2) - gn / mu,
                           np.abs(xn * gn + x1 * g1 + x2 * g2))

        return residual

    return apgd(A, b, np.ones_like(b), project, computeResidual, config)


def hgdCcp(A, b, mu, config):
    maxIter = config['maxIter']

    #initialize variables (currently assuming only a single contact!)
    gamma1, gamma2, gamma3, s1, s2, lam1, lam2 = sy.symbols('gamma_1, gamma_2, gamma_3, s_1, s_2, lambda_1, lambda_2')
    gamma = sy.Matrix([gamma1, gamma2, gamma3])
    variables = [gamma1, gamma2, gamma3, s1, s2, lam1, lam2]
    A = sy.Matrix(A)
    b = sy.Matrix(b)

    #construct Hamiltonian using sympy,
    f = sy.Rational(1, 2)*gamma.T@A@gamma + b.T@gamma
    L = f[0] + lam1*(mu**2*gamma1**2 - gamma2**2 - gamma3**2 - s1**2) + lam2*(mu*gamma1 - s2**2)
    grad_L = [sy.diff(L, variables[i]) for i in range(len(variables))]
    H = sy.Rational(1, 2)*sum([entry**2 for entry in grad_L])
    H_fun = sy.lambdify([variables], H)

    #get the Jacobian of H, then create function version
    grad_H = sy.Matrix([sy.diff(H, variables[i]) for i in range(len(variables))]).T
    def grad_H_fun(x):
        jac = sy.lambdify([variables], grad_H)
        return jac(x).reshape(7)
    grad_H_fun = grad_H_fun

    #generate random initial guess and solve
    gamma_n0 = np.random.uniform(0, 1)
    scale = np.random.uniform(-1, 1, 2)
    gamma0 = np.array([gamma_n0, scale[0]*np.sqrt(gamma_n0**2/2), scale[0]*np.sqrt(gamma_n0**2/2)])
    s0 = np.random.uniform(0, 1, 2)
    lam0 = -np.random.uniform(0, 1, 2)
    x0 = np.concatenate((gamma0, s0, lam0))
    sol = minimize(H_fun, x0, method='BFGS', jac=grad_H_fun, options={'maxiter': maxIter})
    return sol.x[:3]


def solveLcp(A, b, config):
    if config['name'] == 'apgd':
        return apgdLcp(A, b, config)

    raise ValueError('unknown collision solver "' + config['name'] + '"')


def solveCcp(A, b, mu, config):
    if config['name'] == 'apgd':
        return apgdCcp(A, b, mu, config)
    elif config['name'] == 'hgd':
        return hgdCcp(A, b, mu, config)

    raise ValueError('unknown collision solver "' + config['name'] + '"')

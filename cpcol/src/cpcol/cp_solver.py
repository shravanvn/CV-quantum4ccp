import os

import numpy as np


def residualLcp(x, g):
    return max(0.0, np.abs(x * g).max(), -x.min(), -g.min())


def projectLcp(x):
    return np.maximum(x, np.zeros_like(x))


def residualCcp(x, g, mu):
    n = x.shape[0] // 3
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


def projectCcp(x, mu):
    n = x.shape[0] // 3

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


class CpSolver(object):
    def __init__(self):
        pass

    def solveLcp(self, A, b, uid=None):
        raise NotImplementedError

    def solveCcp(self, A, b, mu, uid=None):
        raise NotImplementedError


class ApgdCpSolver(CpSolver):
    def __init__(self, maxIter, residualTol, stepSizeTol, logDir):
        os.makedirs(logDir, exist_ok=True)

        self.maxIter = maxIter
        self.residualTol = residualTol
        self.stepSizeTol = stepSizeTol
        self.logDir = logDir

    def solve(self, A, b, project, residual, uid):
        if uid is None:
            suffix = ''
        else:
            suffix = '_{:>06s}'.format(str(uid))

        logFileName = os.path.join(self.logDir, 'apgd' + suffix + '.txt')

        with open(logFileName, 'w') as fp:
            fp.write('{:>9s} {:>12s} {:>12s} {:>11s}\n'.format(
                'iteration', 'stepSize', 'residual', 'matVecCount'))

            tk = 1.0
            xkm2 = np.ones_like(b)
            xkm1 = np.copy(xkm2)
            matVecCount = 0

            for k in range(1, self.maxIter + 1):
                alpha = (k - 2.0) / (k + 1.0)
                v = xkm1 + alpha * (xkm1 - xkm2)

                Av = A @ v
                matVecCount += 1

                f = np.dot(v, 0.5 * Av + b)
                g = Av + b

                while tk >= self.stepSizeTol:
                    xk = project(v - tk * g)
                    dx = xk - v

                    Axk = A @ xk
                    matVecCount += 1

                    fk = np.dot(xk, 0.5 * Axk + b)

                    if f + np.dot(g, dx) + 0.5 * np.linalg.norm(dx)**2 / tk \
                       >= fk:
                        break

                    tk *= 0.9

                if tk < self.stepSizeTol:
                    fp.write('{:>9d} {:>12.6e} {:>12s} {:>11d}\n'.format(
                        k, tk, '', matVecCount))
                    break

                gk = Axk + b
                res = residual(xk, gk)

                fp.write('{:>9d} {:>12.6e} {:>12.6e} {:>11d}\n'.format(
                    k, tk, res, matVecCount))

                xkm2, xkm1 = xkm1, xk

                if res < self.residualTol:
                    break

        return xkm1

    def solveLcp(self, A, b, uid=None):
        return self.solve(A, b, projectLcp, residualLcp, uid)

    def solveCcp(self, A, b, mu, uid=None):
        return self.solve(A, b, projectCcp, lambda x, g: residualCcp(x, g, mu),
                          uid)


class MinMapNewtonCpSolver(CpSolver):
    def __init__(self, maxIter, residualTol, stepSizeTol, saveHessian, logDir):
        os.makedirs(logDir, exist_ok=True)

        self.maxIter = maxIter
        self.residualTol = residualTol
        self.stepSizeTol = stepSizeTol
        self.saveHessian = saveHessian
        self.logDir = logDir
        if saveHessian:
            self.hessianSizeFile = os.path.join(os.path.dirname(logDir),
                                                'hessian_sizes.txt')
            open(self.hessianSizeFile, 'w').close()

    def linear_solve(self, prefix, k, A, b):
        x = np.linalg.solve(A, b)
        if self.saveHessian:
            prefixIter = prefix + '_hessian_{:>06d}'.format(k)
            with open(self.hessianSizeFile, 'a') as fp:
                fp.write('{:s} {:d}\n'.format(os.path.basename(prefixIter),
                                              x.size))
            np.savetxt(prefixIter + '_A.txt', A)
            np.savetxt(prefixIter + '_b.txt', b)
            np.savetxt(prefixIter + '_x.txt', x)
        return x

    def solveLcp(self, A, b, uid=None):
        prefix = os.path.join(self.logDir, 'mmnewton')
        if uid is not None:
            prefix += '_{:>06s}'.format(str(uid))

        with open(prefix + '.txt', 'w') as fp:
            fp.write('{:>9s} {:>12s} {:>12s} {:>11s} {:>10s}\n'.format(
                'iteration', 'residual', 'stepSize', 'matVecCount',
                'solveCount'))

            x = np.ones_like(b)
            matVecCount = 0
            solveCount = 0
            for k in range(1, self.maxIter + 1):
                g = A @ x + b
                matVecCount += 1

                res = residualLcp(x, g)
                if res < self.residualTol:
                    fp.write(
                        '{:>9d} {:>12.6e} {:>12s} {:>11d} {:>10d}\n'.format(
                            k, res, '', matVecCount, solveCount))
                    break

                phi = np.minimum(x, g)

                indsA = (g < x)
                indsB = (g >= x)

                if all(indsA):
                    dx = self.linear_solve(prefix, k, A, -phi)
                    solveCount += 1
                elif all(indsB):
                    dx = -phi
                else:
                    dx = np.zeros_like(x)
                    dx[indsA] = self.linear_solve(
                        prefix, k, A[indsA, :][:, indsA],
                        A[indsA, :][:, indsB] @ phi[indsB] - phi[indsA])
                    dx[indsB] = -phi[indsB]
                    matVecCount += 1
                    solveCount += 1

                stepSize = np.linalg.norm(dx)
                fp.write('{:>9d} {:>12.6e} {:>12.6e} {:>11d} {:>10d}\n'.format(
                    k, res, stepSize, matVecCount, solveCount))

                if stepSize < self.stepSizeTol:
                    break

                x += dx

        return x


def createCpSolver(config, root_dir):
    if config['name'] == 'apgd':
        maxIter = config['max_iter']
        residualTol = config['residual_tol']
        stepSizeTol = config['step_size_tol']
        logDir = os.path.join(root_dir, config['log_dir'])
        return ApgdCpSolver(maxIter, residualTol, stepSizeTol, logDir)

    if config['name'] == 'mmnewton':
        maxIter = config['max_iter']
        residualTol = config['residual_tol']
        stepSizeTol = config['step_size_tol']
        saveHessian = config['save_hessian']
        logDir = os.path.join(root_dir, config['log_dir'])
        return MinMapNewtonCpSolver(maxIter, residualTol, stepSizeTol,
                                    saveHessian, logDir)

    raise ValueError('unknown solver name')

import numpy as np


def computeTangentDirections(normal):
    # determine an orthogonal direction to normal via Gram-Schmidt with one of
    # the standard 3D basis vectors
    if np.abs(normal[0]) > 1 / np.sqrt(3):
        # normal is dominantly 'incident' towards (1, 0, 0)
        # try tangent direction towards (0, 1, 0) for numerical stability
        tangent1 = np.array([0.0, 1.0, 0.0])
    else:
        tangent1 = np.array([1.0, 0.0, 0.0])

    tangent1 -= np.dot(normal, tangent1) * normal
    tangent1 /= np.linalg.norm(tangent1)

    # third orthogonal direction is cross product of above two
    tangent2 = np.cross(normal, tangent1)

    return tangent1, tangent2


class FrictionModel(object):
    def __init__(self):
        pass

    def computeDirectionMatrix(self, numObject, colPairs, colNormals):
        raise NotImplementedError

    def computeComplementarityProblem(self, MInv, vKnown, seps=None):
        raise NotImplementedError

    def computeContactForce(self, stepInd, solver):
        raise NotImplementedError


class NoFrictionModel(FrictionModel):
    def __init__(self):
        super().__init__()

    def computeDirectionMatrix(self, numObj, colPairs, colNormals):
        numCol = colPairs.shape[0]

        self.Dn = np.zeros((3 * numObj, numCol), dtype=colNormals.dtype)
        for k in range(numCol):
            i, j = colPairs[k, :]

            if i >= 0:
                iBeg, iEnd = 3 * i, 3 * i + 3
                self.Dn[iBeg:iEnd, k] = -colNormals[k, :]

            jBeg, jEnd = 3 * j, 3 * j + 3
            self.Dn[jBeg:jEnd, k] = colNormals[k, :]

    def computeComplementarityProblem(self, MInv, vKnown, seps=None):
        self.A = self.Dn.T @ MInv @ self.Dn
        self.b = self.Dn.T @ vKnown

        if seps is not None:
            self.b += seps

    def computeContactForce(self, stepInd, solver):
        gamma = solver.solveLcp(self.A, self.b, stepInd)
        return self.Dn @ gamma


class LinearizedFrictionModel(FrictionModel):
    def __init__(self, numEdge, frictionCoefficient):
        super().__init__()
        self.numEdge = numEdge
        self.frictionCoefficient = frictionCoefficient

    def computeDirectionMatrix(self, numObj, colPairs, colNormals):
        self.numCol = colPairs.shape[0]

        dTheta = 2.0 * np.pi / self.numEdge
        theta = dTheta * np.arange(self.numEdge)

        refDt = np.stack((np.cos(theta), np.sin(theta)))

        self.Dn = np.zeros((3 * numObj, self.numCol))
        self.Dt = np.zeros((3 * numObj, self.numEdge * self.numCol))
        for k in range(self.numCol):
            i, j = colPairs[k, :]

            tangent1, tangent2 = computeTangentDirections(colNormals[k, :])
            localDt = np.stack((tangent1, tangent2)).T @ refDt

            if i >= 0:
                iBeg, iEnd = 3 * i, 3 * i + 3
                self.Dn[iBeg:iEnd, k] = -colNormals[k, :]
                for s in range(self.numEdge):
                    self.Dt[iBeg:iEnd, self.numEdge * k + s] = -localDt[:, s]

            jBeg, jEnd = 3 * j, 3 * j + 3
            self.Dn[jBeg:jEnd, k] = colNormals[k, :]
            for s in range(self.numEdge):
                self.Dt[jBeg:jEnd, self.numEdge * k + s] = localDt[:, s]

    def computeComplementarityProblem(self, MInv, vKnown, seps=None):
        E = np.kron(np.eye(self.numCol), np.ones((self.numEdge, 1)))

        self.A = np.block([[self.Dn.T @ MInv @ self.Dn,
                            self.Dn.T @ MInv @ self.Dt,
                            np.zeros((self.numCol, self.numCol))],
                           [self.Dt.T @ MInv @ self.Dn,
                            self.Dt.T @ MInv @ self.Dt,
                            E],
                           [self.frictionCoefficient * np.eye(self.numCol),
                            -E.T,
                            np.zeros((self.numCol, self.numCol))]])
        self.b = np.hstack((self.Dn.T @ vKnown,
                            self.Dt.T @ vKnown,
                            np.zeros(self.numCol)))

        if seps is not None:
            self.b[0:self.numCol] += seps

    def computeContactForce(self, stepInd, solver):
        x = solver.solveLcp(self.A, self.b, stepInd)
        gamma = x[0:self.numCol]
        beta = x[self.numCol:(self.numEdge + 1) * self.numCol]
        return self.Dn @ gamma + self.Dt @ beta


class QuadraticFrictionModel(FrictionModel):
    def __init__(self, frictionCoefficient):
        super().__init__()
        self.frictionCoefficient = frictionCoefficient

    def computeDirectionMatrix(self, numObj, colPairs, colNormals):
        numCol = colPairs.shape[0]

        self.D = np.zeros((3 * numObj, 3 * numCol))
        for k in range(numCol):
            i, j = colPairs[k, :]

            tangent1, tangent2 = computeTangentDirections(colNormals[k, :])

            if i >= 0:
                iBeg, iEnd = 3 * i, 3 * i + 3
                self.D[iBeg:iEnd, 3 * k] = -colNormals[k, :]
                self.D[iBeg:iEnd, 3 * k + 1] = -tangent1
                self.D[iBeg:iEnd, 3 * k + 2] = -tangent2

            jBeg, jEnd = 3 * j, 3 * j + 3
            self.D[jBeg:jEnd, 3 * k] = colNormals[k, :]
            self.D[jBeg:jEnd, 3 * k + 1] = tangent1
            self.D[jBeg:jEnd, 3 * k + 2] = tangent2

    def computeComplementarityProblem(self, MInv, vKnown, seps=None):
        self.A = self.D.T @ MInv @ self.D
        self.b = self.D.T @ vKnown

        if seps is not None:
            self.b[::3] += seps

    def computeContactForce(self, stepInd, solver):
        gamma = solver.solveCcp(self.A, self.b, self.frictionCoefficient,
                                stepInd)
        return self.D @ gamma


def createFrictionModel(config):
    if config['name'] == 'none':
        return NoFrictionModel()

    if config['name'] == 'linearized':
        numEdge = config['num_edge']
        frictionCoefficient = config['friction_coefficient']
        return LinearizedFrictionModel(numEdge, frictionCoefficient)

    if config['name'] == 'quadratic':
        frictionCoefficient = config['friction_coefficient']
        return QuadraticFrictionModel(frictionCoefficient)

    raise ValueError('unknown friction model name')

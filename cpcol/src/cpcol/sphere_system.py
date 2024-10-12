import os

import numpy as np
from vtk import vtkPoints, vtkDoubleArray, vtkPolyData, vtkXMLPolyDataWriter

from .collision_solver import solveLcp, solveCcp


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


class SphereSystem(object):
    def __init__(self, initConfigFile, shellRadius, stepSize=1.0e-02,
                 accelerationDueToGravity=10.0, collisionBuffer=0.1):
        self.readConfig(initConfigFile)
        self.shellRadius = shellRadius
        self.stepSize = stepSize
        self.collisionBuffer = collisionBuffer
        self.accelerationDueToGravity = accelerationDueToGravity
        self.computeInverseMassMatrix()
        self.vKnown = np.zeros((3 * self.nSphere,))
        self.forceCol = np.zeros((3 * self.nSphere,))
        self.stepInd = 0
        self.outputInd = 0

    def readConfig(self, configFile):
        data = np.loadtxt(configFile)
        self.nSphere = data.shape[0]
        self.mass = data[:, 0]
        self.radius = data[:, 1]
        self.position = data[:, 2:5].flatten()
        self.velocity = data[:, 5:8].flatten()

    def computeInverseMassMatrix(self):
        self.MInv = np.diag(np.kron(1.0 / self.mass,
                                    np.array([1.0, 1.0, 1.0])))

    def computeVelocityKnown(self):
        self.vKnown = self.velocity + np.kron(
            np.ones(self.nSphere),
            np.array([
                0.0,
                0.0,
                -self.stepSize * self.accelerationDueToGravity
            ])
        )

    def detectCollisions(self):
        collisionPairs = []
        collisionNormals = []
        collisionSeparations = []
        for jSphere in range(self.nSphere):
            jStart = 3 * jSphere
            jStop = 3 * jSphere + 3
            vec = -self.position[jStart:jStop]
            nrm = np.linalg.norm(vec)
            sep = self.shellRadius - nrm - self.radius[jSphere]
            if sep < self.collisionBuffer * self.radius[jSphere]:
                collisionPairs.append([-1, jSphere])
                collisionNormals.append(vec / nrm)
                collisionSeparations.append(sep)
        for iSphere in range(self.nSphere):
            iStart = 3 * iSphere
            iStop = 3 * iSphere + 3
            for jSphere in range(iSphere + 1, self.nSphere):
                jStart = 3 * jSphere
                jStop = 3 * jSphere + 3
                vec = self.position[jStart:jStop] - self.position[iStart:iStop]
                nrm = np.linalg.norm(vec)
                sep = nrm - self.radius[iSphere] - self.radius[jSphere]
                if sep < 0.5 * self.collisionBuffer * (self.radius[iSphere] +
                                                       self.radius[jSphere]):
                    collisionPairs.append([iSphere, jSphere])
                    collisionNormals.append(vec / nrm)
                    collisionSeparations.append(sep)
        self.nCollision = len(collisionPairs)
        self.collisionPairs = np.array(collisionPairs)
        self.collisionNormals = np.array(collisionNormals)
        self.collisionSeparations = np.array(collisionSeparations)

    def computeDirectionMatrix(self):
        raise NotImplementedError

    def setupComplementarityProblem(self):
        raise NotImplementedError

    def computeContactForce(self, A, b):
        raise NotImplementedError

    def step(self, solverConfig):
        self.stepInd += 1
        solverConfig['logFileName'] = \
            os.path.join(solverConfig['outputDir'],
                         'stats_{:06d}.txt'.format(self.stepInd))

        self.forceCol.fill(0.0)
        self.computeVelocityKnown()
        self.detectCollisions()
        if self.nCollision > 0:
            self.computeDirectionMatrix()
            A, b = self.setupComplementarityProblem()
            self.computeContactForce(A, b, solverConfig)
            self.velocity = self.vKnown + self.MInv @ self.forceCol
        else:
            self.velocity = self.vKnown
        self.position += self.stepSize * self.velocity

    def output(self, dataDir):
        os.makedirs(dataDir, exist_ok=True)

        self.outputInd += 1
        file_name = os.path.join(dataDir,
                                 'snapshot_{:06d}.vtp'.format(self.outputInd))

        points = vtkPoints()
        mass = vtkDoubleArray()
        radius = vtkDoubleArray()
        position = vtkDoubleArray()
        velocity = vtkDoubleArray()
        forceCol = vtkDoubleArray()

        position.SetNumberOfComponents(3)
        velocity.SetNumberOfComponents(3)
        forceCol.SetNumberOfComponents(3)

        mass.SetName("mass")
        radius.SetName("radius")
        position.SetName("position")
        velocity.SetName("velocity")
        forceCol.SetName("forceCol")

        for i in range(self.nSphere):
            points.InsertNextPoint(self.position[3 * i],
                                   self.position[3 * i + 1],
                                   self.position[3 * i + 2])
            mass.InsertNextValue(self.mass[i])
            radius.InsertNextValue(self.radius[i])
            position.InsertNextTuple3(self.position[3 * i],
                                      self.position[3 * i + 1],
                                      self.position[3 * i + 2])
            velocity.InsertNextTuple3(self.velocity[3 * i],
                                      self.velocity[3 * i + 1],
                                      self.velocity[3 * i + 2])
            forceCol.InsertNextTuple3(self.forceCol[3 * i],
                                      self.forceCol[3 * i + 1],
                                      self.forceCol[3 * i + 2])

        data = vtkPolyData()
        data.SetPoints(points)
        data.GetPointData().AddArray(mass)
        data.GetPointData().AddArray(radius)
        data.GetPointData().AddArray(position)
        data.GetPointData().AddArray(velocity)
        data.GetPointData().AddArray(forceCol)

        writer = vtkXMLPolyDataWriter()
        writer.SetInputData(data)
        writer.SetFileName(file_name)
        writer.Write()


class SphereSystemNoFriction(SphereSystem):
    def __init__(self, initConfigFile, shellRadius, stepSize=1.0e-02,
                 accelerationDueToGravity=10.0, collisionBuffer=0.1):
        super().__init__(initConfigFile, shellRadius, stepSize,
                         accelerationDueToGravity, collisionBuffer)
        self.Dn = None

    def computeDirectionMatrix(self):
        self.Dn = np.zeros((3 * self.nSphere, self.nCollision))

        for k in range(self.nCollision):
            i = self.collisionPairs[k, 0]
            j = self.collisionPairs[k, 1]

            if i >= 0:
                iStart, iStop = 3 * i, 3 * i + 3
                self.Dn[iStart:iStop, k] = -self.collisionNormals[k, :]

            jStart, jStop = 3 * j, 3 * j + 3
            self.Dn[jStart:jStop, k] = self.collisionNormals[k, :]

    def setupComplementarityProblem(self):
        A = self.Dn.T @ self.MInv @ self.Dn
        b = self.Dn.T @ self.vKnown
        return A, b

    def computeContactForce(self, A, b, solverConfig):
        gamma = solveLcp(A, b, solverConfig)
        self.forceCol = self.Dn @ gamma


class SphereSystemWithFrictionLcp(SphereSystem):
    def __init__(self, initConfigFile, shellRadius, stepSize=1.0e-02,
                 accelerationDueToGravity=10.0,  collisionBuffer=0.1,
                 frictionCoefficient=0.25, nSideLinearCone=8):
        super().__init__(initConfigFile, shellRadius, stepSize,
                         accelerationDueToGravity, collisionBuffer)
        self.frictionCoefficient = frictionCoefficient
        self.nSideLinearCone = nSideLinearCone
        self.Dn = None
        self.Dt = None

    def computeDirectionMatrix(self):
        dTheta = 2.0 * np.pi / self.nSideLinearCone
        theta = dTheta * np.arange(self.nSideLinearCone)

        refDt = np.stack((np.cos(theta), np.sin(theta)))

        self.Dn = np.zeros((3 * self.nSphere, self.nCollision))
        self.Dt = np.zeros((3 * self.nSphere,
                            self.nSideLinearCone * self.nCollision))

        for k in range(self.nCollision):
            i = self.collisionPairs[k, 0]
            j = self.collisionPairs[k, 1]

            tangent1, tangent2 = \
                computeTangentDirections(self.collisionNormals[k, :])
            localDt = np.stack((tangent1, tangent2)).T @ refDt

            if i >= 0:
                iStart, iStop = 3 * i, 3 * i + 3
                self.Dn[iStart:iStop, k] = -self.collisionNormals[k, :]
                for s in range(self.nSideLinearCone):
                    self.Dt[iStart:iStop, self.nSideLinearCone * k + s] = \
                        -localDt[:, s]

            jStart, jStop = 3 * j, 3 * j + 3
            self.Dn[jStart:jStop, k] = self.collisionNormals[k, :]
            for s in range(self.nSideLinearCone):
                self.Dt[jStart:jStop, self.nSideLinearCone * k + s] = \
                    localDt[:, s]

    def setupComplementarityProblem(self):
        E = np.kron(np.eye(self.nCollision),
                    np.ones((self.nSideLinearCone, 1)))

        A = np.block([[self.Dn.T @ self.MInv @ self.Dn,
                       self.Dn.T @ self.MInv @ self.Dt,
                       np.zeros((self.nCollision, self.nCollision))],
                      [self.Dt.T @ self.MInv @ self.Dn,
                       self.Dt.T @ self.MInv @ self.Dt,
                       E],
                      [self.frictionCoefficient * np.eye(self.nCollision),
                       -E.T,
                       np.zeros((self.nCollision, self.nCollision))]])
        b = np.hstack((self.Dn.T @ self.vKnown,
                       self.Dt.T @ self.vKnown,
                       np.zeros(self.nCollision)))

        return A, b

    def computeContactForce(self, A, b, solverConfig):
        x = solveLcp(A, b, solverConfig)
        gamma = x[0:self.nCollision]
        beta = x[self.nCollision:(self.nSideLinearCone + 1) * self.nCollision]
        self.forceCol = self.Dn @ gamma + self.Dt @ beta


class SphereSystemWithFrictionCcp(SphereSystem):
    def __init__(self, initConfigFile, shellRadius, stepSize=1.0e-02,
                 accelerationDueToGravity=10.0, collisionBuffer=0.1,
                 frictionCoefficient=0.25):
        super().__init__(initConfigFile, shellRadius, stepSize,
                         accelerationDueToGravity, collisionBuffer)
        self.frictionCoefficient = frictionCoefficient
        self.D = None

    def computeDirectionMatrix(self):
        self.D = np.zeros((3 * self.nSphere, 3 * self.nCollision))

        for k in range(self.nCollision):
            i = self.collisionPairs[k, 0]
            j = self.collisionPairs[k, 1]

            tangent1, tangent2 = \
                computeTangentDirections(self.collisionNormals[k, :])

            if i >= 0:
                iStart, iStop = 3 * i, 3 * i + 3
                self.D[iStart:iStop, 3 * k] = -self.collisionNormals[k, :]
                self.D[iStart:iStop, 3 * k + 1] = -tangent1
                self.D[iStart:iStop, 3 * k + 2] = -tangent2

            jStart, jStop = 3 * j, 3 * j + 3
            self.D[jStart:jStop, 3 * k] = self.collisionNormals[k, :]
            self.D[jStart:jStop, 3 * k + 1] = tangent1
            self.D[jStart:jStop, 3 * k + 2] = tangent2

    def setupComplementarityProblem(self):
        A = self.D.T @ self.MInv @ self.D
        b = self.D.T @ self.vKnown
        return A, b

    def computeContactForce(self, A, b, solverConfig):
        gamma = solveCcp(A, b, self.frictionCoefficient, solverConfig)
        self.forceCol = self.D @ gamma

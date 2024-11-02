import os

import numpy as np
from vtk import vtkPoints, vtkDoubleArray, vtkPolyData, vtkPolyDataWriter


class SphereSystem(object):
    def __init__(self, initConfig, shellRadius, accelerationDueToGravity,
                 stepSize, numStep, outputFreq, outputDir, collisionBuffer,
                 useSeparations):
        self.readConfig(initConfig)
        self.shellRadius = shellRadius
        self.accelerationDueToGravity = accelerationDueToGravity
        self.stepSize = stepSize
        self.numStep = numStep
        self.outputFreq = outputFreq
        self.outputDir = outputDir
        self.collisionBuffer = collisionBuffer
        self.useSeparations = useSeparations

        os.makedirs(self.outputDir, exist_ok=True)
        self.stepInd = 0
        self.outputInd = 0
        self.computeInverseMassMatrix()
        self.vKnown = np.zeros((3 * self.nSphere,))
        self.forceCol = np.zeros((3 * self.nSphere,))

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

    def step(self, friction_model, solver):
        self.stepInd += 1

        self.forceCol.fill(0.0)
        self.computeVelocityKnown()
        self.detectCollisions()
        if self.nCollision > 0:
            friction_model.computeDirectionMatrix(self.nSphere,
                                                  self.collisionPairs,
                                                  self.collisionNormals)
            friction_model.computeComplementarityProblem(self.MInv,
                                                         self.vKnown)
            if not self.useSeparations:
                self.forceCol = friction_model.computeContactForce(
                    self.stepInd, solver)
            else:
                self.forceCol = friction_model.computeContactForce(
                    self.stepInd, solver,
                    self.collisionSeparations / self.stepSize)
            self.velocity = self.vKnown + self.MInv @ self.forceCol
        else:
            self.velocity = self.vKnown
        self.position += self.stepSize * self.velocity

    def output(self):
        file_name = os.path.join(self.outputDir,
                                 'snapshot_{:06d}.vtk'.format(self.outputInd))

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

        writer = vtkPolyDataWriter()
        writer.SetInputData(data)
        writer.SetFileName(file_name)
        writer.Write()

        self.outputInd += 1

    def run(self, friction_model, solver):
        self.output()
        for iStep in range(1, self.numStep + 1):
            self.step(friction_model, solver)
            if self.outputFreq > 0 and iStep % self.outputFreq == 0:
                self.output()


def createSphereSystem(config, root_dir):
    initConfig = os.path.join(root_dir, config['initial_configuration'])
    shellRadius = config['shell_radius']
    accelerationDueToGravity = config['acceleration_due_to_gravity']
    stepSize = config['step_size']
    numStep = config['num_step']
    outputFreq = config['snapshot_frequency']
    outputDir = os.path.join(root_dir, config['snapshot_dir'])
    collisionBuffer = config['collision_buffer']
    useSeparations = config['use_separations']
    return SphereSystem(initConfig, shellRadius, accelerationDueToGravity,
                        stepSize, numStep, outputFreq, outputDir,
                        collisionBuffer, useSeparations)

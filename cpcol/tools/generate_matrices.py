#!/usr/bin/env python3

import os
import argparse

import numpy as np

from cpcol.sphere_system import \
    SphereSystemNoFriction, \
    SphereSystemWithFrictionLcp, \
    SphereSystemWithFrictionCcp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--collisionSolver',
        type=str,
        required=True,
        choices=['NoFriction', 'WithFrictionLcp', 'WithFrictionCcp'],
        help='type of collision resolution'
    )
    parser.add_argument(
        '--configFile',
        type=str,
        required=True,
        help='initial system configuration file'
    )
    parser.add_argument(
        '--shellRadius',
        type=float,
        required=True,
        help='radius of outer shell'
    )
    parser.add_argument(
        '--stepSize',
        type=float,
        default=1.0e-02,
        help='time step size'
    )
    parser.add_argument(
        '--accelerationDueToGravity',
        type=float,
        default=10.0,
        help='magnitude of acceleration due to gravity'
    )
    parser.add_argument(
        '--collisionBuffer',
        type=float,
        default=0.1,
        help='maximum relative distance between two spheres in contact'
    )
    parser.add_argument(
        '--frictionCoefficient',
        type=float,
        default=0.25,
        help='coefficient of friction'
    )
    parser.add_argument(
        '--nSideLinearCone',
        type=int,
        default=8,
        help='number of sides in linearized collision cone'
    )
    parser.add_argument(
        '--outputDir',
        type=str,
        default='data',
        help='directory where simulation snapshots will be saved'
    )

    args = parser.parse_args()

    if args.collisionSolver == 'NoFriction':
        system = SphereSystemNoFriction(args.configFile,
                                        args.shellRadius,
                                        args.stepSize,
                                        args.accelerationDueToGravity,
                                        args.collisionBuffer)
    elif args.collisionSolver == 'WithFrictionLcp':
        system = SphereSystemWithFrictionLcp(args.configFile,
                                             args.shellRadius,
                                             args.stepSize,
                                             args.accelerationDueToGravity,
                                             args.collisionBuffer,
                                             args.frictionCoefficient,
                                             args.nSideLinearCone)
    elif args.collisionSolver == 'WithFrictionCcp':
        system = SphereSystemWithFrictionCcp(args.configFile,
                                             args.shellRadius,
                                             args.stepSize,
                                             args.accelerationDueToGravity,
                                             args.collisionBuffer,
                                             args.frictionCoefficient)
    else:
        raise ValueError('unknown collisionSolver')

    system.computeVelocityKnown()
    system.detectCollisions()
    system.computeDirectionMatrix()
    A, b = system.setupComplementarityProblem()

    os.makedirs(args.outputDir, exist_ok=True)
    np.savetxt(os.path.join(args.outputDir, 'A.txt'), A)
    np.savetxt(os.path.join(args.outputDir, 'b.txt'), b)

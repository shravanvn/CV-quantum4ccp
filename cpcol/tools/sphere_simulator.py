#!/usr/bin/env python3

import argparse

from cpcol.sphere_system import \
    SphereSystemNoFriction, \
    SphereSystemWithFrictionLcp, \
    SphereSystemWithFrictionCcp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frictionModel',
        type=str,
        required=True,
        choices=['none', 'linearized', 'quadratic'],
        help='frictional contact force model'
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
        '--nStep',
        type=int,
        default=1000,
        help='number of time steps'
    )
    parser.add_argument(
        '--stepSize',
        type=float,
        default=1.0e-03,
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
    parser.add_argument(
        '--outputFrequency',
        type=int,
        default=1,
        help='interval at which simulation snapshots will be saved'
    )

    args = parser.parse_args()

    if args.frictionModel == 'none':
        system = SphereSystemNoFriction(args.configFile,
                                        args.shellRadius,
                                        args.stepSize,
                                        args.accelerationDueToGravity,
                                        args.collisionBuffer)
    elif args.frictionModel == 'linearized':
        system = SphereSystemWithFrictionLcp(args.configFile,
                                             args.shellRadius,
                                             args.stepSize,
                                             args.accelerationDueToGravity,
                                             args.collisionBuffer,
                                             args.frictionCoefficient,
                                             args.nSideLinearCone)
    elif args.frictionModel == 'quadratic':
        system = SphereSystemWithFrictionCcp(args.configFile,
                                             args.shellRadius,
                                             args.stepSize,
                                             args.accelerationDueToGravity,
                                             args.collisionBuffer,
                                             args.frictionCoefficient)
    else:
        raise ValueError('unknown frictionModel')

    solverConfig = {
        'name': 'apgd',
        'maxIter': 1000,
        'residualTol': 1.0e-06,
        'stepSizeTol': 1.0e-12,
        'outputDir': args.outputDir
    }

    system.output(args.outputDir)
    for iStep in range(1, args.nStep + 1):
        system.step(solverConfig)
        if args.outputFrequency > 0 and iStep % args.outputFrequency == 0:
            system.output(args.outputDir)

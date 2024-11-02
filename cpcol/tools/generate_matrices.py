#!/usr/bin/env python3

import argparse
import os

import numpy as np
import yaml

from cpcol.sphere_system import createSphereSystem
from cpcol.friction_model import createFrictionModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
        help='path to YAML config'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    root_dir = os.path.dirname(os.path.abspath(args.config))
    config['root_dir'] = root_dir

    print(yaml.dump(config))

    sphere_system = createSphereSystem(config['sphere_system'], root_dir)
    friction_model = createFrictionModel(config['friction_model'])

    sphere_system.computeVelocityKnown()
    sphere_system.detectCollisions()

    if sphere_system.nCollision > 0:
        friction_model.computeDirectionMatrix(sphere_system.nSphere,
                                              sphere_system.collisionPairs,
                                              sphere_system.collisionNormals)
        friction_model.computeComplementarityProblem(sphere_system.MInv,
                                                     sphere_system.vKnown)

        np.savetxt(os.path.join(root_dir, 'A.txt'), friction_model.A)
        np.savetxt(os.path.join(root_dir, 'b.txt'), friction_model.b)

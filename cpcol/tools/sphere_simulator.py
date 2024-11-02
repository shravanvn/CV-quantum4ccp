#!/usr/bin/env python3

import argparse
import os

import yaml

from cpcol.sphere_system import createSphereSystem
from cpcol.friction_model import createFrictionModel
from cpcol.cp_solver import createCpSolver


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
    solver = createCpSolver(config['solver'], root_dir)

    sphere_system.run(friction_model, solver)

# Complementarity Formulation of Rigid Body Collision

This python package enables detection and resolution of rigid collisions between
rigid spheres using the complementarity formulation.

## Dependencies

To automatically install the dependencies and set up the python package, run the
following:
```sh
python3 -m pip install -e /path/to/cpcol
```
We recommend creating a fresh virtual environment before the installation step
using, e.g.,
```sh
python3 -m venv env
. env/bin/activate
```

## Entry Points

We provide two entry points:

- `cpcol/tools/generate_matrices.py` generates the matrices for the
  complementarity problem and saves them to plain-text files.

- `cpcol/tools/sphere_simulator.py` simulates a system of free-falling rigid
  spheres, and writes the system configuration to VTK files.

Both scripts accept a '--configFile' option which accepts a plain-text file. Two
examples are provided in the `configs/` directory. Three types of collision
solvers are supported, which is specified through the '--collisionSolver'
option:

- `NoFriction`, which uses non-frictional LCP

- `WithFrictionLcp`, which uses frictional LCP. This accepts an additional
  `--nSideLinearCone` argument, which should be a positive even integer, and
  specifies the number of sides in the base of the polyhedral cone used to
  approximate the quadratic Coulomb friction cone.

- `WithFrictionCcp`, which uses frictional CCP.

Run the scripts with `--help` option to get the full list of available options.

## Features

- [ ] Matrix generation
  - [x] Non-frictional LCP
  - [ ] Frictional LCP (needs testing!)
  - [x] Frictional CCP

- [ ] First-order APGD solver
  - [x] Non-frictional LCP
  - [ ] Frictional LCP (needs testing!)
  - [x] Frictional CCP

- [ ] Second order Newton minimum-map solver
  - [ ] Non-frictional LCP
  - [ ] Frictional LCP
  - [ ] Frictional CCP

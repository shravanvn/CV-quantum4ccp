# Complementarity Formulation of Rigid Body Collision

This python package enables detection and resolution of rigid collisions between
rigid spheres using the complementarity formulation.

## Usage

> [!IMPORTANT]
> We will assume that you are working from the `</path/to/repo/root>/cpcol`
> directory that contains *this* README file.

### Dependencies

To automatically install the dependencies and set up the python package, run the
following:
```sh
python3 -m pip install -e .
```
This sets up an 'editable' package installation - any change in the source files
will be automatically sourced on the next (re-)import of the `cpcol` module.

We recommend creating a fresh virtual environment before the installation step
using, e.g.,
```sh
python3 -m venv </path/to/env/dir>
. </path/to/env/dir>/env/bin/activate
```

In case any of the scripts below throws errors due to missing packages, it is
likely that the package requirements have changed. In that case, rerun the
editable pip install step from above.

### Code Structure

Collision code is split into three classes:

- `SphereSystem` handles setting up a system of spheres inside a larger
  spherical shell, detecting collisions, time stepping, writing snapshots etc.

- `FrictionModel` is concerned with setting up the complementarity model of
  rigid contact. Three possibilities are considered:

  - No friction, in which case we solve a linear complementarity problem (LCP).
  - With friction, but the friction cone is linearized. In this case we still
    solve an LCP, but is augmented with additional variables.
  - With friction with quadratic friction cone, where we solve a cone
    complementarity problem (CCP).

  In each case, this class sets up the complementarity problem (CP), and parses
  the solution to compute the contact force.

- `CpSolver` actually solves the CP. Currently two methods are available:

  - Accelerated projected gradient descent (APGD), works for both LCP and CCP.
  - Minimum-map Newton (mmNewton), only works for LCP.

Each module provides a corresponding `create*` method that instantiates an
appropriate object from a `dict` of options.

### Utility Scripts

We provide two utility scripts:

- `tools/generate_matrices.py` generates the matrices for the complementarity
  problem and saves them to plain-text files.

- `tools/sphere_simulator.py` simulates a system of free-falling rigid spheres,
  and writes the system snapshots to VTK files.

Both scripts require a `config.yaml` file argument; this YAML file lists the
key-value pairs required to construct a `SphereSystem`, `FrictionModel`, or
`CpSolver` object via their corresponding `create*` method. See
`../runs/sphere_system_3/config.yaml` for an example.

The `sphere_system` section of the YAML config requires an
`initial_configuration` that points to a file specifying the mass, radius,
initial position and initial velocity of the each sphere. The format of this
file is very simple: each line has the form
```
<mass> <radius> <pos_x> <pos_y> <pos_z> <vel_x> <vel_y> <vel_z>
```
specifying the details of one sphere. Consecutive numbers are separated by one
or more white-spaces. Any line that starts with `#` are ignored. Note that the
outer shell is *not* specified in this file. See
`../runs/sphere_system_3/init.txt` for an example.

When running these two scripts, all paths (files and directories) specified in
the YAML file are treated as relative to the directory containing the
configuration. For example, the `../runs/sphere_system_3/config.yaml` file
specifies
```yaml
sphere_system:
  initial_configuration: 'init.txt'
  snapshot_dir: 'snapshots'

solver:
  log_dir: 'solver_logs':
```
along with other options. When you run
```
tools/sphere_simulator.py ../runs/sphere_system_3/config.yaml
```
then the directory containing the configuration file is
`../runs/sphere_system_3`; the script then looks for the initial configuration
file `init.txt` inside this directory, and saves the snapshots and solver logs
in `../runs/sphere_system_3/snapshots` and `../runs/sphere_system_3/solver_logs`
directories, respectively.

## Features

- [ ] Matrix generation
  - [x] Non-frictional LCP
  - [ ] Frictional LCP (needs testing!)
  - [x] Frictional CCP

- [ ] First-order APGD solver
  - [x] Non-frictional LCP
  - [ ] Frictional LCP (needs testing!)
  - [x] Frictional CCP

- [ ] Second order minimum-map Newton solver
  - [x] Non-frictional LCP
  - [ ] Frictional LCP (needs testing!)

- [ ] Second order symmetric-cone interior point solver
  - [ ] Frictional CCP

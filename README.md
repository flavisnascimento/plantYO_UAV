# PlantYO UAV

ROS package for UAV-based aerial seeding in Cerrado restoration scenarios.

The package provides mission planning, route generation, seed dispensing and
Gazebo simulation using the MRS UAV System.

## Overview

The system compares two dispenser configurations:

- `1comp`: single-compartment operation;
- `3comp`: multi-compartment operation.

Both configurations use the same total payload. They differ in how the payload
is partitioned and consumed during the mission.

Supported planning methods:

- `HGS`: CVRP solver;
- `DAHA`: discrete Artificial Hummingbird Algorithm;
- `NN`: nearest-neighbor algorithm;
- `TSP`: TSP-based route construction and splitting.

The official execution flow uses `tmux` and `dispensor_planter.launch`.

## Requirements

- Ubuntu 20.04;
- ROS Noetic;
- Gazebo 11;
- MRS UAV System;
- Catkin tools;
- tmux;
- tmuxinator;
- Python 3;
- NumPy;
- hygese for HGS.

## Package structure

```text
plantYO_UAV/
├── CMakeLists.txt
├── package.xml
├── config/
├── launch/
├── models/
├── scripts/
│   ├── dispensor_planter_node.py
│   ├── mission_planter_node.py
│   ├── hgs_solver.py
│   ├── solver_benchmark.py
│   ├── HGS.py
│   ├── DAHA.py
│   ├── NN.py
│   ├── TSP.py
│   └── ...
├── tmux/
│   ├── setup_run.sh
│   ├── start.sh
│   └── session.yml
└── worlds/
```

`mission_planter_node.py` contains the general mission logic.
`dispensor_planter_node.py` selects the planning method and configures the
dispenser operation.

## Build

```bash
source /opt/ros/noetic/setup.bash

cd ~/plantyo_ws
catkin build plantyo_uav
source devel/setup.bash
```

Verify the package:

```bash
rospack find plantyo_uav
```

Expected result:

```text
/root/plantyo_ws/src/plantYO_UAV
```

## Run a mission

```bash
source /opt/ros/noetic/setup.bash
source ~/plantyo_ws/devel/setup.bash

cd ~/plantyo_ws/src/plantYO_UAV/tmux
./setup_run.sh NN 1comp 150
./start.sh
```

The setup command has the following format:

```text
./setup_run.sh <algorithm> <mode> <grid_size>
```

Examples:

```bash
./setup_run.sh HGS 1comp 75
./setup_run.sh DAHA 3comp 100
./setup_run.sh NN 1comp 150
./setup_run.sh TSP 3comp 150
```

`start.sh` uses `tmuxinator` and loads `tmux/session.yml`.

## Stop the simulation

```bash
tmux -L mrs kill-server
```

## Benchmark

```bash
cd ~/plantyo_ws/src/plantYO_UAV/scripts
python3 run_benchmark.py
```

The benchmark uses the implementations in `HGS.py`, `DAHA.py`, `NN.py` and
`TSP.py`. The `solver_benchmark.py` module provides compatibility imports for
the mission node and benchmark runner.

## Mission data

Mission data is saved in:

```text
/root/plantyo_logs
```

The logger generates CSV files for waypoints and routes, as well as a JSON
summary for each mission.

ROS diagnostic logs are stored separately in:

```text
/root/.ros/log
```

## Development checks

```bash
cd ~/plantyo_ws/src/plantYO_UAV
python3 -m py_compile scripts/*.py
git diff --check
git status
```

The ROS package name is `plantyo_uav`.

# plantYO_UAV

Commodity-Constrained Split Delivery Vehicle Routing Problem (C-SDVRP) for
aerial seeding in Cerrado restoration using UAVs.

Companion code for the paper "Optimizing UAV Operations under Capacity and
Distance Constraints: A Comparison of Routing Heuristics for Cerrado
Restoration" (accepted at ICUAS 2026). Developed at LARIS / UFSCar.

## Overview

The drone plants three seed commodities (grass, shrub, tree) over a field,
under payload and flight-range constraints, returning to a base to reload.
The routing is modeled as a C-SDVRP and solved by several algorithms, then
validated in Gazebo through the MRS UAV System.

Implemented solvers: D-AHA (Discrete Artificial Hummingbird Algorithm),
HGS (Hybrid Genetic Search), Nearest Neighbor, and LKH-Split.

## Structure

    python/plantYO_UAV/
      scripts/    routing solvers and ROS nodes
      models/     Gazebo models (colored seeds, terrain, trees)
      worlds/     simulation world
      launch/     ROS launch files
      tmux/       session scripts to run the simulation
      config/     RViz and parameter configs

## Requirements

- Ubuntu 20.04 / ROS Noetic
- MRS UAV System (https://github.com/ctu-mrs/mrs_uav_system)
- Python 3.8, numpy, hygese

## Installation

    cd ~/catkin_ws/src
    git clone -b master https://github.com/flavisnascimento/plantYO_UAV.git
    cd ~/catkin_ws
    catkin build
    source ~/catkin_ws/devel/setup.bash
    pip3 install hygese

## Running

    roscd plantyo_uav/tmux
    ./start.sh
    ./kill.sh

The solver and mission mode are set in tmux/session.yml:

- modo: 1comp (one commodity per campaign) or 3comp (all three together)
- solver: HGS, DAHA, or NN

## License

See python/plantYO_UAV/LICENSE.

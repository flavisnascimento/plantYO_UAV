#!/usr/bin/env python3
"""Interface de compatibilidade para os algoritmos de roteamento."""

import os
import sys

# Garante que a pasta scripts/ (um nível acima deste pacote) esteja no sys.path,
# para os imports de módulos irmãos (grid_generator, joao_tsp_solver) funcionarem
# em qualquer diretório de trabalho, como fazia o arquivo único original.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_PKG_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from split import OptimalSplit
from result import SolverResult
from instance import BenchmarkInstance, GRID_GENERATOR_AVAILABLE
from base import BaseSolver
from HGS import HGSSolverBenchmark
from DAHA import AHASolverBenchmark
from NN import NearestNeighborSolver
from TSP import TSPGreedySolver, TSP2OptSolver, TSPExactSolver
from runner import BenchmarkRunner

__all__ = [
    "OptimalSplit",
    "SolverResult",
    "BenchmarkInstance",
    "GRID_GENERATOR_AVAILABLE",
    "BaseSolver",
    "HGSSolverBenchmark",
    "AHASolverBenchmark",
    "NearestNeighborSolver",
    "TSPGreedySolver",
    "TSP2OptSolver",
    "TSPExactSolver",
    "BenchmarkRunner",
]

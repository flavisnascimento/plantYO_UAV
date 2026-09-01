#!/usr/bin/env python3
"""Solver HGS aplicado ao CVRP."""

import time
from typing import List
import numpy as np

from base import BaseSolver
from result import SolverResult


class HGSSolverBenchmark(BaseSolver):
    """Implementação do solver HGS para CVRP."""

    def __init__(self):
        try:
            import hygese as hgs
            self.hgs = hgs
            self.available = True
        except ImportError:
            self.available = False
            print("[HGS] AVISO: hygese não instalado. pip install hygese")

    @property
    def name(self) -> str:
        return "HGS-CVRP"

    @property
    def reference(self) -> str:
        return "Hybrid Genetic Search para CVRP"

    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:

        if not self.available:
            return SolverResult(
                solver_name=self.name,
                instance_name=kwargs.get('instance_name', 'unknown'),
                routes=[],
                total_distance=float('inf'),
                num_routes=0,
                computation_time=0,
                feasible=False,
                metadata={'error': 'hygese not installed'}
            )

        start_time = time.time()

        # Configura dados para HGS
        data = {
            'distance_matrix': distance_matrix.astype(np.float64),
            'demands': demands,
            'vehicle_capacity': capacity,
            'num_vehicles': int(np.ceil(1.3 * sum(demands) / capacity)) + 3,
            'depot': 0,
            'duration_limit': autonomy,  # Usa autonomia como duration_limit
            'service_times': [0.0] * len(demands)
        }

        ap = self.hgs.AlgorithmParameters(timeLimit=time_limit)
        solver = self.hgs.Solver(parameters=ap, verbose=False)

        result = solver.solve_cvrp(data)

        computation_time = time.time() - start_time

        routes = result.routes if hasattr(result, 'routes') else []

        total_distance = 0.0
        for route in routes:
            total_distance += self._calc_route_distance(route, distance_matrix)

        return SolverResult(
            solver_name=self.name,
            instance_name=kwargs.get('instance_name', 'unknown'),
            routes=routes,
            total_distance=total_distance,
            num_routes=len(routes),
            computation_time=computation_time,
            feasible=True,
            metadata={
                'hgs_cost': result.cost if hasattr(result, 'cost') else None,
                'capacity': capacity,
                'autonomy': autonomy
            }
        )

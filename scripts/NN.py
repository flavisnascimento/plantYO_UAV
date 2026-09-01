#!/usr/bin/env python3
"""Algoritmo de vizinho mais próximo."""

import time
from typing import List
import numpy as np

from base import BaseSolver
from result import SolverResult


class NearestNeighborSolver(BaseSolver):
    """
    Nearest Neighbor heurístico como baseline.

    Construção gulosa: sempre visita o waypoint mais próximo não visitado
    que caiba na rota atual.
    """

    @property
    def name(self) -> str:
        return "Nearest-Neighbor"

    @property
    def reference(self) -> str:
        return "Baseline heurístico - Vizinho mais próximo"

    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:

        start_time = time.time()
        n = len(demands)

        # Waypoints não visitados (exclui depósito)
        unvisited = set(range(1, n))
        routes = []

        while unvisited:
            route = []
            current_demand = 0
            current_distance = 0.0
            current_pos = 0  # Começa na base

            while unvisited:
                # Encontra vizinho mais próximo que caiba
                best_wp = None
                best_dist = float('inf')

                for wp in unvisited:
                    dist_to_wp = distance_matrix[current_pos, wp]
                    dist_to_base = distance_matrix[wp, 0]

                    if (current_demand + demands[wp] <= capacity and
                            current_distance + dist_to_wp + dist_to_base <= autonomy):
                        if dist_to_wp < best_dist:
                            best_dist = dist_to_wp
                            best_wp = wp

                if best_wp is None:
                    break  # Não cabe mais ninguém

                route.append(best_wp)
                unvisited.remove(best_wp)
                current_demand += demands[best_wp]
                current_distance += best_dist
                current_pos = best_wp

            if route:
                routes.append(route)

        computation_time = time.time() - start_time

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
            metadata={'capacity': capacity, 'autonomy': autonomy}
        )

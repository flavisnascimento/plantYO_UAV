#!/usr/bin/env python3
"""Algoritmos TSP para construção e divisão de rotas."""

import time
from typing import List, Tuple
import numpy as np

from base import BaseSolver
from result import SolverResult
from split import OptimalSplit
from util_rotas import build_nn_tour, tour_cost


class TSPGreedySolver(BaseSolver):
    """TSP Greedy (Nearest Neighbor) + Split Ótimo (Route-First, Cluster-Second)."""

    def __init__(self, use_optimal_split: bool = True):
        self.use_optimal_split = use_optimal_split

    @property
    def name(self) -> str:
        return "TSP-Greedy"

    @property
    def reference(self) -> str:
        return "Nearest Neighbor TSP + Split ótimo"

    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:

        start_time = time.time()
        n = len(demands) - 1  # Exclui depósito

        if n == 0:
            return SolverResult(
                solver_name=self.name,
                instance_name=kwargs.get('instance_name', 'unknown'),
                routes=[],
                total_distance=0.0,
                num_routes=0,
                computation_time=0,
                feasible=True
            )

        # Fase 1: tour gigante por Nearest Neighbor
        tour = self._build_nn_tour(distance_matrix, n)

        # Fase 2: Split para dividir em rotas
        routes, total_distance = OptimalSplit.split(
            tour, demands, capacity, autonomy, distance_matrix
        )

        computation_time = time.time() - start_time

        return SolverResult(
            solver_name=self.name,
            instance_name=kwargs.get('instance_name', 'unknown'),
            routes=routes,
            total_distance=total_distance,
            num_routes=len(routes),
            computation_time=computation_time,
            feasible=True,
            metadata={
                'split_type': 'optimal' if self.use_optimal_split else 'greedy',
                'tour_construction': 'nearest_neighbor'
            }
        )

    def _build_nn_tour(self, dm: np.ndarray, n: int) -> List[int]:
        """Constrói tour usando Nearest Neighbor (ver util_rotas)."""
        return build_nn_tour(dm, n)


class TSP2OptSolver(BaseSolver):
    """
    TSP Heurístico (2-opt/3-opt) + Split Ótimo.

    Constrói tour inicial com NN, melhora com 2-opt (ou 3-opt) até convergir
    ou dar timeout, e aplica Split Ótimo. Heurístico, não garante ótimo global.

    Referências:
      - Lin, S. (1965). Computer solutions of the traveling salesman problem
      - Croes, G. A. (1958). A method for solving traveling-salesman problems
    """

    def __init__(self,
                 use_optimal_split: bool = True,
                 max_no_improve: int = 100,
                 use_3opt: bool = False):
        self.use_optimal_split = use_optimal_split
        self.max_no_improve = max_no_improve
        self.use_3opt = use_3opt

    @property
    def name(self) -> str:
        if self.use_3opt:
            return "TSP-3opt"
        return "TSP-2opt"

    @property
    def reference(self) -> str:
        base = "Lin & Kernighan (1973) 2-opt/3-opt"
        return f"{base} + Split ótimo"

    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:

        start_time = time.time()
        n = len(demands) - 1

        if n == 0:
            return SolverResult(
                solver_name=self.name,
                instance_name=kwargs.get('instance_name', 'unknown'),
                routes=[],
                total_distance=0.0,
                num_routes=0,
                computation_time=0,
                feasible=True
            )

        # Fase 1: tour inicial com NN
        tour = self._build_nn_tour(distance_matrix, n)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, distance_matrix)

        # Fase 2: melhora com 2-opt (ou 3-opt)
        no_improve = 0
        iteration = 0

        while no_improve < self.max_no_improve:
            if time.time() - start_time > time_limit * 0.8:  # Reserva tempo para split
                break

            if self.use_3opt and n > 10:
                new_tour, new_cost = self._three_opt_pass(tour, distance_matrix)
            else:
                new_tour, new_cost = self._two_opt_pass(tour, distance_matrix)

            if new_cost < best_cost - 1e-6:
                best_tour = new_tour.copy()
                best_cost = new_cost
                tour = new_tour
                no_improve = 0
            else:
                no_improve += 1

            iteration += 1

        # Fase 3: Split
        if self.use_optimal_split:
            routes, total_distance = OptimalSplit.split(
                best_tour, demands, capacity, autonomy, distance_matrix
            )
        else:
            routes, total_distance = OptimalSplit.split_greedy(
                best_tour, demands, capacity, autonomy, distance_matrix
            )

        computation_time = time.time() - start_time

        return SolverResult(
            solver_name=self.name,
            instance_name=kwargs.get('instance_name', 'unknown'),
            routes=routes,
            total_distance=total_distance,
            num_routes=len(routes),
            computation_time=computation_time,
            feasible=True,
            metadata={
                'split_type': 'optimal' if self.use_optimal_split else 'greedy',
                'tour_cost_before_split': best_cost,
                'iterations': iteration,
                'optimization': '3-opt' if self.use_3opt else '2-opt'
            }
        )

    def _build_nn_tour(self, dm: np.ndarray, n: int) -> List[int]:
        """Tour inicial por Nearest Neighbor (ver util_rotas)."""
        return build_nn_tour(dm, n)

    def _tour_cost(self, tour: List[int], dm: np.ndarray) -> float:
        """Custo do tour fechado (ver util_rotas)."""
        return tour_cost(tour, dm)

    def _two_opt_pass(self, tour: List[int], dm: np.ndarray) -> Tuple[List[int], float]:
        """Um passo completo de 2-opt: testa todas as trocas (i,j) e retorna a melhor."""
        n = len(tour)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, dm)

        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_cost = self._tour_cost(new_tour, dm)

                if new_cost < best_cost - 1e-6:
                    best_tour = new_tour
                    best_cost = new_cost

        return best_tour, best_cost

    def _three_opt_pass(self, tour: List[int], dm: np.ndarray) -> Tuple[List[int], float]:
        """Um passo de 3-opt simplificado (algumas reconexões por tripla)."""
        n = len(tour)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, dm)

        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n + 1):
                    A = tour[0:i+1]
                    B = tour[i+1:j+1]
                    C = tour[j+1:k] if k <= n else tour[j+1:]
                    D = tour[k:] if k < n else []

                    candidates = [
                        A + B[::-1] + C + D,        # Reverte B
                        A + B + C[::-1] + D,        # Reverte C
                        A + B[::-1] + C[::-1] + D,  # Reverte ambos
                        A + C + B + D,              # Troca B e C
                    ]

                    for candidate in candidates:
                        if len(candidate) != n:
                            continue
                        cost = self._tour_cost(candidate, dm)
                        if cost < best_cost - 1e-6:
                            best_tour = candidate
                            best_cost = cost

        return best_tour, best_cost



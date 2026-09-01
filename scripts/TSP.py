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


class TSPExactSolver(BaseSolver):
    """
    TSP Exato (Held-Karp) + Split Ótimo.

    Resolve o TSP exatamente por programação dinâmica Held-Karp (O(n² 2^n)),
    depois aplica Split. Só viável para instâncias pequenas (n <= 16); acima
    disso tenta python-tsp e, por fim, NN + 2-opt intensivo (não garante ótimo).

    Referências:
      - Held, M. & Karp, R. (1962). A dynamic programming approach to sequencing
      - Bellman, R. (1962). Dynamic programming treatment of the travelling salesman
    """

    MAX_DP_NODES = 16  # Limite para DP exato (2^16 = 65k estados)

    def __init__(self, use_optimal_split: bool = True):
        self.use_optimal_split = use_optimal_split
        self._check_dependencies()

    def _check_dependencies(self):
        """Verifica bibliotecas opcionais disponíveis"""
        self.has_python_tsp = False
        self.has_concorde = False

        try:
            from python_tsp.exact import solve_tsp_dynamic_programming  # noqa: F401
            self.has_python_tsp = True
        except ImportError:
            pass

        try:
            import concorde  # noqa: F401
            self.has_concorde = True
        except ImportError:
            pass

    @property
    def name(self) -> str:
        return "TSP-Exact"

    @property
    def reference(self) -> str:
        return "Held & Karp (1962) - Programação Dinâmica O(n² 2^n)"

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

        method_used = "unknown"
        is_optimal = False

        if n <= self.MAX_DP_NODES:
            tour = self._held_karp_dp(distance_matrix, n)
            method_used = "held-karp-dp"
            is_optimal = True
        elif self.has_python_tsp and n <= 25:
            tour = self._python_tsp_solve(distance_matrix, n)
            method_used = "python-tsp"
            is_optimal = True
        else:
            tour = self._nn_with_intensive_2opt(distance_matrix, n, time_limit * 0.8)
            method_used = "nn-2opt-heuristic"
            is_optimal = False

        # Fase 2: Split
        if self.use_optimal_split:
            routes, total_distance = OptimalSplit.split(
                tour, demands, capacity, autonomy, distance_matrix
            )
        else:
            routes, total_distance = OptimalSplit.split_greedy(
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
                'method': method_used,
                'is_optimal': is_optimal,
                'split_type': 'optimal' if self.use_optimal_split else 'greedy',
                'num_nodes': n,
                'max_dp_nodes': self.MAX_DP_NODES
            }
        )

    def _held_karp_dp(self, dm: np.ndarray, n: int) -> List[int]:
        """Held-Karp (DP) para TSP. O(n² 2^n). Garante o ótimo do TSP."""
        INF = float('inf')
        num_states = 1 << n

        dp = [[INF] * n for _ in range(num_states)]
        parent = [[-1] * n for _ in range(num_states)]

        for i in range(n):
            node = i + 1
            mask = 1 << i
            dp[mask][i] = dm[0, node]

        for mask in range(1, num_states):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue

                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue

                    new_mask = mask | (1 << next_node)
                    new_cost = dp[mask][last] + dm[last + 1, next_node + 1]

                    if new_cost < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_cost
                        parent[new_mask][next_node] = last

        full_mask = num_states - 1
        best_cost = INF
        best_last = -1

        for i in range(n):
            total_cost = dp[full_mask][i] + dm[i + 1, 0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_last = i

        tour = []
        mask = full_mask
        current = best_last

        while current != -1:
            tour.append(current + 1)
            prev = parent[mask][current]
            mask ^= (1 << current)
            current = prev

        tour.reverse()
        return tour

    def _python_tsp_solve(self, dm: np.ndarray, n: int) -> List[int]:
        """Usa a biblioteca python-tsp para resolver exatamente."""
        try:
            from python_tsp.exact import solve_tsp_dynamic_programming

            permutation, distance = solve_tsp_dynamic_programming(dm)

            tour = [p for p in permutation if p != 0]
            return tour
        except Exception:
            return self._held_karp_dp(dm, n)

    def _nn_with_intensive_2opt(self, dm: np.ndarray, n: int, time_limit: float) -> List[int]:
        """Fallback heurístico para instâncias grandes: NN + 2-opt até timeout."""
        import time as _time
        start = _time.time()

        tour = build_nn_tour(dm, n)

        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, dm)

        improved = True
        while improved and _time.time() - start < time_limit:
            improved = False
            for i in range(len(tour) - 1):
                if _time.time() - start > time_limit:
                    break
                for j in range(i + 2, len(tour)):
                    new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    new_cost = self._tour_cost(new_tour, dm)
                    if new_cost < best_cost - 0.01:
                        best_tour = new_tour
                        best_cost = new_cost
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break

        return best_tour

    def _tour_cost(self, tour: List[int], dm: np.ndarray) -> float:
        """Custo do tour fechado (ver util_rotas)."""
        return tour_cost(tour, dm)

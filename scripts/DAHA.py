#!/usr/bin/env python3
"""Algoritmo D-AHA aplicado ao C-SDVRP."""

import time
import random
from typing import List, Dict, Tuple
import numpy as np

from base import BaseSolver
from result import SolverResult
from split import OptimalSplit
from util_rotas import route_distance, route_demand


class AHASolverBenchmark(BaseSolver):
    """
    Adaptação Discreta do Artificial Hummingbird Algorithm (D-AHA) para C-SDVRP.

    contínua, para o domínio discreto do C-SDVRP via operadores de permutação
    (Swap, 2-Opt). Estratégia Order-First, Split-Second: cada solução é uma
    permutação de clientes (Giant Tour), avaliada por decodificação ótima com o

    Referência:
        Computer Methods in Applied Mechanics and Engineering, 388, 114194.
    """

    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = None,
                 use_optimal_split: bool = True):
        """
        Args:
            population_size: Tamanho da população de soluções
            max_iterations: Máximo de iterações (None = sem limite, usa time_limit)
            use_optimal_split: Se True, usa Split ótimo
        """
        self.population_size = population_size
        self.max_iterations = max_iterations  # None = roda até time_limit
        self.use_optimal_split = use_optimal_split

    @property
    def name(self) -> str:
        return "D-AHA"

    @property
    def reference(self) -> str:
        return "Artificial Hummingbird Algorithm adaptado ao C-SDVRP"

    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        """
        Executa o D-AHA com representação por permutação (Giant Tour), avaliada
        por Split Ótimo. Reserva 5% do time_limit para o processamento final.
        """
        import random

        start_time = time.time()

        # Clientes virtuais: commodities pré-processadas na transformação
        commodity_capacities = None
        commodities = None

        n = len(demands) - 1  # Número de clientes (exclui depósito)

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

        # Inicialização da população: permutações aleatórias
        population = []
        for _ in range(self.population_size):
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            population.append(perm)

        best_solution = None
        best_fitness = float('inf')

        visit_table = [[0.0] * self.population_size for _ in range(self.population_size)]
        for i in range(self.population_size):
            visit_table[i][i] = float('-inf')  # nao se visita

        iteration = 0
        max_iter = self.max_iterations if self.max_iterations else float('inf')
        stagnation_counter = 0
        last_best_fitness = float('inf')

        # Loop principal (reserva 5% do tempo para processamento final)
        while time.time() - start_time < time_limit * 0.95 and iteration < max_iter:
            fitness_values = []
            for individual in population:
                _, fitness = self._evaluate(individual, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)
                fitness_values.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()

                    # Recompensa por descoberta: busca local intensiva na melhor solução
                    remaining_time = (time_limit * 0.95) - (time.time() - start_time)
                    if remaining_time > 1.0:
                        improved = self._intensive_two_opt(
                            best_solution, demands, capacity, autonomy, distance_matrix,
                            max_time=min(1.0, remaining_time * 0.3),
                            commodity_capacities=commodity_capacities, commodities=commodities
                        )
                        _, improved_fitness = self._evaluate(improved, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)
                        if improved_fitness < best_fitness:
                            best_fitness = improved_fitness
                            best_solution = improved.copy()
                            worst_idx = fitness_values.index(max(fitness_values))
                            population[worst_idx] = improved.copy()

            # Monitoramento de convergência
            if abs(best_fitness - last_best_fitness) < 0.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = best_fitness

            elapsed_ratio = (time.time() - start_time) / (time_limit * 0.95)
            elapsed_ratio = min(1.0, max(0.0, elapsed_ratio))

            # Diversificação só em estagnação severa (>30 iterações)
            if stagnation_counter > 30:
                stagnation_counter = 0
                sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
                for idx in sorted_indices[-self.population_size // 6:]:
                    new_perm = list(range(1, n + 1))
                    random.shuffle(new_perm)
                    population[idx] = new_perm

            # Atualiza população com operadores AHA
            new_population = []

            for i, hummingbird in enumerate(population):
                # Verificação de tempo rigorosa
                if time.time() - start_time >= time_limit * 0.95:
                    new_population.append(hummingbird)
                    continue

                r = random.random()

                j_star = self._select_target_via_vt(i, visit_table, fitness_values)
                target = population[j_star] if j_star is not None else best_solution

                if r < 0.33:
                    new_hb = self._guided_foraging(hummingbird, target, iteration, elapsed_ratio)
                    operator_used = "guided"
                elif r < 0.66:
                    new_hb = self._territorial_foraging(hummingbird)
                    operator_used = "territorial"
                else:
                    new_hb = self._migration_foraging(hummingbird)
                    operator_used = "migration"

                _, new_fitness = self._evaluate(new_hb, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)

                if operator_used == "guided" and j_star is not None:
                    if new_fitness < fitness_values[i]:
                        visit_table[i][j_star] = 0.0
                    else:
                        visit_table[i][j_star] += 1.0
                    for k in range(self.population_size):
                        if k != j_star and k != i:
                            visit_table[i][k] += 1.0
                elif operator_used == "territorial":
                    for k in range(self.population_size):
                        if k != i:
                            visit_table[i][k] += 1.0
                elif operator_used == "migration":
                    for k in range(self.population_size):
                        visit_table[i][k] = float("-inf") if k == i else 0.0

                if new_fitness < fitness_values[i]:
                    new_population.append(new_hb)
                else:
                    new_population.append(hummingbird)

            population = new_population
            iteration += 1

        computation_time = time.time() - start_time

        # Gera rotas finais usando Split Ótimo
        if best_solution:
            routes, total_distance = self._evaluate(best_solution, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)

            # Pós-processamento: busca local inter-rota (granular k=20)
            if routes and len(routes) >= 2:
                routes = self._granular_refinement(
                    routes, demands, capacity, autonomy, distance_matrix,
                    k_neighbors=20, max_time=2.0
                )
                total_distance = OptimalSplit._calculate_cost(routes, distance_matrix)
        else:
            routes = []
            total_distance = 0.0

        return SolverResult(
            solver_name=self.name,
            instance_name=kwargs.get('instance_name', 'unknown'),
            routes=routes,
            total_distance=total_distance,
            num_routes=len(routes),
            computation_time=computation_time,
            feasible=True,
            metadata={
                'iterations': iteration,
                'population_size': self.population_size,
                'split_type': 'optimal' if self.use_optimal_split else 'greedy',
                'capacity': capacity,
                'autonomy': autonomy
            }
        )

    def _evaluate(self, sequence: List[int], demands: List[int],
                  capacity: int, autonomy: float, dm: np.ndarray,
                  commodity_capacities: Dict = None, commodities: List = None) -> Tuple[List[List[int]], float]:
        """Avalia uma sequência usando Split Ótimo ou guloso"""
        if self.use_optimal_split:
            return OptimalSplit.split(sequence, demands, capacity, autonomy, dm, commodity_capacities, commodities)
        else:
            return OptimalSplit.split_greedy(sequence, demands, capacity, autonomy, dm, commodity_capacities, commodities)

    def _guided_foraging(self, hummingbird: List[int], target: List[int], iteration: int,
                         elapsed_ratio: float = 0.5) -> List[int]:
        """Operador de forrageamento guiado - Busca local leve integrada"""
        import random
        new_hb = hummingbird.copy()

        decay = 1 - elapsed_ratio

        # Crossover parcial com melhor solução (aprendizado social)
        if random.random() < 0.5 * decay and target:
            cut = random.randint(1, len(hummingbird) - 1)
            segment = target[:cut]
            segment_set = set(segment)
            remaining = [x for x in hummingbird if x not in segment_set]
            new_hb = segment + remaining

        # Busca local leve: 2-opt simples (não intensivo)
        if random.random() < 0.4 and iteration > 5:
            new_hb = self._two_opt_light(new_hb)

        return new_hb

    def _territorial_foraging(self, hummingbird: List[int]) -> List[int]:
        """Operador de forrageamento territorial (swap)"""
        import random
        new_hb = hummingbird.copy()
        if len(new_hb) > 1:
            i, j = random.sample(range(len(new_hb)), 2)
            new_hb[i], new_hb[j] = new_hb[j], new_hb[i]
        return new_hb

    def _migration_foraging(self, hummingbird: List[int]) -> List[int]:
        """Operador de migração (reverse segment)"""
        import random
        new_hb = hummingbird.copy()
        if len(new_hb) > 2:
            i = random.randint(0, len(new_hb) - 2)
            j = random.randint(i + 1, len(new_hb) - 1)
            new_hb[i:j+1] = reversed(new_hb[i:j+1])
        return new_hb

    def _two_opt_light(self, solution: List[int]) -> List[int]:
        """Busca local 2-opt leve - versão rápida para uso nos operadores"""
        import random
        if len(solution) < 4:
            return solution

        new_sol = solution.copy()
        attempts = min(3, len(solution) // 4)

        for _ in range(attempts):
            i = random.randint(0, len(solution) - 3)
            j = random.randint(i + 2, min(i + 5, len(solution) - 1))
            if j > i + 1:
                new_sol[i:j+1] = reversed(new_sol[i:j+1])
                break

        return new_sol

    def _intensive_two_opt(self, solution: List[int], demands: List[int],
                           capacity: int, autonomy: float, dm: np.ndarray,
                           max_time: float = 2.0, commodity_capacities: Dict = None, commodities: List = None) -> List[int]:
        """
        Polimento intensivo, chamado só quando o D-AHA encontra uma nova melhor
        solução. Filtro barato (tour-length delta) antes de rodar o Split (caro).
        """
        import time as _time
        start = _time.time()

        best = solution.copy()
        _, best_fitness = self._evaluate(best, demands, capacity, autonomy, dm, commodity_capacities, commodities)

        improved = True
        max_no_improve = 2
        no_improve = 0

        def tour_length(seq: List[int]) -> float:
            s = 0.0
            for k in range(len(seq) - 1):
                s += dm[seq[k], seq[k+1]]
            return s

        best_tour_len = tour_length(best)

        neighbor_limit = min(50, max(5, int(len(best) ** 0.5 * 5)))

        while improved and no_improve < max_no_improve:
            if _time.time() - start > max_time:
                break

            improved = False
            for i in range(len(best) - 1):
                if _time.time() - start > max_time:
                    break

                j_end = min(len(best), i + 2 + neighbor_limit)
                for j in range(i + 2, j_end):
                    a = best[i]
                    b = best[j]
                    prev = best[i-1] if i > 0 else None
                    nex = best[j+1] if j + 1 < len(best) else None

                    def edge(u, v):
                        return dm[u, v] if u is not None and v is not None else 0.0
                    delta = -edge(prev, a) - edge(b, nex) + edge(prev, b) + edge(a, nex)
                    candidate_tour_len = best_tour_len + delta

                    # Só avalia com Split se o tour-length melhorou
                    if candidate_tour_len < best_tour_len - 1e-6:
                        candidate = best.copy()
                        candidate[i:j+1] = reversed(candidate[i:j+1])
                        _, candidate_fitness = self._evaluate(candidate, demands, capacity, autonomy, dm, commodity_capacities, commodities)

                        if candidate_fitness < best_fitness - 0.01:
                            best = candidate
                            best_fitness = candidate_fitness
                            best_tour_len = candidate_tour_len
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                no_improve += 1
                if _time.time() - start > max_time:
                    break
                # Or-opt (relocate) - versão mais rápida
                for i in range(0, len(best), 2):
                    if _time.time() - start > max_time:
                        break
                    for j in range(0, len(best), 2):
                        if abs(i - j) <= 1:
                            continue
                        candidate = best.copy()
                        node = candidate.pop(i)
                        insert_pos = j if j < i else j - 1
                        insert_pos = min(insert_pos, len(candidate))
                        candidate.insert(insert_pos, node)

                        cand_len = tour_length(candidate)
                        if cand_len < best_tour_len - 1e-6:
                            _, candidate_fitness = self._evaluate(candidate, demands, capacity, autonomy, dm, commodity_capacities, commodities)

                            if candidate_fitness < best_fitness - 0.01:
                                best = candidate
                                best_fitness = candidate_fitness
                                best_tour_len = cand_len
                                improved = True
                                no_improve = 0
                                break
                    if improved:
                        break

        return best

    # ================================================================
    # BUSCA LOCAL INTER-ROTA (granular neighborhood k=20)
    # Aplicada como pós-processamento nas rotas finais do D-AHA.
    # ================================================================

    def _build_neighbor_list(self, routes, dm, k):
        """Pre-computa lista dos k vizinhos mais próximos de cada cliente."""
        all_clients = [c for r in routes for c in r]
        if not all_clients:
            return {}
        neighbors = {}
        for c in all_clients:
            others = [(dm[c, other], other) for other in all_clients if other != c]
            others.sort()
            neighbors[c] = [other for _, other in others[:k]]
        return neighbors

    def _swap_inter(self, routes, demands, capacity, autonomy, dm, neighbors, max_time):
        """SWAP inter-rota: troca cliente X (rota A) com cliente Y (rota B)."""
        import time as _time
        start = _time.time()
        improved_any = False

        client_pos = {c: (r_idx, pos)
                      for r_idx, route in enumerate(routes)
                      for pos, c in enumerate(route)}

        improved = True
        while improved:
            improved = False
            if _time.time() - start > max_time:
                break
            for x, (a_idx, x_pos) in list(client_pos.items()):
                if _time.time() - start > max_time:
                    break
                route_a = routes[a_idx]
                for y in neighbors.get(x, []):
                    if y == x:
                        continue
                    pos_y = client_pos.get(y)
                    if pos_y is None:
                        continue
                    b_idx, y_pos = pos_y
                    if b_idx == a_idx:
                        continue
                    route_b = routes[b_idx]
                    new_a = list(route_a); new_a[x_pos] = y
                    new_b = list(route_b); new_b[y_pos] = x
                    if route_demand(new_a, demands) > capacity or route_demand(new_b, demands) > capacity:
                        continue
                    da = route_distance(new_a, dm); db = route_distance(new_b, dm)
                    if da > autonomy or db > autonomy:
                        continue
                    if da + db < route_distance(route_a, dm) + route_distance(route_b, dm) - 1e-6:
                        routes[a_idx] = new_a
                        routes[b_idx] = new_b
                        client_pos[x] = (b_idx, y_pos)
                        client_pos[y] = (a_idx, x_pos)
                        improved = True
                        improved_any = True
                        break
                if improved:
                    break
        return routes, improved_any

    def _or_opt_inter(self, routes, demands, capacity, autonomy, dm, neighbors, max_time):
        """Or-opt inter-rota: move cliente X (rota A) para após vizinho Y (rota B)."""
        import time as _time
        start = _time.time()
        improved_any = False

        client_pos = {c: (r_idx, pos)
                      for r_idx, route in enumerate(routes)
                      for pos, c in enumerate(route)}

        improved = True
        while improved:
            improved = False
            if _time.time() - start > max_time:
                break
            for x, (a_idx, x_pos) in list(client_pos.items()):
                if _time.time() - start > max_time:
                    break
                route_a = routes[a_idx]
                if len(route_a) <= 1:
                    continue
                for y in neighbors.get(x, []):
                    if y == x:
                        continue
                    pos_y = client_pos.get(y)
                    if pos_y is None:
                        continue
                    b_idx, y_pos = pos_y
                    if b_idx == a_idx:
                        continue
                    route_b = routes[b_idx]
                    new_a = list(route_a); new_a.pop(x_pos)
                    new_b = list(route_b); new_b.insert(y_pos + 1, x)
                    if route_demand(new_b, demands) > capacity:
                        continue
                    da = route_distance(new_a, dm); db = route_distance(new_b, dm)
                    if db > autonomy:
                        continue
                    if da + db < route_distance(route_a, dm) + route_distance(route_b, dm) - 1e-6:
                        routes[a_idx] = new_a
                        routes[b_idx] = new_b
                        client_pos = {c: (r_idx, pos)
                                      for r_idx, route in enumerate(routes)
                                      for pos, c in enumerate(route)}
                        improved = True
                        improved_any = True
                        break
                if improved:
                    break
        return routes, improved_any

    def _select_target_via_vt(self, i, visit_table, fitness_values):
        """
        maior visit level na linha i, desempate por melhor fitness.
        """
        linha = visit_table[i]
        max_vl = max(linha)
        candidatos = [j for j, vl in enumerate(linha) if vl == max_vl and j != i]
        if not candidatos:
            return None
        return min(candidatos, key=lambda j: fitness_values[j])

    def _granular_refinement(self, routes, demands, capacity, autonomy, dm,
                             k_neighbors=20, max_time=2.0):
        """
        Refinamento granular: SWAP e Or-opt inter-rota em loop até não melhorar
        ou atingir max_time. Poda por vizinhança .
        """
        import time as _time
        start = _time.time()
        if len(routes) < 2:
            return routes
        neighbors = self._build_neighbor_list(routes, dm, k_neighbors)
        if not neighbors:
            return routes
        for _iter in range(10):
            if _time.time() - start > max_time:
                break
            remaining = max_time - (_time.time() - start)
            if remaining <= 0.1:
                break
            t_each = remaining / 2.0
            routes, swap_ok = self._swap_inter(routes, demands, capacity, autonomy, dm, neighbors, t_each)
            if _time.time() - start > max_time:
                break
            routes, orop_ok = self._or_opt_inter(routes, demands, capacity, autonomy, dm, neighbors, t_each)
            if not (swap_ok or orop_ok):
                break
        return routes

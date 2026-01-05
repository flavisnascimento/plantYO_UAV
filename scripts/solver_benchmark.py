#!/usr/bin/env python3
"""
Solver Benchmark Framework

Framework para comparação de algoritmos de roteamento (CVRP/TSP) para
missões de plantio com UAV.

Solvers suportados:
  - HGS (Vidal, 2022): Hybrid Genetic Search - Estado da arte para CVRP
  - AHA (Zhao et al., 2022): Artificial Hummingbird Algorithm - Metaheurística bio-inspirada
  - João's Solver: [PLACEHOLDER] - A ser integrado

Métricas comparadas:
  - Distância total da solução
  - Número de rotas/viagens
  - Tempo de computação
  - Gap para melhor solução conhecida
  - Violações de restrições

Para dissertação de mestrado - Comparação experimental de solvers para C-SDVRP
"""

import numpy as np
import time
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkInstance:
    """Instância de problema para benchmark"""
    name: str
    distance_matrix: np.ndarray
    demands: List[int]
    capacity: int
    autonomy: float  # Distância máxima por rota (metros)
    num_waypoints: int
    description: str = ""
    optimal_known: Optional[float] = None  # Melhor solução conhecida
    
    @classmethod
    def from_grid(cls, 
                  grid_size_x: float, 
                  grid_size_y: float,
                  spacing: float,
                  base_x: float,
                  base_y: float,
                  margin: float = 2.5,
                  capacity: int = 225,
                  autonomy: float = 2025.0,
                  demand_per_wp: int = 15,
                  name: str = "grid_instance"):
        """
        Cria instância a partir de um grid regular (como usado no plantio)
        
        Args:
            grid_size_x: Largura do campo (metros)
            grid_size_y: Altura do campo (metros)
            spacing: Espaçamento entre waypoints (metros)
            base_x, base_y: Posição da base
            margin: Margem das bordas
            capacity: Capacidade do veículo
            autonomy: Autonomia em metros
            demand_per_wp: Demanda por waypoint
            name: Nome da instância
        """
        # Gera waypoints em grid
        waypoints = [(base_x, base_y)]  # Índice 0 = base/depósito
        
        x = margin
        while x <= grid_size_x - margin:
            y = margin
            while y <= grid_size_y - margin:
                waypoints.append((x, y))
                y += spacing
            x += spacing
        
        n = len(waypoints)
        
        # Matriz de distâncias
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = waypoints[i][0] - waypoints[j][0]
                    dy = waypoints[i][1] - waypoints[j][1]
                    distance_matrix[i, j] = math.sqrt(dx*dx + dy*dy)
        
        # Demandas (base = 0)
        demands = [0] + [demand_per_wp] * (n - 1)
        
        return cls(
            name=name,
            distance_matrix=distance_matrix,
            demands=demands,
            capacity=capacity,
            autonomy=autonomy,
            num_waypoints=n - 1,  # Exclui base
            description=f"Grid {grid_size_x}x{grid_size_y}m, spacing={spacing}m, {n-1} waypoints"
        )


@dataclass
class SolverResult:
    """Resultado de um solver"""
    solver_name: str
    instance_name: str
    routes: List[List[int]]
    total_distance: float
    num_routes: int
    computation_time: float
    feasible: bool = True
    capacity_violations: int = 0
    autonomy_violations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gap_percent(self) -> Optional[float]:
        """Gap percentual para melhor solução conhecida"""
        if 'optimal' in self.metadata and self.metadata['optimal']:
            return ((self.total_distance - self.metadata['optimal']) / 
                    self.metadata['optimal']) * 100
        return None
    
    def to_dict(self) -> Dict:
        return {
            'solver': self.solver_name,
            'instance': self.instance_name,
            'distance': self.total_distance,
            'num_routes': self.num_routes,
            'time_s': self.computation_time,
            'feasible': self.feasible,
            'capacity_violations': self.capacity_violations,
            'autonomy_violations': self.autonomy_violations,
            'gap_percent': self.gap_percent
        }


# =============================================================================
# ABSTRACT BASE SOLVER
# =============================================================================

class BaseSolver(ABC):
    """
    Interface abstrata para solvers de CVRP/VRP
    
    Todos os solvers devem implementar esta interface para
    permitir comparação justa no benchmark.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do solver para identificação"""
        pass
    
    @property
    @abstractmethod
    def reference(self) -> str:
        """Referência bibliográfica do método"""
        pass
    
    @abstractmethod
    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        """
        Resolve o problema CVRP
        
        Args:
            distance_matrix: Matriz de distâncias NxN (índice 0 = depósito)
            demands: Lista de demandas (demands[0] = 0 para depósito)
            capacity: Capacidade do veículo
            autonomy: Distância máxima por rota
            time_limit: Tempo máximo em segundos
            
        Returns:
            SolverResult com a solução encontrada
        """
        pass
    
    def validate_solution(self,
                         result: SolverResult,
                         instance: BenchmarkInstance) -> Dict:
        """Valida uma solução contra as restrições"""
        validation = {
            'valid': True,
            'capacity_violations': [],
            'autonomy_violations': [],
            'missing_waypoints': [],
            'duplicate_waypoints': []
        }
        
        visited = set()
        expected = set(range(1, len(instance.demands)))
        
        for route_idx, route in enumerate(result.routes):
            # Verifica capacidade
            route_demand = sum(instance.demands[wp] for wp in route)
            if route_demand > instance.capacity:
                validation['capacity_violations'].append({
                    'route': route_idx,
                    'demand': route_demand,
                    'capacity': instance.capacity,
                    'excess': route_demand - instance.capacity
                })
                validation['valid'] = False
            
            # Verifica autonomia
            route_dist = self._calc_route_distance(route, instance.distance_matrix)
            if route_dist > instance.autonomy:
                validation['autonomy_violations'].append({
                    'route': route_idx,
                    'distance': route_dist,
                    'autonomy': instance.autonomy,
                    'excess': route_dist - instance.autonomy
                })
                validation['valid'] = False
            
            # Verifica duplicatas
            for wp in route:
                if wp in visited:
                    validation['duplicate_waypoints'].append(wp)
                    validation['valid'] = False
                visited.add(wp)
        
        # Verifica cobertura
        missing = expected - visited
        if missing:
            validation['missing_waypoints'] = list(missing)
            validation['valid'] = False
        
        return validation
    
    def _calc_route_distance(self, route: List[int], dm: np.ndarray) -> float:
        """Calcula distância de uma rota (ida e volta da base)"""
        if not route:
            return 0.0
        
        dist = dm[0, route[0]]  # Base -> primeiro
        for i in range(len(route) - 1):
            dist += dm[route[i], route[i+1]]
        dist += dm[route[-1], 0]  # Último -> base
        
        return dist


# =============================================================================
# HGS SOLVER WRAPPER
# =============================================================================

class HGSSolverBenchmark(BaseSolver):
    """
    Wrapper do HGS (Vidal, 2022) para benchmark
    
    Hybrid Genetic Search - Estado da arte para CVRP
    Características:
      - Algoritmo genético híbrido
      - Busca local com SWAP*
      - Split algorithm para multi-trip
    """
    
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
        return "Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. C&OR."
    
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
        
        # Configura e executa solver
        ap = self.hgs.AlgorithmParameters(timeLimit=time_limit)
        solver = self.hgs.Solver(parameters=ap, verbose=False)
        
        result = solver.solve_cvrp(data)
        
        computation_time = time.time() - start_time
        
        # Extrai rotas
        routes = result.routes if hasattr(result, 'routes') else []
        
        # Calcula distância total
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


# =============================================================================
# AHA SOLVER WRAPPER
# =============================================================================

class AHASolverBenchmark(BaseSolver):
    """
    Artificial Hummingbird Algorithm adaptado para CVRP
    
    Metaheurística bio-inspirada baseada no comportamento
    de forrageamento de beija-flores.
    
    Adaptações para CVRP:
      - Representação por permutação
      - Split para respeitar capacidade/autonomia
      - Operadores: guided, territorial, migration foraging
    """
    
    def __init__(self, population_size: int = 30, max_iterations: int = 100):
        self.population_size = population_size
        self.max_iterations = max_iterations
    
    @property
    def name(self) -> str:
        return "AHA-CVRP"
    
    @property
    def reference(self) -> str:
        return "Zhao, W. et al. (2022). Artificial hummingbird algorithm: A new bio-inspired optimizer. Expert Systems with Applications."
    
    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        
        import random
        
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
        
        # Inicializa população (permutações de waypoints 1..n)
        population = []
        for _ in range(self.population_size):
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            population.append(perm)
        
        # Melhor solução
        best_solution = None
        best_fitness = float('inf')
        
        iteration = 0
        while time.time() - start_time < time_limit and iteration < self.max_iterations:
            # Avalia população
            fitness_values = []
            for individual in population:
                routes = self._split_into_routes(individual, demands, capacity, autonomy, distance_matrix)
                fitness = self._calculate_total_distance(routes, distance_matrix)
                fitness_values.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Atualiza população com operadores AHA
            new_population = []
            
            for i, hummingbird in enumerate(population):
                r = random.random()
                
                if r < 0.33:
                    # Guided foraging
                    new_hb = self._guided_foraging(hummingbird, best_solution, iteration)
                elif r < 0.66:
                    # Territorial foraging
                    new_hb = self._territorial_foraging(hummingbird)
                else:
                    # Migration foraging
                    new_hb = self._migration_foraging(hummingbird)
                
                # Avalia nova solução
                routes = self._split_into_routes(new_hb, demands, capacity, autonomy, distance_matrix)
                new_fitness = self._calculate_total_distance(routes, distance_matrix)
                
                # Aceita se melhor
                if new_fitness < fitness_values[i]:
                    new_population.append(new_hb)
                else:
                    new_population.append(hummingbird)
            
            population = new_population
            iteration += 1
        
        computation_time = time.time() - start_time
        
        # Gera rotas finais
        if best_solution:
            routes = self._split_into_routes(best_solution, demands, capacity, autonomy, distance_matrix)
        else:
            routes = []
        
        total_distance = self._calculate_total_distance(routes, distance_matrix)
        
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
                'capacity': capacity,
                'autonomy': autonomy
            }
        )
    
    def _guided_foraging(self, hummingbird: List[int], best: List[int], iteration: int) -> List[int]:
        """Operador de forrageamento guiado"""
        import random
        new_hb = hummingbird.copy()
        
        decay = 1 - (iteration / self.max_iterations)
        if random.random() < 0.5 * decay and best:
            # Crossover parcial com melhor
            cut = random.randint(1, len(hummingbird) - 1)
            segment = best[:cut]
            remaining = [x for x in hummingbird if x not in segment]
            new_hb = segment + remaining
        
        # 2-opt com probabilidade
        if random.random() < 0.3:
            new_hb = self._two_opt(new_hb)
        
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
    
    def _two_opt(self, solution: List[int]) -> List[int]:
        """Busca local 2-opt"""
        import random
        if len(solution) < 4:
            return solution
        new_sol = solution.copy()
        i = random.randint(0, len(solution) - 2)
        j = random.randint(i + 1, len(solution) - 1)
        new_sol[i:j+1] = reversed(new_sol[i:j+1])
        return new_sol
    
    def _split_into_routes(self, 
                           sequence: List[int], 
                           demands: List[int],
                           capacity: int,
                           autonomy: float,
                           dm: np.ndarray) -> List[List[int]]:
        """
        Split algorithm para dividir sequência em rotas factíveis
        
        Respeita:
          - Capacidade do veículo
          - Autonomia (distância máxima por rota)
        """
        routes = []
        current_route = []
        current_demand = 0
        current_distance = 0.0
        last_wp = 0  # Começa na base
        
        for wp in sequence:
            wp_demand = demands[wp]
            dist_to_wp = dm[last_wp, wp]
            dist_to_base = dm[wp, 0]
            
            # Verifica se cabe na rota atual
            new_demand = current_demand + wp_demand
            new_distance = current_distance + dist_to_wp + dist_to_base
            
            if new_demand <= capacity and new_distance <= autonomy:
                current_route.append(wp)
                current_demand = new_demand
                current_distance = current_distance + dist_to_wp
                last_wp = wp
            else:
                # Fecha rota atual e inicia nova
                if current_route:
                    routes.append(current_route)
                current_route = [wp]
                current_demand = wp_demand
                current_distance = dm[0, wp]  # Base -> wp
                last_wp = wp
        
        # Adiciona última rota
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _calculate_total_distance(self, routes: List[List[int]], dm: np.ndarray) -> float:
        """Calcula distância total de todas as rotas"""
        total = 0.0
        for route in routes:
            total += self._calc_route_distance(route, dm)
        return total


# =============================================================================
# PLACEHOLDER PARA SOLVER DO JOÃO
# =============================================================================

class JoaoSolverBenchmark(BaseSolver):
    """
    [PLACEHOLDER] Solver do João
    
    TODO: Implementar quando tiver acesso ao código/algoritmo
    
    Informações necessárias:
      - Nome do método/algoritmo
      - Referência bibliográfica
      - Código fonte ou API
    """
    
    def __init__(self):
        self.available = False
    
    @property
    def name(self) -> str:
        return "João-Solver"
    
    @property
    def reference(self) -> str:
        return "[PLACEHOLDER] - A ser definido"
    
    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        
        # TODO: Implementar quando tiver o código
        return SolverResult(
            solver_name=self.name,
            instance_name=kwargs.get('instance_name', 'unknown'),
            routes=[],
            total_distance=float('inf'),
            num_routes=0,
            computation_time=0,
            feasible=False,
            metadata={'error': 'Solver não implementado ainda'}
        )


# =============================================================================
# NEAREST NEIGHBOR BASELINE
# =============================================================================

class NearestNeighborSolver(BaseSolver):
    """
    Nearest Neighbor heurístico como baseline
    
    Construção gulosa: sempre visita o waypoint mais próximo
    não visitado que caiba na rota atual.
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
                    
                    # Verifica restrições
                    if (current_demand + demands[wp] <= capacity and
                        current_distance + dist_to_wp + dist_to_base <= autonomy):
                        if dist_to_wp < best_dist:
                            best_dist = dist_to_wp
                            best_wp = wp
                
                if best_wp is None:
                    break  # Não cabe mais ninguém
                
                # Adiciona à rota
                route.append(best_wp)
                unvisited.remove(best_wp)
                current_demand += demands[best_wp]
                current_distance += best_dist
                current_pos = best_wp
            
            if route:
                routes.append(route)
        
        computation_time = time.time() - start_time
        
        # Calcula distância total
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


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """
    Executa benchmarks comparativos entre solvers
    """
    
    def __init__(self, output_dir: str = None):
        self.solvers: List[BaseSolver] = []
        self.instances: List[BenchmarkInstance] = []
        self.results: List[SolverResult] = []
        
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.expanduser("~/plantyo_benchmarks")
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_solver(self, solver: BaseSolver):
        """Adiciona solver ao benchmark"""
        self.solvers.append(solver)
        print(f"[BENCHMARK] Solver adicionado: {solver.name}")
        print(f"            Referência: {solver.reference}")
    
    def add_instance(self, instance: BenchmarkInstance):
        """Adiciona instância de teste"""
        self.instances.append(instance)
        print(f"[BENCHMARK] Instância adicionada: {instance.name}")
        print(f"            {instance.description}")
    
    def add_standard_instances(self):
        """Adiciona instâncias padrão para testes"""
        # Pequena (10x10m, ~9 waypoints)
        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=15.0, grid_size_y=15.0, spacing=5.0,
            base_x=7.5, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name="small_10x10"
        ))
        
        # Média (50x50m, ~81 waypoints)
        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=50.0, grid_size_y=50.0, spacing=5.0,
            base_x=25.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name="medium_50x50"
        ))
        
        # Grande (100x100m, ~361 waypoints)
        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=100.0, grid_size_y=100.0, spacing=5.0,
            base_x=50.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name="large_100x100"
        ))
        
        # Muito grande (200x200m) - teste de escalabilidade
        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=200.0, grid_size_y=200.0, spacing=5.0,
            base_x=100.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name="xlarge_200x200"
        ))
    
    def run(self, 
            time_limit: float = 30.0, 
            num_runs: int = 1,
            verbose: bool = True) -> Dict:
        """
        Executa benchmark completo
        
        Args:
            time_limit: Tempo limite por solver por instância
            num_runs: Número de execuções para média
            verbose: Imprime progresso
            
        Returns:
            Dict com resultados agregados
        """
        if verbose:
            print("\n" + "="*70)
            print("BENCHMARK DE SOLVERS PARA C-SDVRP")
            print("="*70)
            print(f"Solvers: {[s.name for s in self.solvers]}")
            print(f"Instâncias: {[i.name for i in self.instances]}")
            print(f"Time limit: {time_limit}s | Runs: {num_runs}")
            print("="*70 + "\n")
        
        all_results = []
        
        for instance in self.instances:
            if verbose:
                print(f"\n--- Instância: {instance.name} ---")
                print(f"    Waypoints: {instance.num_waypoints}")
                print(f"    Capacidade: {instance.capacity}")
                print(f"    Autonomia: {instance.autonomy}m")
            
            for solver in self.solvers:
                run_results = []
                
                for run in range(num_runs):
                    if verbose:
                        print(f"    [{solver.name}] Run {run+1}/{num_runs}...", end=" ")
                    
                    result = solver.solve(
                        distance_matrix=instance.distance_matrix,
                        demands=instance.demands,
                        capacity=instance.capacity,
                        autonomy=instance.autonomy,
                        time_limit=time_limit,
                        instance_name=instance.name
                    )
                    
                    # Valida solução
                    validation = solver.validate_solution(result, instance)
                    result.feasible = validation['valid']
                    result.capacity_violations = len(validation['capacity_violations'])
                    result.autonomy_violations = len(validation['autonomy_violations'])
                    
                    if instance.optimal_known:
                        result.metadata['optimal'] = instance.optimal_known
                    
                    run_results.append(result)
                    
                    if verbose:
                        print(f"Dist: {result.total_distance:.1f}m | "
                              f"Rotas: {result.num_routes} | "
                              f"Tempo: {result.computation_time:.3f}s")
                
                # Média das execuções
                if num_runs > 1:
                    avg_result = self._average_results(run_results)
                    all_results.append(avg_result)
                else:
                    all_results.append(run_results[0])
        
        self.results = all_results
        
        # Gera relatório
        report = self._generate_report()
        
        return report
    
    def _average_results(self, results: List[SolverResult]) -> SolverResult:
        """Calcula média de múltiplas execuções"""
        avg_distance = np.mean([r.total_distance for r in results])
        avg_time = np.mean([r.computation_time for r in results])
        avg_routes = np.mean([r.num_routes for r in results])
        
        return SolverResult(
            solver_name=results[0].solver_name,
            instance_name=results[0].instance_name,
            routes=results[0].routes,  # Usa rotas da primeira execução
            total_distance=avg_distance,
            num_routes=int(np.round(avg_routes)),
            computation_time=avg_time,
            feasible=all(r.feasible for r in results),
            metadata={
                'num_runs': len(results),
                'std_distance': np.std([r.total_distance for r in results]),
                'min_distance': min(r.total_distance for r in results),
                'max_distance': max(r.total_distance for r in results)
            }
        )
    
    def _generate_report(self) -> Dict:
        """Gera relatório comparativo"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'solvers': [s.name for s in self.solvers],
            'instances': [i.name for i in self.instances],
            'results': [r.to_dict() for r in self.results],
            'summary': {}
        }
        
        # Resumo por solver
        for solver in self.solvers:
            solver_results = [r for r in self.results if r.solver_name == solver.name]
            
            if solver_results:
                report['summary'][solver.name] = {
                    'avg_distance': np.mean([r.total_distance for r in solver_results]),
                    'avg_time': np.mean([r.computation_time for r in solver_results]),
                    'avg_routes': np.mean([r.num_routes for r in solver_results]),
                    'feasible_rate': sum(1 for r in solver_results if r.feasible) / len(solver_results)
                }
        
        return report
    
    def print_comparison_table(self):
        """Imprime tabela comparativa formatada"""
        print("\n" + "="*90)
        print("TABELA COMPARATIVA")
        print("="*90)
        
        # Cabeçalho
        header = f"{'Instância':<20}"
        for solver in self.solvers:
            header += f" | {solver.name:>15}"
        print(header)
        print("-"*90)
        
        # Por instância - Distância
        print("\n📏 DISTÂNCIA TOTAL (metros):")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            best_dist = float('inf')
            
            # Encontra melhor
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result and result.total_distance < best_dist:
                    best_dist = result.total_distance
            
            # Imprime com destaque no melhor
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    if abs(result.total_distance - best_dist) < 0.01:
                        row += f" | {result.total_distance:>13.1f}*"  # Melhor
                    else:
                        gap = ((result.total_distance - best_dist) / best_dist) * 100
                        row += f" | {result.total_distance:>10.1f}(+{gap:.0f}%)"
                else:
                    row += f" | {'N/A':>15}"
            print(row)
        
        # Por instância - Tempo
        print("\n⏱️  TEMPO DE COMPUTAÇÃO (segundos):")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" | {result.computation_time:>15.3f}"
                else:
                    row += f" | {'N/A':>15}"
            print(row)
        
        # Por instância - Rotas
        print("\n🚁 NÚMERO DE ROTAS:")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" | {result.num_routes:>15}"
                else:
                    row += f" | {'N/A':>15}"
            print(row)
        
        print("\n" + "="*90)
        print("* = Melhor resultado para a instância")
        print("="*90)
    
    def save_results(self, filename: str = None):
        """Salva resultados em JSON"""
        if filename is None:
            filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        report = self._generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[BENCHMARK] Resultados salvos em: {filepath}")
        return filepath
    
    def export_latex_table(self, filename: str = None) -> str:
        """Exporta tabela em formato LaTeX para dissertação"""
        if filename is None:
            filename = f"benchmark_table_{time.strftime('%Y%m%d')}.tex"
        
        filepath = os.path.join(self.output_dir, filename)
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Comparação de Solvers para C-SDVRP}")
        latex.append(r"\label{tab:solver_comparison}")
        
        # Configura colunas
        cols = "l" + "r" * len(self.solvers)
        latex.append(r"\begin{tabular}{" + cols + "}")
        latex.append(r"\toprule")
        
        # Cabeçalho
        header = "Instância"
        for solver in self.solvers:
            header += f" & {solver.name}"
        header += r" \\"
        latex.append(header)
        latex.append(r"\midrule")
        
        # Subtabela: Distância
        latex.append(r"\multicolumn{" + str(len(self.solvers)+1) + r"}{c}{\textbf{Distância Total (m)}} \\")
        latex.append(r"\midrule")
        
        for instance in self.instances:
            row = instance.name.replace("_", r"\_")
            best_dist = float('inf')
            
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result and result.total_distance < best_dist:
                    best_dist = result.total_distance
            
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    if abs(result.total_distance - best_dist) < 0.01:
                        row += f" & \\textbf{{{result.total_distance:.1f}}}"
                    else:
                        row += f" & {result.total_distance:.1f}"
                else:
                    row += " & --"
            row += r" \\"
            latex.append(row)
        
        latex.append(r"\midrule")
        
        # Subtabela: Tempo
        latex.append(r"\multicolumn{" + str(len(self.solvers)+1) + r"}{c}{\textbf{Tempo de Computação (s)}} \\")
        latex.append(r"\midrule")
        
        for instance in self.instances:
            row = instance.name.replace("_", r"\_")
            for solver in self.solvers:
                result = next((r for r in self.results 
                              if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" & {result.computation_time:.3f}"
                else:
                    row += " & --"
            row += r" \\"
            latex.append(row)
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        latex_content = "\n".join(latex)
        
        with open(filepath, 'w') as f:
            f.write(latex_content)
        
        print(f"[BENCHMARK] Tabela LaTeX salva em: {filepath}")
        return latex_content


# =============================================================================
# MAIN - EXECUÇÃO DO BENCHMARK
# =============================================================================

def main():
    """Executa benchmark completo"""
    print("\n" + "="*70)
    print("BENCHMARK DE SOLVERS PARA PLANTIO COM UAV")
    print("Dissertação de Mestrado - C-SDVRP")
    print("="*70)
    
    # Cria benchmark runner
    runner = BenchmarkRunner()
    
    # Adiciona solvers
    runner.add_solver(NearestNeighborSolver())  # Baseline
    runner.add_solver(AHASolverBenchmark(population_size=30, max_iterations=100))
    runner.add_solver(HGSSolverBenchmark())
    # runner.add_solver(JoaoSolverBenchmark())  # Descomentar quando implementado
    
    # Adiciona instâncias de teste
    runner.add_standard_instances()
    
    # Executa benchmark
    report = runner.run(
        time_limit=30.0,
        num_runs=1,  # Aumentar para 5 em experimentos finais
        verbose=True
    )
    
    # Imprime tabela comparativa
    runner.print_comparison_table()
    
    # Salva resultados
    runner.save_results()
    
    # Exporta tabela LaTeX
    runner.export_latex_table()
    
    print("\n[BENCHMARK] Concluído!")
    return runner


if __name__ == "__main__":
    runner = main()

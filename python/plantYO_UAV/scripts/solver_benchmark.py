#!/usr/bin/env python3
"""
Framework de Benchmark para Solvers de Roteamento em C-SDVRP

Este módulo implementa um framework abrangente para avaliação comparativa de algoritmos
de otimização combinatorial aplicados ao Problema de Roteamento de Veículos com Capacidade
e Demandas por Commodities (C-SDVRP), com foco em missões de plantio com drones UAV.

Algoritmos Implementados:
  - HGS (Vidal, 2022): Busca Genética Híbrida - Estado da arte para CVRP
  - D-AHA (Zhao et al., 2022): Adaptação Discreta do Artificial Hummingbird Algorithm
  - João-LKH-Optimal: Heurística Lin-Kernighan + Split Ótimo (metaheurística avançada)
  - João-LKH-Greedy: Heurística Lin-Kernighan + Split Guloso (alta velocidade)
  - TSP-Greedy: Vizinho Mais Próximo + Split Ótimo (baseline de referência)

Métricas de Avaliação:
  - Distância total percorrida pelas rotas
  - Número de rotas/viagens necessárias
  - Tempo computacional de execução
  - Gap percentual para melhor solução conhecida
  - Verificação de violações de restrições

Aplicação: Dissertação de Mestrado em Otimização Combinatorial para Reflorestamento com UAV.
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
import sys

# Configuração do ambiente para importação de módulos auxiliares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from grid_generator import GridGenerator, GridConfig, CommodityCapacity, PlantType
    GRID_GENERATOR_AVAILABLE = True
except ImportError:
    GRID_GENERATOR_AVAILABLE = False
    print("[WARNING] grid_generator não disponível. Usando instâncias genéricas.")

try:
    from joao_tsp_solver import JoaoTSPSolver
    JOAO_SOLVER_AVAILABLE = True
except ImportError:
    JOAO_SOLVER_AVAILABLE = False
    print("[WARNING] joao_tsp_solver não disponível. Solver João será pulado.")


# =============================================================================
# ALGORITMO DE SPLIT ÓTIMO (PRINS/BELLMAN)
# =============================================================================

class OptimalSplit:
    """
    Implementação do Algoritmo de Split Ótimo via Bellman em DAG (Prins, 2004).

    Este algoritmo resolve o subproblema de dividir um tour gigante (sequência de clientes)
    em múltiplas rotas factíveis, garantindo otimalidade na divisão de carga sob restrições
    de capacidade e autonomia. Utiliza programação dinâmica em grafo acíclico direcionado (DAG)
    para encontrar a partição ótima.

    Estratégia Route-First, Cluster-Second: Primeiro constrói-se um tour viável para o TSP,
    depois divide-se em rotas respeitando as restrições do CVRP.

    Complexidade: O(n²) no pior caso, mas O(n) em média com bounds eficientes.
    Garantia: Solução ótima para a divisão de rotas dadas as restrições.

    Referências:
        Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle
        routing problem. In Evolutionary Computation, 2004. CEC2004, 1, 187-193.

        Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation
        and SWAP* neighborhood. Computers & Operations Research, 140, 105643.
    """

    @staticmethod
    def split(sequence: List[int],
              demands: List[int],
              capacity: int,
              autonomy: float,
              distance_matrix: np.ndarray,
              commodity_capacities: Dict = None,
              commodities: List = None) -> Tuple[List[List[int]], float]:
        """
        Executa a divisão ótima de um tour gigante em rotas factíveis via Bellman em DAG.

        Esta implementação garante a otimalidade da partição sob as restrições de capacidade
        e autonomia, utilizando programação dinâmica para avaliar todas as possíveis quebras
        do tour. Para instâncias com clientes virtuais (Petris, 2024), as restrições de
        commodities são automaticamente respeitadas pela transformação.

        Args:
            sequence: Sequência ordenada de IDs dos clientes (exclui depósito).
            demands: Vetor de demandas por cliente (índice 0 = depósito, sempre 0).
            capacity: Capacidade máxima de carga por rota (unidades).
            autonomy: Distância máxima percorrida por rota (ida e volta ao depósito).
            distance_matrix: Matriz de distâncias euclidianas NxN (índice 0 = depósito).
            commodity_capacities: Capacidades por tipo de commodity (opcional para C-SDVRP).
            commodities: Mapeamento cliente→commodity (opcional para C-SDVRP).

        Returns:
            Tuple[List[List[int]], float]: (rotas_otimas, custo_total_otimo)
                - rotas_otimas: Lista de rotas, cada rota é uma lista de IDs de clientes
                - custo_total_otimo: Distância total mínima garantida
        """
        n = len(sequence)
        if n == 0:
            return [], 0.0

        # Inicialização da programação dinâmica em DAG
        # V[i] = custo mínimo para processar os primeiros i clientes da sequência
        # Esta formulação garante otimalidade via princípio da otimalidade de Bellman
        V = [float('inf')] * (n + 1)
        V[0] = 0.0

        # Vetor de predecessores para reconstrução das rotas ótimas
        P = [-1] * (n + 1)

        # Iteração sobre todas as posições iniciais possíveis para novas rotas
        for i in range(n):
            if V[i] == float('inf'):
                continue  # Pula estados inalcançáveis (otimização de performance)

            # Inicialização de nova rota potencial começando na posição i
            route_demand = 0
            route_distance = 0.0
            last_customer = 0  # Inicia sempre do depósito

            # Controle de commodities para C-SDVRP tradicional
            # Nota: Para clientes virtuais (Petris, 2024), commodities são pré-processadas
            current_commodity_demands = {}
            if commodity_capacities and commodities is None:
                current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}

            # Expansão da rota: tenta incluir clientes j >= i até violar restrições
            for j in range(i, n):
                customer = sequence[j]
                customer_demand = demands[customer]

                # Cálculos de distâncias para verificação de autonomia
                dist_to_customer = distance_matrix[last_customer, customer]
                dist_to_depot = distance_matrix[customer, 0]

                # Verificação de restrição de capacidade total
                if route_demand + customer_demand > capacity:
                    break  # Rota inviável, interrompe expansão

                # Verificação de restrições por commodity (C-SDVRP)
                # Para clientes virtuais, esta verificação é desnecessária pois a transformação
                # garante que cada cliente virtual respeita as capacidades por commodity internamente
                commodity_feasible = True
                if current_commodity_demands and commodities is None:
                    customer_commodity = commodities[customer]
                    new_commodity_demand = current_commodity_demands[customer_commodity] + customer_demand
                    if new_commodity_demand > commodity_capacities[customer_commodity]:
                        commodity_feasible = False

                if not commodity_feasible:
                    break  # Violação de commodity, interrompe expansão

                # Verificação crítica de autonomia: rota deve ser capaz de retornar ao depósito
                # Calcula distância total da rota considerando ida ao cliente e retorno à base
                if i == j:
                    # Caso especial: primeiro cliente da rota
                    total_route_dist = distance_matrix[0, customer] + dist_to_depot
                else:
                    # Cliente adicional: recalcula distância incremental
                    total_route_dist = (route_distance - distance_matrix[sequence[j-1], 0] +
                                      dist_to_customer + dist_to_depot)

                if total_route_dist > autonomy:
                    break  # Violação de autonomia, interrompe expansão
                
                # Atualização incremental dos acumuladores da rota
                route_demand += customer_demand
                if current_commodity_demands:
                    customer_commodity = commodities[customer]
                    current_commodity_demands[customer_commodity] += customer_demand

                # Atualização da distância acumulada da rota
                if i == j:
                    route_distance = distance_matrix[0, customer] + dist_to_depot
                else:
                    route_distance = total_route_dist
                last_customer = customer

                # Cálculo do custo da rota candidata (ida e volta ao depósito)
                route_cost = route_distance

                # Atualização da programação dinâmica: verifica se esta rota melhora V[j+1]
                new_cost = V[i] + route_cost
                if new_cost < V[j + 1]:
                    V[j + 1] = new_cost
                    P[j + 1] = i  # Registra predecessor para reconstrução

        # Reconstrução das rotas ótimas via backtracking nos predecessores
        routes = []
        j = n
        while j > 0:
            i = P[j]
            if i < 0:
                # Estratégia de fallback: rotas unitárias garantem factibilidade
                # Usada quando a PD não encontra solução válida (casos extremos)
                routes = [[c] for c in sequence]
                return routes, OptimalSplit._calculate_cost(routes, distance_matrix)
            routes.append(sequence[i:j])  # Adiciona rota ótima encontrada
            j = i  # Retrocede para rota anterior

        routes.reverse()  # Inverte para ordem correta (do início ao fim)

        return routes, V[n]  # Retorna solução ótima encontrada

    @staticmethod
    def _calculate_cost(routes: List[List[int]], dm: np.ndarray) -> float:
        """
        Calcula o custo total de um conjunto de rotas (distância total percorrida).

        Args:
            routes: Lista de rotas, cada rota é uma sequência de IDs de clientes.
            dm: Matriz de distâncias NxN.

        Returns:
            float: Distância total percorrida por todas as rotas.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depósito → primeiro cliente
            total += dm[0, route[0]]
            # Entre clientes consecutivos
            for i in range(len(route) - 1):
                total += dm[route[i], route[i + 1]]
            # Último cliente → depósito
            total += dm[route[-1], 0]
        return total
    
    @staticmethod
    def split_greedy(sequence: List[int],
                     demands: List[int],
                     capacity: int,
                     autonomy: float,
                     distance_matrix: np.ndarray,
                     commodity_capacities: Dict = None,
                     commodities: List = None) -> Tuple[List[List[int]], float]:
        """
        Split guloso (para comparação).
        
        Adiciona clientes à rota atual até violar restrições.
        Não garante ótimo, mas é O(n).
        """
        routes = []
        current_route = []
        current_demand = 0
        current_distance = 0.0
        last_pos = 0  # Depósito
        
        # Inicializa contadores de commodity se fornecidos
        # NOTA: Para clientes virtuais, não precisamos rastrear commodities
        current_commodity_demands = {}
        if commodity_capacities and commodities is None:
            current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}
        
        for customer in sequence:
            cust_demand = demands[customer]
            dist_to_cust = distance_matrix[last_pos, customer]
            dist_to_depot = distance_matrix[customer, 0]
            
            new_demand = current_demand + cust_demand
            
            # Verifica capacidades por commodity
            # NOTA: Para clientes virtuais, commodity check é automático via capacity
            commodity_feasible = True
            if current_commodity_demands:
                cust_commodity = commodities[customer]
                new_commodity_demand = current_commodity_demands[cust_commodity] + cust_demand
                if new_commodity_demand > commodity_capacities[cust_commodity]:
                    commodity_feasible = False
            
            # Distância se adicionarmos este cliente
            if not current_route:
                new_distance = distance_matrix[0, customer] + dist_to_depot
            else:
                # Remove retorno anterior, adiciona caminho ao novo cliente e novo retorno
                new_distance = current_distance - distance_matrix[last_pos, 0] + dist_to_cust + dist_to_depot
            
            if new_demand <= capacity and new_distance <= autonomy and commodity_feasible:
                current_route.append(customer)
                current_demand = new_demand
                current_distance = new_distance
                if current_commodity_demands:
                    cust_commodity = commodities[customer]
                    current_commodity_demands[cust_commodity] += cust_demand
                last_pos = customer
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer]
                current_demand = cust_demand
                current_distance = distance_matrix[0, customer] + dist_to_depot
                if current_commodity_demands:
                    current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}
                    cust_commodity = commodities[customer]
                    current_commodity_demands[cust_commodity] = cust_demand
                last_pos = customer
        
        if current_route:
            routes.append(current_route)
        
        cost = OptimalSplit._calculate_cost(routes, distance_matrix)
        return routes, cost


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkInstance:
    """
    Estrutura de dados para instâncias de benchmark em C-SDVRP.

    Esta classe encapsula todas as informações necessárias para definir uma instância
    do Problema de Roteamento de Veículos com Capacidade e Demandas por Commodities
    (C-SDVRP), incluindo a Transformação de Clientes Virtuais (Petris, 2024) que
    converte o problema original em CVRP puro para avaliação justa dos solvers.

    A transformação garante que restrições de commodities sejam automaticamente
    respeitadas quando as restrições de capacidade total são satisfeitas, permitindo
    comparação equitativa entre algoritmos que não suportam commodities nativamente.
    """
    name: str
    distance_matrix: np.ndarray
    demands: List[int]
    capacity: int
    autonomy: float  # Distância máxima por rota (metros)
    num_waypoints: int
    description: str = ""
    optimal_known: Optional[float] = None  # Melhor solução conhecida
    commodity_capacities: Optional[Dict] = None  # Capacidades por commodity
    commodities: Optional[List] = None  # Commodity de cada cliente
    
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
    
    @classmethod
    def from_grid_generator(cls,
                            grid_size_x: float = 100.0,
                            grid_size_y: float = 100.0,
                            waypoint_spacing: float = 2.5,
                            line_spacing: float = 2.5,
                            base_x: float = None,
                            base_y: float = 0.0,
                            margin: float = 2.5,
                            seeds_per_waypoint: int = 15,
                            commodity_capacity: Tuple[int, int, int] = (100, 100, 100),
                            autonomy: float = 2025.0,
                            use_virtual_clients: bool = True,
                            name: str = None) -> 'BenchmarkInstance':
        """
        Cria instância utilizando Transformação de Clientes Virtuais (Petris, 2024).

        Esta implementação representa uma contribuição metodológica crucial para a dissertação,
        aplicando a transformação C-SDVRP → CVRP através de clientes virtuais. Cada commodity
        (erva, arbusto, árvore) é representada por um conjunto de clientes virtuais que respeitam
        internamente as restrições de capacidade por compartimento, garantindo que a satisfação
        da capacidade total implique automaticamente no respeito às restrições de commodities.

        Vantagens da Transformação:
          - Permite avaliação justa de solvers que não suportam commodities nativamente
          - Mantém equivalência teórica com o problema C-SDVRP original
          - Reduz complexidade computacional para algoritmos CVRP padrão

        Args:
            grid_size_x, grid_size_y: Dimensões do talhão de reflorestamento (metros).
            waypoint_spacing: Distância entre waypoints na mesma linha de plantio.
            line_spacing: Distância entre linhas consecutivas de plantio.
            base_x, base_y: Coordenadas da base do drone (depósito).
            margin: Margem de segurança nas bordas do talhão.
            seeds_per_waypoint: Número de sementes necessárias por waypoint.
            commodity_capacity: Capacidades dos compartimentos (erva, arbusto, árvore).
            autonomy: Distância máxima percorrida por rota (ida e volta ao depósito).
            use_virtual_clients: Ativa transformação de clientes virtuais (recomendado).
            name: Identificador único da instância.

        Returns:
            BenchmarkInstance: Instância configurada com transformação aplicada.

        Raises:
            ImportError: Quando GridGenerator não está disponível no ambiente.
        """
        if not GRID_GENERATOR_AVAILABLE:
            raise ImportError("GridGenerator não disponível. Use from_grid() como alternativa.")

        if base_x is None:
            base_x = grid_size_x / 2

        # Configuração da grade de plantio com parâmetros de commodities
        config = GridConfig(
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            waypoint_spacing=waypoint_spacing,
            line_spacing=line_spacing,
            margin=margin,
            base_x=base_x,
            base_y=base_y,
            seeds_per_waypoint=seeds_per_waypoint,
            commodity_capacity=CommodityCapacity(
                erva=commodity_capacity[0],
                arbusto=commodity_capacity[1],
                arvore=commodity_capacity[2]
            )
        )

        # Geração da grade com transformação aplicada
        generator = GridGenerator(config)
        generator.generate()
        
        if use_virtual_clients:
            # Usa clientes virtuais (transformação Petris 2024)
            distance_matrix = generator.get_distance_matrix()
            demands = generator.get_demands()
            commodities = generator.get_commodities()  # Mantém para validação, mas não passa para solvers
            num_clients = len(generator.virtual_clients)
            capacity = config.commodity_capacity.total  # 300 sementes totais
            # NÃO passa commodity_capacities para solvers - eles resolvem CVRP puro
            commodity_capacities = None
            
            desc = (f"C-SDVRP Grid {grid_size_x}x{grid_size_y}m, "
                   f"{num_clients} clientes virtuais (Petris), "
                   f"padrão E-A-Á-A-E")
        else:
            # Usa waypoints individuais
            distance_matrix = generator.get_individual_distance_matrix()
            demands = generator.get_individual_demands()
            # Implementa mapeamento de commodities por waypoint (para validação)
            waypoints = generator.get_all_waypoints()
            commodities = [None] + [wp.plant_type for wp in waypoints]
            num_clients = len(waypoints)
            capacity = generator.get_effective_capacity()
            # Expondo capacidades por commodity para permitir validação quando necessário
            commodity_capacities = {
                PlantType.ERVA: config.commodity_capacity.get(PlantType.ERVA),
                PlantType.ARBUSTO: config.commodity_capacity.get(PlantType.ARBUSTO),
                PlantType.ARVORE: config.commodity_capacity.get(PlantType.ARVORE)
            }
            
            desc = (f"Grid {grid_size_x}x{grid_size_y}m, "
                   f"{num_clients} waypoints individuais")
        
        if name is None:
            name = f"csdvrp_{int(grid_size_x)}x{int(grid_size_y)}"
        
        return cls(
            name=name,
            distance_matrix=distance_matrix,
            demands=demands,
            capacity=capacity,
            autonomy=autonomy,
            num_waypoints=num_clients,
            description=desc,
            commodity_capacities=commodity_capacities,
            commodities=commodities
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
            'commodity_violations': [],
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
            
            # Verifica capacidades por commodity
            # NOTA: Para instâncias com clientes virtuais (C-SDVRP transformado),
            # cada cliente virtual já respeita as capacidades por commodity.
            # Respeitar capacity total (300) automaticamente respeita os compartimentos.
            # Portanto, pulamos a validação de commodity para evitar falsos positivos.
            if instance.commodity_capacities and instance.commodities is not None:
                commodity_demands = {}
                for wp in route:
                    commodity = instance.commodities[wp]
                    if commodity:
                        commodity_demands[commodity] = commodity_demands.get(commodity, 0) + instance.demands[wp]
                
                for commodity, demand in commodity_demands.items():
                    capacity = instance.commodity_capacities.get(commodity, float('inf'))
                    if demand > capacity:
                        validation['commodity_violations'].append({
                            'route': route_idx,
                            'commodity': commodity,
                            'demand': demand,
                            'capacity': capacity,
                            'excess': demand - capacity
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
    Adaptação Discreta do Artificial Hummingbird Algorithm (D-AHA) para C-SDVRP.

    Esta implementação representa uma contribuição metodológica significativa, adaptando o
    Artificial Hummingbird Algorithm (Zhao et al., 2022) - originalmente desenvolvido para
    otimização contínua - para o domínio discreto do C-SDVRP através de operadores de
    permutação (Swap, 2-Opt) em vez de vetores contínuos.

    Estratégia Order-First, Split-Second: O D-AHA utiliza representação por Giant Tour
    (tour gigante), onde cada solução é uma permutação de clientes. A aptidão (fitness)
    é avaliada através da decodificação ótima pelo Split Algorithm (Prins, 2004), que
    divide o tour em rotas factíveis respeitando capacidade e autonomia.

    A busca local intensiva é aplicada seletivamente apenas quando novas soluções
    melhores são descobertas, preservando a natureza exploratória do algoritmo original.

    Operadores de Forrageamento (adaptados para espaço de permutações):
      - Guided Foraging: Crossover parcial com melhor solução + 2-Opt leve
      - Territorial Foraging: Operador Swap para exploração local
      - Migration Foraging: Reversão de segmentos para diversificação

    Características Inovadoras:
      - Preserva inspiração biológica original (comportamento de beija-flores)
      - Busca local como "recompensa" por descobertas promissoras
      - Controle temporal rigoroso para evitar estouro de limite computacional

    Referência:
        Zhao, W., Wang, L., & Zhang, Z. (2022). Artificial hummingbird algorithm:
        A new bio-inspired optimizer with its engineering applications. Computer
        Methods in Applied Mechanics and Engineering, 388, 114194.
    """

    def __init__(self,
                 population_size: int = 50,
                 max_iterations: int = None,
                 use_optimal_split: bool = True):
        """
        Args:
            population_size: Tamanho da população de soluções
            max_iterations: Máximo de iterações (None = sem limite, usa time_limit)
            use_optimal_split: Se True, usa Split Ótimo de Prins (recomendado)
                              Se False, usa split guloso
        """
        self.population_size = population_size
        self.max_iterations = max_iterations  # None = roda até time_limit
        self.use_optimal_split = use_optimal_split
    
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
        """
        Executa o D-AHA utilizando representação por permutação (Giant Tour).

        A aptidão (fitness) de cada solução é avaliada através da decodificação ótima
        pelo Split Algorithm (Prins, 2004), que divide o tour gigante em rotas factíveis
        respeitando restrições de capacidade e autonomia.

        A implementação preserva a filosofia bio-inspirada original, onde beija-flores exploram
        o espaço de soluções através de diferentes estratégias de forrageamento. A população
        representa um "campo de flores" onde cada solução é uma permutação de clientes.
        A busca local intensiva funciona como "exame microscópico" aplicado apenas quando
        uma nova flor promissora (melhor solução) é descoberta.

        Controle Temporal: Reserva 5% do time_limit para processamento final, garantindo
        que o algoritmo respeite os limites computacionais mesmo em sistemas concorrentes.

        Args:
            distance_matrix: Matriz de distâncias euclidianas NxN (índice 0 = depósito).
            demands: Vetor de demandas por cliente (demands[0] = 0 para depósito).
            capacity: Capacidade máxima de carga por rota.
            autonomy: Distância máxima percorrida por rota (ida e volta).
            time_limit: Limite temporal em segundos para execução completa.
            **kwargs: Parâmetros adicionais (instance_name para identificação).

        Returns:
            SolverResult: Solução encontrada com métricas completas de avaliação.
        """
        import random

        start_time = time.time()

        # Simplificação para clientes virtuais: commodities pré-processadas na transformação
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

        # Inicialização da população: "campo de flores" com permutações aleatórias
        # Cada indivíduo representa um beija-flor explorando diferentes sequências de visitas
        population = []
        for _ in range(self.population_size):
            perm = list(range(1, n + 1))
            random.shuffle(perm)
            population.append(perm)

        # Estado global da colônia de beija-flores
        best_solution = None  # Melhor flor descoberta (solução ótima conhecida)
        best_fitness = float('inf')  # Qualidade da melhor flor

        # Controle de convergência e diversificação
        iteration = 0
        max_iter = self.max_iterations if self.max_iterations else float('inf')
        stagnation_counter = 0  # Contador de estagnação para diversificação
        last_best_fitness = float('inf')  # Última melhor fitness para detecção de melhoria

        # Loop principal: simulação do comportamento de forrageamento dos beija-flores
        # Reserva 5% do tempo para processamento final (Split Algorithm + métricas)
        while time.time() - start_time < time_limit * 0.95 and iteration < max_iter:
            # Avaliação da população: cada beija-flor avalia sua "flor" atual
            fitness_values = []
            for individual in population:
                _, fitness = self._evaluate(individual, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)
                fitness_values.append(fitness)

                # Detecção de nova flor promissora (melhor solução global)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()

                    # 🐦 RECOMPENSA POR DESCOBERTA: "Exame Microscópico" da Flor Promissora
                    # Quando um beija-flor encontra uma flor excepcional, aplica-se busca local
                    # intensiva como "recompensa" para extrair todo o potencial do néctar
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
                            # Difusão social: injeta solução melhorada na população
                            worst_idx = fitness_values.index(max(fitness_values))
                            population[worst_idx] = improved.copy()

            # Monitoramento de convergência: detecta estagnação da colônia
            if abs(best_fitness - last_best_fitness) < 0.01:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = best_fitness

            # Controle temporal adaptativo: ratio decrescente para intensificação final
            elapsed_ratio = (time.time() - start_time) / (time_limit * 0.95)
            elapsed_ratio = min(1.0, max(0.0, elapsed_ratio))

            # Estratégia de diversificação: só quando estagnação severa (>30 iterações)
            # Mantém diversidade populacional sem interferir na exploração natural
            if stagnation_counter > 30:
                stagnation_counter = 0
                # Diversificação suave: substitui apenas 1/6 das piores soluções
                sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
                for idx in sorted_indices[-self.population_size // 6:]:
                    new_perm = list(range(1, n + 1))
                    random.shuffle(new_perm)
                    population[idx] = new_perm
                    new_perm = list(range(1, n + 1))
                    random.shuffle(new_perm)
                    population[idx] = new_perm
            
            # Atualiza população com operadores AHA
            new_population = []
            
            for i, hummingbird in enumerate(population):
                # VERIFICAÇÃO DE TEMPO RIGOROSA: corta imediatamente se exceder
                if time.time() - start_time >= time_limit * 0.95:
                    new_population.append(hummingbird)  # Mantém solução atual
                    continue
                
                r = random.random()
                
                if r < 0.33:
                    # Guided foraging
                    new_hb = self._guided_foraging(hummingbird, best_solution, iteration, elapsed_ratio)
                elif r < 0.66:
                    # Territorial foraging
                    new_hb = self._territorial_foraging(hummingbird)
                else:
                    # Migration foraging
                    new_hb = self._migration_foraging(hummingbird)
                
                # Avalia nova solução
                _, new_fitness = self._evaluate(new_hb, demands, capacity, autonomy, distance_matrix, commodity_capacities, commodities)
                
                # Aceita se melhor
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
    
    def _guided_foraging(self, hummingbird: List[int], best: List[int], iteration: int, 
                          elapsed_ratio: float = 0.5) -> List[int]:
        """Operador de forrageamento guiado - Busca local leve integrada"""
        import random
        new_hb = hummingbird.copy()
        
        # Decay baseado no tempo decorrido (0 a 1)
        decay = 1 - elapsed_ratio
        
        # Crossover parcial com melhor solução (aprendizado social)
        if random.random() < 0.5 * decay and best:
            cut = random.randint(1, len(hummingbird) - 1)
            segment = best[:cut]
            remaining = [x for x in hummingbird if x not in segment]
            new_hb = segment + remaining
        
        # Busca local leve: 2-opt simples (não intensivo)
        # Só aplica se não for muito no início (deixa exploração inicial)
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
    
    def _two_opt(self, solution: List[int]) -> List[int]:
        """Busca local 2-opt - versão completa (para polimento intensivo)"""
        import random
        if len(solution) < 4:
            return solution
        new_sol = solution.copy()
        i = random.randint(0, len(solution) - 2)
        j = random.randint(i + 1, len(solution) - 1)
        new_sol[i:j+1] = reversed(new_sol[i:j+1])
        return new_sol
    
    def _two_opt_light(self, solution: List[int]) -> List[int]:
        """Busca local 2-opt leve - versão rápida para uso nos operadores"""
        import random
        if len(solution) < 4:
            return solution
        
        # Tenta apenas algumas trocas aleatórias (não todas)
        new_sol = solution.copy()
        attempts = min(3, len(solution) // 4)  # Máximo 3 tentativas
        
        for _ in range(attempts):
            i = random.randint(0, len(solution) - 3)  # Garante i+2 <= len-1
            j = random.randint(i + 2, min(i + 5, len(solution) - 1))  # Segmentos pequenos
            if j > i + 1:
                new_sol[i:j+1] = reversed(new_sol[i:j+1])
                break  # First improvement simples
        
        return new_sol
    
    def _intensive_two_opt(self, solution: List[int], demands: List[int], 
                           capacity: int, autonomy: float, dm: np.ndarray,
                           max_time: float = 2.0, commodity_capacities: Dict = None, commodities: List = None) -> List[int]:
        """
        🐦 POLIMENTO INTENSIVO - "Exame Microscópico da Flor Promissora"
        
        Só é chamado quando o AHA encontra uma nova melhor solução.
        Aplica busca local completa para extrair todo o potencial da solução.
        
        Esta é a "recompensa" por encontrar uma flor promissora - 
        o beija-flor ganha um exame detalhado para maximizar o néctar.
        """
        import time as _time
        start = _time.time()
        
        best = solution.copy()
        _, best_fitness = self._evaluate(best, demands, capacity, autonomy, dm, commodity_capacities, commodities)
        
        improved = True
        max_no_improve = 2  # Máximo de passadas sem melhoria
        no_improve = 0

        # Versão otimizada: usa um filtro barato (tour-length delta) antes de rodar Split (caro)
        def tour_length(seq: List[int]) -> float:
            s = 0.0
            for k in range(len(seq) - 1):
                s += dm[seq[k], seq[k+1]]
            return s

        best_tour_len = tour_length(best)

        neighbor_limit = min(50, max(5, int(len(best) ** 0.5 * 5)))  # heurística para limitar j

        while improved and no_improve < max_no_improve:
            if _time.time() - start > max_time:
                break

            improved = False
            for i in range(len(best) - 1):
                if _time.time() - start > max_time:
                    break

                # limite de vizinhança para j
                j_end = min(len(best), i + 2 + neighbor_limit)
                for j in range(i + 2, j_end):
                    # calcula delta de 2-opt em O(1)
                    a = best[i]
                    b = best[j]
                    prev = best[i-1] if i > 0 else None
                    nex = best[j+1] if j + 1 < len(best) else None
                    def edge(u,v):
                        return dm[u,v] if u is not None and v is not None else 0.0
                    delta = -edge(prev, a) - edge(b, nex) + edge(prev, b) + edge(a, nex)
                    candidate_tour_len = best_tour_len + delta

                    # filtro rápido: só avalia com Split se o tour-length melhorou
                    if candidate_tour_len < best_tour_len - 1e-6:
                        candidate = best.copy()
                        candidate[i:j+1] = reversed(candidate[i:j+1])
                        _, candidate_fitness = self._evaluate(candidate, demands, capacity, autonomy, dm, commodity_capacities, commodities)

                        if candidate_fitness < best_fitness - 0.01:
                            best = candidate
                            best_fitness = candidate_fitness
                            best_tour_len = candidate_tour_len
                            improved = True
                            break  # First improvement
                if improved:
                    break

            if not improved:
                no_improve += 1
                if _time.time() - start > max_time:
                    break
                # Tenta Or-opt (relocate) - versão mais rápida
                for i in range(0, len(best), 2):  # Salta de 2 em 2 para acelerar
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

                        # usa filtro simples baseado em tour len para or-opt
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


# =============================================================================
# TEMPLATE PARA ADICIONAR NOVO SOLVER
# =============================================================================

class NovoSolverTemplate(BaseSolver):
    """
    TEMPLATE para adicionar um novo solver ao benchmark.
    
    COMO USAR:
    1. Copie esta classe e renomeie (ex: MeuSolverBenchmark)
    2. Implemente os métodos: name, reference, solve
    3. O método solve deve retornar um SolverResult com:
       - routes: Lista de rotas (cada rota é lista de waypoint IDs)
       - total_distance: Distância total percorrida
       - num_routes: Número de rotas
       - computation_time: Tempo de execução
    
    ENTRADA do solve():
       - distance_matrix: np.ndarray NxN (índice 0 = base/depósito)
       - demands: List[int] (demands[0]=0 é a base, demands[1..N] são waypoints)
       - capacity: int (capacidade máxima por rota)
       - autonomy: float (distância máxima por rota em metros)
       - time_limit: float (tempo máximo de execução)
    
    SAÍDA esperada:
       - routes: [[3,5,2], [7,1,4], ...] (IDs dos waypoints, sem incluir base)
       - Cada rota começa e termina na base implicitamente
    """
    
    def __init__(self, parametro1: int = 100, parametro2: float = 0.5):
        """Inicialize seus parâmetros aqui"""
        self.parametro1 = parametro1
        self.parametro2 = parametro2
    
    @property
    def name(self) -> str:
        return "Novo-Solver"  # Altere para o nome do seu algoritmo
    
    @property
    def reference(self) -> str:
        return "Autor (Ano). Título do Paper. Conferência/Journal."
    
    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        """
        Implemente seu algoritmo aqui.
        
        Exemplo de estrutura:
        1. Inicialize sua solução
        2. Execute seu algoritmo de otimização
        3. Divida em rotas respeitando capacity e autonomy
        4. Retorne SolverResult
        """
        import time as time_module
        start_time = time_module.time()
        
        n = len(demands) - 1  # Número de waypoints (exclui depósito)
        
        # =====================================================
        # TODO: IMPLEMENTE SEU ALGORITMO AQUI
        # =====================================================
        
        # Exemplo simples: cria uma rota com todos os waypoints em ordem
        # (substitua isso pelo seu algoritmo real)
        sequence = list(range(1, n + 1))  # Waypoints 1 até n
        
        # Divide em rotas respeitando restrições
        routes = self._split_into_routes(sequence, demands, capacity, autonomy, distance_matrix)
        
        # =====================================================
        # FIM DO SEU ALGORITMO
        # =====================================================
        
        computation_time = time_module.time() - start_time
        
        # Calcula distância total
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
                'parametro1': self.parametro1,
                'parametro2': self.parametro2
            }
        )
    
    def _split_into_routes(self, sequence, demands, capacity, autonomy, dm, commodity_capacities=None, commodities=None):
        """Divide sequência em rotas factíveis (pode reutilizar)"""
        routes = []
        current_route = []
        current_demand = 0
        current_distance = 0.0
        last_wp = 0
        
        # Inicializa contadores de commodity se fornecidos
        if commodity_capacities and commodities:
            current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}
        else:
            current_commodity_demands = None
        
        for wp in sequence:
            wp_demand = demands[wp]
            dist_to_wp = dm[last_wp, wp]
            dist_to_base = dm[wp, 0]
            
            new_demand = current_demand + wp_demand
            new_distance = current_distance + dist_to_wp + dist_to_base
            
            # Verifica capacidades por commodity
            commodity_feasible = True
            if current_commodity_demands is not None:
                wp_commodity = commodities[wp]
                new_commodity_demand = current_commodity_demands[wp_commodity] + wp_demand
                if new_commodity_demand > commodity_capacities[wp_commodity]:
                    commodity_feasible = False
            
            if new_demand <= capacity and new_distance <= autonomy and commodity_feasible:
                current_route.append(wp)
                current_demand = new_demand
                current_distance = current_distance + dist_to_wp
                if current_commodity_demands is not None:
                    wp_commodity = commodities[wp]
                    current_commodity_demands[wp_commodity] += wp_demand
                last_wp = wp
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [wp]
                current_demand = wp_demand
                current_distance = dm[0, wp]
                if current_commodity_demands is not None:
                    current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}
                    wp_commodity = commodities[wp]
                    current_commodity_demands[wp_commodity] = wp_demand
                last_wp = wp
        
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _calculate_total_distance(self, routes, dm):
        """Calcula distância total de todas as rotas"""
        total = 0.0
        for route in routes:
            if route:
                total += dm[0, route[0]]  # Base -> primeiro
                for i in range(len(route) - 1):
                    total += dm[route[i], route[i+1]]
                total += dm[route[-1], 0]  # Último -> base
        return total


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
# TSP SOLVERS - BASELINES COM SPLIT ÓTIMO
# =============================================================================

class TSPGreedySolver(BaseSolver):
    """
    TSP Greedy (Nearest Neighbor) + Split Ótimo
    
    Implementa estratégia Route-First, Cluster-Second: primeiro constrói
    tour gigante usando heurística nearest neighbor para TSP, depois
    aplica Split Ótimo de Prins para dividir em rotas factíveis.
    
    Esta abordagem separa explicitamente a construção do tour (TSP)
    da divisão em rotas (Split), permitindo comparação justa com
    metaheurísticas que usam representação Giant Tour + Split.
    
    Contrasta com D-AHA que usa Order-First, Split-Second.
    """
    
    def __init__(self, use_optimal_split: bool = True):
        """
        Args:
            use_optimal_split: Se True, usa Split Ótimo (Prins/Bellman)
                              Se False, usa split guloso
        """
        self.use_optimal_split = use_optimal_split
    
    @property
    def name(self) -> str:
        return "TSP-Greedy"
    
    @property
    def reference(self) -> str:
        return "Nearest Neighbor TSP + Split de Prins (2004)"
    
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
        
        # Fase 1: Constrói tour gigante usando Nearest Neighbor
        tour = self._build_nn_tour(distance_matrix, n)
        
        # Fase 2: Aplica Split para dividir em rotas
        # Para clientes virtuais, não há necessidade de commodity constraints
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
        """Constrói tour usando Nearest Neighbor"""
        unvisited = set(range(1, n + 1))
        tour = []
        current = 0  # Começa no depósito
        
        while unvisited:
            # Encontra mais próximo
            nearest = min(unvisited, key=lambda x: dm[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour


class TSP2OptSolver(BaseSolver):
    """
    TSP Heurístico (2-opt/3-opt) + Split Ótimo
    
    Implementa estratégia Route-First, Cluster-Second: primeiro constrói
    e otimiza tour gigante usando heurísticas 2-opt/3-opt para TSP,
    depois aplica Split Ótimo de Prins para dividir em rotas factíveis.
    
    ATENÇÃO: Este é um solver HEURÍSTICO, não exato!
    2-opt/3-opt não garantem o ótimo global matemático.
    
    Constrói tour inicial com nearest neighbor, melhora iterativamente
    até convergência ou timeout, depois aplica Split Ótimo.
    
    Use TSPExactSolver para soluções comprovadamente ótimas (instâncias pequenas).
    
    Contrasta com D-AHA que usa Order-First, Split-Second.
    
    Referências:
      - Lin, S. (1965). Computer solutions of the traveling salesman problem
      - Croes, G. A. (1958). A method for solving traveling-salesman problems
    """
    
    def __init__(self, 
                 use_optimal_split: bool = True,
                 max_no_improve: int = 100,
                 use_3opt: bool = False):
        """
        Args:
            use_optimal_split: Se True, usa Split Ótimo
            max_no_improve: Iterações sem melhoria antes de parar
            use_3opt: Se True, usa 3-opt (mais lento, melhor qualidade)
        """
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
        return f"{base} + Split de Prins (2004)"
    
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
        
        # Fase 1: Constrói tour inicial com NN
        tour = self._build_nn_tour(distance_matrix, n)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, distance_matrix)
        
        # Fase 2: Melhora com 2-opt (ou 3-opt)
        no_improve = 0
        iteration = 0
        
        while no_improve < self.max_no_improve:
            if time.time() - start_time > time_limit * 0.8:  # Reserva tempo para split
                break
            
            improved = False
            
            if self.use_3opt and n > 10:
                # 3-opt: mais lento mas melhor qualidade
                new_tour, new_cost = self._three_opt_pass(tour, distance_matrix)
            else:
                # 2-opt: padrão
                new_tour, new_cost = self._two_opt_pass(tour, distance_matrix)
            
            if new_cost < best_cost - 1e-6:
                best_tour = new_tour.copy()
                best_cost = new_cost
                tour = new_tour
                improved = True
                no_improve = 0
            else:
                no_improve += 1
            
            iteration += 1
        
        # Fase 3: Aplica Split Ótimo
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
        """Constrói tour inicial usando Nearest Neighbor"""
        unvisited = set(range(1, n + 1))
        tour = []
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: dm[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def _tour_cost(self, tour: List[int], dm: np.ndarray) -> float:
        """Calcula custo do tour (fechado, começando/terminando no depósito)"""
        if not tour:
            return 0.0
        cost = dm[0, tour[0]]  # Depósito -> primeiro
        for i in range(len(tour) - 1):
            cost += dm[tour[i], tour[i + 1]]
        cost += dm[tour[-1], 0]  # Último -> depósito
        return cost
    
    def _two_opt_pass(self, tour: List[int], dm: np.ndarray) -> Tuple[List[int], float]:
        """
        Um passo completo de 2-opt.
        
        Tenta todas as trocas (i,j) e retorna a melhor.
        2-opt: reverte o segmento entre i e j.
        """
        n = len(tour)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, dm)
        improved = False
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Tenta reverter tour[i+1:j+1]
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                new_cost = self._tour_cost(new_tour, dm)
                
                if new_cost < best_cost - 1e-6:
                    best_tour = new_tour
                    best_cost = new_cost
                    improved = True
        
        return best_tour, best_cost
    
    def _three_opt_pass(self, tour: List[int], dm: np.ndarray) -> Tuple[List[int], float]:
        """
        Um passo de 3-opt simplificado.
        
        3-opt considera 8 reconexões possíveis para cada tripla (i,j,k).
        Esta implementação usa uma versão simplificada.
        """
        n = len(tour)
        best_tour = tour.copy()
        best_cost = self._tour_cost(tour, dm)
        
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n + 1):
                    # Segmentos: A = tour[0:i+1], B = tour[i+1:j+1], C = tour[j+1:k]
                    A = tour[0:i+1]
                    B = tour[i+1:j+1]
                    C = tour[j+1:k] if k <= n else tour[j+1:]
                    D = tour[k:] if k < n else []
                    
                    # Algumas reconexões possíveis
                    candidates = [
                        A + B[::-1] + C + D,      # Reverte B
                        A + B + C[::-1] + D,      # Reverte C
                        A + B[::-1] + C[::-1] + D,  # Reverte ambos
                        A + C + B + D,            # Troca B e C
                    ]
                    
                    for candidate in candidates:
                        if len(candidate) != n:
                            continue
                        cost = self._tour_cost(candidate, dm)
                        if cost < best_cost - 1e-6:
                            best_tour = candidate
                            best_cost = cost
        
        return best_tour, best_cost


# =============================================================================
# TSP EXACT SOLVER (Programação Dinâmica / Held-Karp)
# =============================================================================

class TSPExactSolver(BaseSolver):
    """
    TSP Exato (Programação Dinâmica Held-Karp) + Split Ótimo
    
    Implementa estratégia Route-First, Cluster-Second: primeiro resolve
    TSP exatamente usando programação dinâmica Held-Karp, depois aplica
    Split Ótimo de Prins para dividir em rotas factíveis.
    
    ATENÇÃO: Este solver é EXATO e garante o ótimo global matemático!
    Porém, tem complexidade O(n² * 2^n), então só funciona para
    instâncias pequenas (n ≤ 20 tipicamente).
    
    Para instâncias maiores, tenta usar python-tsp ou Concorde se disponível.
    
    Contrasta com D-AHA que usa Order-First, Split-Second.
    
    Referências:
      - Held, M. & Karp, R. (1962). A dynamic programming approach to sequencing
      - Bellman, R. (1962). Dynamic programming treatment of the travelling salesman
    """
    
    MAX_DP_NODES = 16  # Limite para DP exato (2^16 = 65k estados) - seguro para Python puro
    
    def __init__(self, use_optimal_split: bool = True):
        """
        Args:
            use_optimal_split: Se True, usa Split Ótimo (Prins/Bellman)
        """
        self.use_optimal_split = use_optimal_split
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verifica bibliotecas opcionais disponíveis"""
        self.has_python_tsp = False
        self.has_concorde = False
        
        try:
            from python_tsp.exact import solve_tsp_dynamic_programming
            self.has_python_tsp = True
        except ImportError:
            pass
        
        # Concorde wrapper (opcional)
        try:
            import concorde
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
        
        # Escolhe método baseado no tamanho da instância
        method_used = "unknown"
        is_optimal = False
        
        if n <= self.MAX_DP_NODES:
            # Usa DP exato (Held-Karp)
            tour = self._held_karp_dp(distance_matrix, n)
            method_used = "held-karp-dp"
            is_optimal = True
        elif self.has_python_tsp and n <= 25:
            # Usa python-tsp
            tour = self._python_tsp_solve(distance_matrix, n)
            method_used = "python-tsp"
            is_optimal = True
        else:
            # Fallback: Nearest Neighbor + 2-opt intensivo
            # Para instâncias grandes, não podemos garantir o ótimo
            tour = self._nn_with_intensive_2opt(distance_matrix, n, time_limit * 0.8)
            method_used = "nn-2opt-heuristic"
            is_optimal = False
        
        # Fase 2: Aplica Split Ótimo
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
        """
        Algoritmo Held-Karp (Programação Dinâmica) para TSP.
        
        Complexidade: O(n² * 2^n)
        Memória: O(n * 2^n)
        
        Garante solução ÓTIMA para o TSP.
        """
        # Estados: dp[mask][i] = custo mínimo para visitar nós em 'mask' terminando em 'i'
        # mask é um bitmask onde bit j indica se nó j+1 foi visitado
        
        INF = float('inf')
        num_states = 1 << n
        
        # dp[mask][i] = (custo, predecessor)
        dp = [[INF] * n for _ in range(num_states)]
        parent = [[-1] * n for _ in range(num_states)]
        
        # Inicialização: começar do depósito (0) e ir para cada nó
        for i in range(n):
            node = i + 1  # Nó real (1-indexed)
            mask = 1 << i
            dp[mask][i] = dm[0, node]
        
        # Preenche a tabela DP
        for mask in range(1, num_states):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                # Tenta adicionar cada nó não visitado
                for next_node in range(n):
                    if mask & (1 << next_node):
                        continue
                    
                    new_mask = mask | (1 << next_node)
                    new_cost = dp[mask][last] + dm[last + 1, next_node + 1]
                    
                    if new_cost < dp[new_mask][next_node]:
                        dp[new_mask][next_node] = new_cost
                        parent[new_mask][next_node] = last
        
        # Encontra o melhor final (volta ao depósito)
        full_mask = num_states - 1
        best_cost = INF
        best_last = -1
        
        for i in range(n):
            total_cost = dp[full_mask][i] + dm[i + 1, 0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_last = i
        
        # Reconstrói o tour
        tour = []
        mask = full_mask
        current = best_last
        
        while current != -1:
            tour.append(current + 1)  # Converte para 1-indexed
            prev = parent[mask][current]
            mask ^= (1 << current)
            current = prev
        
        tour.reverse()
        return tour
    
    def _python_tsp_solve(self, dm: np.ndarray, n: int) -> List[int]:
        """Usa biblioteca python-tsp para resolver exatamente"""
        try:
            from python_tsp.exact import solve_tsp_dynamic_programming
            
            # python-tsp espera matriz completa incluindo depósito
            permutation, distance = solve_tsp_dynamic_programming(dm)
            
            # Remove o depósito (índice 0) e ajusta ordem
            tour = [p for p in permutation if p != 0]
            return tour
        except Exception as e:
            # Fallback para DP próprio
            return self._held_karp_dp(dm, n)
    
    def _nn_with_intensive_2opt(self, dm: np.ndarray, n: int, time_limit: float) -> List[int]:
        """Fallback heurístico para instâncias grandes"""
        import time as _time
        start = _time.time()
        
        # Nearest Neighbor
        unvisited = set(range(1, n + 1))
        tour = []
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: dm[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 2-opt intensivo até timeout
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
        """Calcula custo do tour fechado"""
        if not tour:
            return 0.0
        cost = dm[0, tour[0]]
        for i in range(len(tour) - 1):
            cost += dm[tour[i], tour[i + 1]]
        cost += dm[tour[-1], 0]
        return cost


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
        """Adiciona instâncias padrão para testes (grid simples)"""
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
    
    def add_csdvrp_instances(self):
        """
        Adiciona instâncias C-SDVRP reais usando GridGenerator.
        
        Estas instâncias usam a transformação de Petris (2024) com
        clientes virtuais por commodity (Erva, Arbusto, Árvore).
        
        Recomendado para benchmark na dissertação.
        """
        if not GRID_GENERATOR_AVAILABLE:
            print("[WARNING] GridGenerator não disponível. Usando instâncias simples.")
            self.add_standard_instances()
            return
        
        # Instância pequena - 25x25m com espaçamento 2.5m
        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=25.0, grid_size_y=25.0,
            waypoint_spacing=2.5, line_spacing=2.5,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=2025.0,
            name="csdvrp_25x25"
        ))
        
        # Instância média - 50x50m
        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=50.0, grid_size_y=50.0,
            waypoint_spacing=2.5, line_spacing=2.5,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=2025.0,
            name="csdvrp_50x50"
        ))
        
        # Instância grande - 75x75m
        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=75.0, grid_size_y=75.0,
            waypoint_spacing=2.5, line_spacing=2.5,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=2025.0,
            name="csdvrp_75x75"
        ))
        
        # Instância muito grande - 100x100m
        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=100.0, grid_size_y=100.0,
            waypoint_spacing=2.5, line_spacing=2.5,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=2025.0,
            name="csdvrp_100x100"
        ))
        
        # Instância extra grande - 150x150m (teste de escalabilidade)
        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=150.0, grid_size_y=150.0,
            waypoint_spacing=2.5, line_spacing=2.5,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=2025.0,
            name="csdvrp_150x150"
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
        all_individual_results = []  # Armazena resultados individuais de cada run
        
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
                        # NÃO passa commodity_capacities/commodities - CVRP puro
                    )
                    
                    # Valida solução
                    validation = solver.validate_solution(result, instance)
                    result.feasible = validation['valid']
                    result.capacity_violations = len(validation['capacity_violations'])
                    result.autonomy_violations = len(validation['autonomy_violations'])
                    
                    if instance.optimal_known:
                        result.metadata['optimal'] = instance.optimal_known
                    
                    run_results.append(result)
                    all_individual_results.append(result)  # Armazena resultado individual
                    
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
        self.individual_results = all_individual_results  # Armazena para exportação
        
        # Gera relatório
        report = self._generate_report()
        
        return report
    
    def _average_results(self, results: List[SolverResult]) -> SolverResult:
        """Calcula média de múltiplas execuções com estatísticas completas"""
        distances = [r.total_distance for r in results]
        times = [r.computation_time for r in results]
        routes = [r.num_routes for r in results]
        
        avg_distance = np.mean(distances)
        avg_time = np.mean(times)
        avg_routes = np.mean(routes)
        
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
                # Estatísticas de distância
                'distance_mean': float(avg_distance),
                'distance_std': float(np.std(distances)),
                'distance_min': float(np.min(distances)),
                'distance_max': float(np.max(distances)),
                # Estatísticas de tempo
                'time_mean': float(avg_time),
                'time_std': float(np.std(times)),
                'time_min': float(np.min(times)),
                'time_max': float(np.max(times)),
                # Estatísticas de rotas
                'routes_mean': float(avg_routes),
                'routes_std': float(np.std(routes)),
                'routes_min': float(np.min(routes)),
                'routes_max': float(np.max(routes)),
            }
        )
    
    def _generate_report(self) -> Dict:
        """Gera relatório comparativo com dados brutos e agregados"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'solvers': [s.name for s in self.solvers],
            'instances': [i.name for i in self.instances],
            'results': [r.to_dict() for r in self.results],
            'summary': {},
            'summary_by_instance': {},  # ← NOVO: Desvio padrão por instância
            'individual_runs': [r.to_dict() for r in self.individual_results] if hasattr(self, 'individual_results') else []
        }
        
        # Resumo por solver (agregado)
        for solver in self.solvers:
            solver_results = [r for r in self.results if r.solver_name == solver.name]
            
            if solver_results:
                report['summary'][solver.name] = {
                    'avg_distance': np.mean([r.total_distance for r in solver_results]),
                    'std_distance': np.mean([r.metadata.get('distance_std', 0) for r in solver_results]),
                    'avg_time': np.mean([r.computation_time for r in solver_results]),
                    'std_time': np.mean([r.metadata.get('time_std', 0) for r in solver_results]),
                    'avg_routes': np.mean([r.num_routes for r in solver_results]),
                    'feasible_rate': sum(1 for r in solver_results if r.feasible) / len(solver_results)
                }
        
        # Resumo por instância (mostra desvio padrão e variabilidade de cada solver)
        for instance in self.instances:
            instance_results = [r for r in self.results if r.instance_name == instance.name]
            report['summary_by_instance'][instance.name] = {}
            
            for result in instance_results:
                solver_name = result.solver_name
                report['summary_by_instance'][instance.name][solver_name] = {
                    'distance_mean': float(result.total_distance),
                    'distance_std': float(result.metadata.get('distance_std', 0)),
                    'distance_cv': float(result.metadata.get('distance_std', 0) / result.total_distance) if result.total_distance > 0 else 0,  # Coef. Variação
                    'distance_min': float(result.metadata.get('distance_min', 0)),
                    'distance_max': float(result.metadata.get('distance_max', 0)),
                    'time_mean': float(result.computation_time),
                    'time_std': float(result.metadata.get('time_std', 0)),
                    'num_runs': result.metadata.get('num_runs', 1)
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
    """
    Ponto de entrada principal para execução do benchmark experimental.

    Esta função orquestra a avaliação comparativa completa dos algoritmos de otimização
    aplicados ao C-SDVRP, conforme proposto na dissertação de mestrado. O framework
    garante reprodutibilidade dos experimentos através da configuração padronizada
    de solvers, instâncias e métricas de avaliação.

    Configuração Experimental:
      - Solvers: Conjunto representativo incluindo baselines e estado da arte
      - Instâncias: Problemas reais de reflorestamento via Transformação de Clientes Virtuais
      - Métricas: Distância total, tempo computacional, qualidade relativa
      - Replicação: Múltiplas execuções (10) para confiabilidade estatística

    Resultados: Tabelas comparativas em formato acadêmico (LaTeX) para dissertação.
              Dados brutos com estatísticas completas (média, std, min, max).
    """
    print("\n" + "="*70)
    print("BENCHMARK DE SOLVERS PARA PLANTIO COM UAV")
    print("Dissertação de Mestrado - C-SDVRP")
    print("="*70)

    # Inicialização do framework de benchmark
    runner = BenchmarkRunner()

    # Configuração da bateria de solvers para avaliação abrangente
    runner.add_solver(NearestNeighborSolver())     # Baseline de referência
    runner.add_solver(AHASolverBenchmark(population_size=30, max_iterations=100))  # D-AHA proposto
    runner.add_solver(HGSSolverBenchmark())        # Estado da arte (Vidal, 2022)

    # Integração dos solvers TSP com Split Ótimo (João Rafael)
    if JOAO_SOLVER_AVAILABLE:
        runner.add_solver(JoaoTSPSolver(split_strategy='optimal', use_2opt_refinement=True))
        runner.add_solver(JoaoTSPSolver(split_strategy='greedy', use_2opt_refinement=True))
    else:
        print("[WARNING] Solvers do João não disponíveis - pulando")

    # Configuração das instâncias experimentais via Transformação de Clientes Virtuais
    runner.add_csdvrp_instances()

    # Execução experimental com parâmetros controlados
    report = runner.run(
        time_limit=30.0,  # Tempo limite por solver/instância (segundos)
        num_runs=10,       # Número de replicações para análise estatística robusta
        verbose=True      # Saída detalhada para monitoramento experimental
    )

    # Apresentação dos resultados experimentais
    runner.print_comparison_table()  # Tabela formatada para análise visual

    # Persistência dos dados para análise posterior
    runner.save_results()      # Arquivo JSON com dados brutos + estatísticas
    runner.export_latex_table() # Tabela LaTeX para dissertação

    print("\n[BENCHMARK] Experimento concluído com sucesso!")
    print("Resultados salvos para análise estatística e publicação acadêmica.")
    return runner


if __name__ == "__main__":
    runner = main()

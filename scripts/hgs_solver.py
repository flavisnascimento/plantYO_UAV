#!/usr/bin/env python3
"""
HGS-CVRP Solver Wrapper

Wrapper para o solver Hybrid Genetic Search (hygese) para resolver
o problema de roteamento de veículos capacitado (CVRP).

Considera:
  - Capacidade do dispenser (sementes)
  - Autonomia do drone (distância máxima)
  - Retornos à base para recarregar

Referência:
  Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source 
  implementation and SWAP* neighborhood. Computers & Operations Research.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

try:
    import hygese as hgs
    HGS_AVAILABLE = True
except ImportError:
    HGS_AVAILABLE = False
    print("AVISO: hygese não instalado. Use: pip install hygese")


@dataclass
class DroneConfig:
    """Configuração do drone"""
    dispenser_capacity: int = 300  # Sementes totais (100 cada tipo)
    autonomy_meters: float = 2250.0  # Metros úteis de voo
    reserve_percent: float = 0.10  # Reserva para retorno (10%)
    
    @property
    def effective_autonomy(self) -> float:
        """Autonomia efetiva considerando reserva"""
        return self.autonomy_meters * (1 - self.reserve_percent)


@dataclass
class CVRPSolution:
    """Solução do problema CVRP"""
    routes: List[List[int]]  # Lista de rotas (cada rota é lista de waypoint IDs)
    total_distance: float
    num_routes: int  # Número de viagens (retornos à base)
    computation_time: float
    solver_info: Dict
    
    def __str__(self):
        result = []
        result.append(f"Solução CVRP:")
        result.append(f"  Distância total: {self.total_distance:.2f}m")
        result.append(f"  Número de rotas: {self.num_routes}")
        result.append(f"  Tempo de computação: {self.computation_time:.3f}s")
        result.append(f"  Rotas:")
        for i, route in enumerate(self.routes):
            result.append(f"    Rota {i+1}: {len(route)} waypoints")
        return "\n".join(result)


class HGSSolver:
    """
    Solver HGS-CVRP para otimização de rota de plantio
    
    O problema é modelado como CVRP onde:
      - Depósito (índice 0) = Base do drone
      - Clientes (índices 1..N) = Waypoints de plantio
      - Demanda = Sementes necessárias por waypoint
      - Capacidade = Capacidade do dispenser
    """
    
    def __init__(self, drone_config: DroneConfig = None):
        if not HGS_AVAILABLE:
            raise ImportError("hygese não está instalado. Execute: pip install hygese")
        
        self.drone_config = drone_config or DroneConfig()
        self.last_solution: Optional[CVRPSolution] = None
        
    def solve(self, 
              distance_matrix: np.ndarray,
              demands: List[int],
              time_limit: float = 30.0,
              verbose: bool = True,
              use_duration_constraint: bool = False,
              duration_limit: float = None,
              service_times: List[float] = None) -> CVRPSolution:
        """
        Resolve o problema CVRP usando HGS (Vidal, 2022)
        
        Args:
            distance_matrix: Matriz de distâncias (índice 0 = base)
            demands: Lista de demandas (índice 0 = base com demanda 0)
            time_limit: Tempo máximo de execução em segundos
            verbose: Se True, imprime informações do solver
            use_duration_constraint: Se True, usa restrição de duração (autonomia)
            duration_limit: Limite de duração/distância por rota (metros)
            service_times: Tempo de serviço por cliente (opcional)
            
        Returns:
            CVRPSolution com rotas otimizadas
            
        Referência:
            Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source 
            implementation and SWAP* neighborhood. Computers & Operations Research.
        """
        start_time = time.time()
        
        n_locations = len(demands)
        
        if verbose:
            print(f"[HGS] Iniciando solver (Vidal 2022)...")
            print(f"[HGS] Locais: {n_locations} (1 base + {n_locations-1} waypoints)")
            print(f"[HGS] Capacidade: {self.drone_config.dispenser_capacity} sementes")
            print(f"[HGS] Demanda total: {sum(demands)} sementes")
            if use_duration_constraint and duration_limit:
                print(f"[HGS] Restrição de duração: {duration_limit:.0f}m por rota")
        
        # Configurar dados para o solver
        data = dict()
        data['distance_matrix'] = distance_matrix.astype(np.float64)
        data['demands'] = demands
        data['vehicle_capacity'] = self.drone_config.dispenser_capacity
        
        # Calcular número de veículos (fórmula do HGS original: ceil(1.3*demand/cap) + 3)
        total_demand = sum(demands)
        min_vehicles = int(np.ceil(1.3 * total_demand / self.drone_config.dispenser_capacity)) + 3
        data['num_vehicles'] = min_vehicles
        data['depot'] = 0
        
        # Restrição de duração (autonomia) - recurso nativo do HGS
        if use_duration_constraint and duration_limit:
            data['duration_limit'] = duration_limit
            # Tempo de serviço por cliente (tempo de plantio)
            if service_times:
                data['service_times'] = service_times
            else:
                # Default: sem tempo de serviço adicional
                data['service_times'] = [0.0] * n_locations
        
        # Criar solver e configurar
        ap = hgs.AlgorithmParameters(timeLimit=time_limit)
        solver = hgs.Solver(parameters=ap, verbose=verbose)
        
        # Resolver
        result = solver.solve_cvrp(data)
        
        computation_time = time.time() - start_time
        
        # Extrair rotas
        routes = result.routes if hasattr(result, 'routes') else []
        
        # Calcular distância total real
        total_distance = self._calculate_total_distance(routes, distance_matrix)
        
        # Criar solução
        solution = CVRPSolution(
            routes=routes,
            total_distance=total_distance,
            num_routes=len(routes),
            computation_time=computation_time,
            solver_info={
                'cost': result.cost if hasattr(result, 'cost') else total_distance,
                'time_limit': time_limit,
                'capacity': self.drone_config.dispenser_capacity
            }
        )
        
        self.last_solution = solution
        
        if verbose:
            print(f"[HGS] Solução encontrada!")
            print(f"[HGS] Distância total: {total_distance:.2f}m")
            print(f"[HGS] Número de rotas: {len(routes)}")
            print(f"[HGS] Tempo: {computation_time:.3f}s")
        
        return solution
    
    def sort_routes_by_proximity(self, 
                                  solution: CVRPSolution, 
                                  distance_matrix: np.ndarray) -> CVRPSolution:
        """
        Ordena as rotas para que a primeira seja a mais próxima da base.
        Isso economiza bateria no início da missão.
        
        Args:
            solution: Solução original do HGS
            distance_matrix: Matriz de distâncias
            
        Returns:
            Nova solução com rotas ordenadas
        """
        if not solution.routes:
            return solution
        
        # Calcula distância da base ao primeiro cliente de cada rota
        route_distances = []
        for i, route in enumerate(solution.routes):
            if route:
                # Distância base -> primeiro waypoint da rota
                first_wp = route[0]
                dist_to_first = distance_matrix[0, first_wp]
                route_distances.append((dist_to_first, i, route))
            else:
                route_distances.append((float('inf'), i, route))
        
        # Ordena por distância (mais próxima primeiro)
        route_distances.sort(key=lambda x: x[0])
        
        # Extrai rotas ordenadas
        sorted_routes = [r[2] for r in route_distances]
        
        return CVRPSolution(
            routes=sorted_routes,
            total_distance=solution.total_distance,
            num_routes=solution.num_routes,
            computation_time=solution.computation_time,
            solver_info=solution.solver_info
        )
    
    def _calculate_total_distance(self, 
                                   routes: List[List[int]], 
                                   distance_matrix: np.ndarray) -> float:
        """Calcula distância total de todas as rotas"""
        total = 0.0
        
        for route in routes:
            if not route:
                continue
            
            # Base -> primeiro waypoint
            total += distance_matrix[0, route[0]]
            
            # Entre waypoints
            for i in range(len(route) - 1):
                total += distance_matrix[route[i], route[i + 1]]
            
            # Último waypoint -> base
            total += distance_matrix[route[-1], 0]
        
        return total
    
    def get_execution_order(self) -> List[Tuple[str, int]]:
        """
        Retorna ordem de execução com indicação de retornos à base
        
        Returns:
            Lista de tuplas (ação, waypoint_id)
            Ações: 'visit' para visitar waypoint, 'return' para voltar à base
        """
        if not self.last_solution:
            return []
        
        execution_order = []
        
        for route_idx, route in enumerate(self.last_solution.routes):
            for wp_id in route:
                execution_order.append(('visit', wp_id - 1))  # -1 porque índice 0 é base
            
            # Retorno à base após cada rota (exceto a última?)
            if route_idx < len(self.last_solution.routes) - 1:
                execution_order.append(('return', -1))
        
        return execution_order
    
    def validate_solution(self, 
                          demands: List[int],
                          distance_matrix: np.ndarray) -> Dict:
        """
        Valida a solução encontrada
        
        Returns:
            Dict com informações de validação
        """
        if not self.last_solution:
            return {'valid': False, 'error': 'Nenhuma solução disponível'}
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'route_details': []
        }
        
        visited = set()
        
        for route_idx, route in enumerate(self.last_solution.routes):
            route_demand = sum(demands[wp_id] for wp_id in route)
            route_distance = self._calculate_route_distance(route, distance_matrix)
            
            route_info = {
                'route_id': route_idx,
                'waypoints': len(route),
                'demand': route_demand,
                'distance': route_distance
            }
            
            # Verificar capacidade
            if route_demand > self.drone_config.dispenser_capacity:
                validation['valid'] = False
                validation['errors'].append(
                    f"Rota {route_idx}: demanda {route_demand} excede capacidade {self.drone_config.dispenser_capacity}"
                )
            
            # Verificar autonomia (simplificado - só distância da rota)
            if route_distance > self.drone_config.effective_autonomy:
                validation['warnings'].append(
                    f"Rota {route_idx}: distância {route_distance:.1f}m pode exceder autonomia {self.drone_config.effective_autonomy:.1f}m"
                )
            
            # Verificar duplicatas
            for wp_id in route:
                if wp_id in visited:
                    validation['valid'] = False
                    validation['errors'].append(f"Waypoint {wp_id} visitado mais de uma vez")
                visited.add(wp_id)
            
            validation['route_details'].append(route_info)
        
        # Verificar se todos os waypoints foram visitados
        expected = set(range(1, len(demands)))  # Índices 1..N (0 é base)
        missing = expected - visited
        if missing:
            validation['valid'] = False
            validation['errors'].append(f"Waypoints não visitados: {missing}")
        
        return validation
    
    def _calculate_route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Calcula distância de uma única rota"""
        if not route:
            return 0.0
        
        distance = distance_matrix[0, route[0]]  # Base -> primeiro
        for i in range(len(route) - 1):
            distance += distance_matrix[route[i], route[i + 1]]
        distance += distance_matrix[route[-1], 0]  # Último -> base
        
        return distance
    
    def split_routes_by_autonomy(self, 
                                  routes: List[List[int]], 
                                  distance_matrix: np.ndarray) -> List[List[int]]:
        """
        Divide rotas que excedem a autonomia do drone.
        
        Para cada rota que excede autonomia efetiva:
        1. Percorre waypoints acumulando distância
        2. Quando vai exceder, corta e cria nova sub-rota
        3. Considera distância de ida E volta à base
        
        Returns:
            Lista de rotas respeitando autonomia
        """
        autonomy = self.drone_config.effective_autonomy
        new_routes = []
        
        for route in routes:
            if not route:
                continue
            
            route_dist = self._calculate_route_distance(route, distance_matrix)
            
            # Se cabe na autonomia, mantém
            if route_dist <= autonomy:
                new_routes.append(route)
                continue
            
            # Precisa dividir
            current_sub_route = []
            current_distance = 0.0
            last_pos = 0  # Base
            
            for wp_id in route:
                # Distância para ir ao wp + voltar à base
                dist_to_wp = distance_matrix[last_pos, wp_id]
                dist_return = distance_matrix[wp_id, 0]
                
                # Se adicionar este wp + volta excede autonomia
                if current_distance + dist_to_wp + dist_return > autonomy:
                    # Salva sub-rota atual e começa nova
                    if current_sub_route:
                        new_routes.append(current_sub_route)
                    
                    # Nova sub-rota começa da base
                    current_sub_route = [wp_id]
                    current_distance = distance_matrix[0, wp_id]  # Base -> wp
                    last_pos = wp_id
                else:
                    # Adiciona wp à sub-rota atual
                    current_sub_route.append(wp_id)
                    current_distance += dist_to_wp
                    last_pos = wp_id
            
            # Não esquecer última sub-rota
            if current_sub_route:
                new_routes.append(current_sub_route)
        
        return new_routes
    
    def solve_with_autonomy(self,
                            distance_matrix: np.ndarray,
                            demands: List[int],
                            time_limit: float = 30.0,
                            verbose: bool = True,
                            use_native_duration: bool = True) -> CVRPSolution:
        """
        Resolve CVRP considerando autonomia do drone.
        
        Duas abordagens disponíveis:
        
        1. use_native_duration=True (RECOMENDADO):
           Usa restrição de duração nativa do HGS (Vidal 2022)
           O solver considera autonomia durante a otimização
           Gera soluções globalmente melhores
        
        2. use_native_duration=False:
           Resolve CVRP sem autonomia, depois divide rotas
           Pós-processamento heurístico
           Pode gerar mais rotas que o necessário
        
        Args:
            distance_matrix: Matriz de distâncias
            demands: Lista de demandas
            time_limit: Tempo limite do solver
            verbose: Imprimir informações
            use_native_duration: Usar restrição nativa (recomendado)
            
        Returns:
            CVRPSolution com rotas otimizadas
        """
        autonomy = self.drone_config.effective_autonomy
        
        if use_native_duration:
            # Abordagem 1: Restrição nativa do HGS
            if verbose:
                print(f"[HGS] Usando restrição de duração nativa")
                print(f"[HGS] Autonomia efetiva: {autonomy:.0f}m")
            
            solution = self.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                time_limit=time_limit,
                verbose=verbose,
                use_duration_constraint=True,
                duration_limit=autonomy
            )
            
            # Verificar se todas as rotas respeitam autonomia
            if solution and solution.routes:
                violations = 0
                for route in solution.routes:
                    route_dist = self._calculate_route_distance(route, distance_matrix)
                    if route_dist > autonomy:
                        violations += 1
                
                if violations > 0 and verbose:
                    print(f"[HGS] AVISO: {violations} rotas excedem autonomia (solver pode ter relaxado restrição)")
                elif verbose:
                    print(f"[HGS] Todas as {len(solution.routes)} rotas respeitam autonomia ✓")
            
            return solution
        
        else:
            # Abordagem 2: Pós-processamento (método antigo)
            if verbose:
                print(f"[HGS] Usando pós-processamento de autonomia")
            
            # Resolver CVRP normal
            solution = self.solve(distance_matrix, demands, time_limit, verbose)
            
            if not solution or not solution.routes:
                return solution
            
            # Dividir rotas por autonomia
            if verbose:
                print(f"[HGS] Verificando autonomia ({autonomy:.0f}m)...")
            
            original_routes = len(solution.routes)
            new_routes = self.split_routes_by_autonomy(solution.routes, distance_matrix)
            
            if len(new_routes) > original_routes:
                if verbose:
                    print(f"[HGS] Rotas divididas: {original_routes} → {len(new_routes)}")
                
                # Recalcular distância total
                total_distance = self._calculate_total_distance(new_routes, distance_matrix)
                
                solution = CVRPSolution(
                    routes=new_routes,
                    total_distance=total_distance,
                    num_routes=len(new_routes),
                    computation_time=solution.computation_time,
                    solver_info={
                        **solution.solver_info,
                        'autonomy_splits': len(new_routes) - original_routes,
                        'autonomy_limit': autonomy
                    }
                )
                self.last_solution = solution
            else:
                if verbose:
                    print(f"[HGS] Todas as rotas respeitam autonomia ✓")
            
            return solution


def main():
    """Teste do solver HGS com dados do grid generator"""
    from grid_generator import GridGenerator, GridConfig
    
    print("=" * 60)
    print("TESTE DO SOLVER HGS-CVRP")
    print("=" * 60)
    
    # Gerar grid
    config = GridConfig(grid_size_x=100.0, grid_size_y=100.0)
    generator = GridGenerator(config)
    waypoints = generator.generate()
    
    print(f"\nGrid gerado: {len(waypoints)} waypoints")
    
    # Obter dados para o solver
    distance_matrix = generator.get_distance_matrix()
    demands = generator.get_demands()
    
    print(f"Matriz de distâncias: {distance_matrix.shape}")
    print(f"Demandas: {len(demands)} (total: {sum(demands)} sementes)")
    
    # Configurar drone
    drone_config = DroneConfig(
        dispenser_capacity=300,  # 100 sementes de cada tipo
        autonomy_meters=2250.0,  # Autonomia útil
        reserve_percent=0.10
    )
    
    print(f"\nConfiguração do drone:")
    print(f"  Capacidade: {drone_config.dispenser_capacity} sementes")
    print(f"  Autonomia: {drone_config.autonomy_meters}m")
    print(f"  Autonomia efetiva: {drone_config.effective_autonomy}m")
    
    # Resolver
    solver = HGSSolver(drone_config)
    solution = solver.solve(
        distance_matrix=distance_matrix,
        demands=demands,
        time_limit=30.0,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("SOLUÇÃO")
    print("=" * 60)
    print(solution)
    
    # Validar
    print("\n" + "=" * 60)
    print("VALIDAÇÃO")
    print("=" * 60)
    validation = solver.validate_solution(demands, distance_matrix)
    print(f"Válida: {validation['valid']}")
    if validation['errors']:
        print(f"Erros: {validation['errors']}")
    if validation['warnings']:
        print(f"Avisos: {validation['warnings']}")
    
    # Detalhes das rotas
    print("\n" + "=" * 60)
    print("DETALHES DAS ROTAS")
    print("=" * 60)
    for detail in validation['route_details'][:10]:  # Primeiras 10
        print(f"  Rota {detail['route_id']}: {detail['waypoints']} wps, "
              f"{detail['demand']} sementes, {detail['distance']:.1f}m")
    if len(validation['route_details']) > 10:
        print(f"  ... e mais {len(validation['route_details']) - 10} rotas")
    
    # Ordem de execução (amostra)
    print("\n" + "=" * 60)
    print("ORDEM DE EXECUÇÃO (primeiros 20 passos)")
    print("=" * 60)
    execution = solver.get_execution_order()
    for i, (action, wp_id) in enumerate(execution[:20]):
        if action == 'visit':
            wp = waypoints[wp_id]
            print(f"  {i+1}. Visitar WP{wp_id} ({wp.plant_type.value}) em ({wp.x}, {wp.y})")
        else:
            print(f"  {i+1}. RETORNAR À BASE para recarregar")
    if len(execution) > 20:
        print(f"  ... e mais {len(execution) - 20} passos")


if __name__ == "__main__":
    main()

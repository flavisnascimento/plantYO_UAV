#!/usr/bin/env python3
"""Interface abstrata para solvers de CVRP/VRP."""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from result import SolverResult
from instance import BenchmarkInstance
from util_rotas import route_distance


class BaseSolver(ABC):
    """
    Interface abstrata para solvers de CVRP/VRP.

    Todos os solvers devem implementar esta interface para permitir
    comparação justa no benchmark.
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
        Resolve o problema CVRP.

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

            # Verifica capacidades por commodity.
            # Para clientes virtuais (C-SDVRP transformado), cada cliente virtual
            # já respeita as capacidades por compartimento, então respeitar a
            # capacity total (300) já respeita os compartimentos. Só valida quando
            # commodities está explicitamente presente (modo waypoint individual).
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
        """Calcula distância de uma rota (ida e volta da base)."""
        return route_distance(route, dm)

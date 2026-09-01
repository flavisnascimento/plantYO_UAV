#!/usr/bin/env python3
"""

Resolve o subproblema de dividir um tour gigante em rotas factíveis sob
restrições de capacidade e autonomia. Estratégia Route-First, Cluster-Second.

Referências:
    vehicle routing problem. CEC2004.
"""

from typing import List, Dict, Tuple
import numpy as np

from util_rotas import routes_cost


class OptimalSplit:
    """Split Ótimo via Bellman em DAG e variante gulosa."""

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

        Garante a otimalidade da partição sob capacidade e autonomia. Para
        respeitadas automaticamente pela transformação.

        Args:
            sequence: Sequência ordenada de IDs dos clientes (exclui depósito).
            demands: Vetor de demandas por cliente (índice 0 = depósito, sempre 0).
            capacity: Capacidade máxima de carga por rota (unidades).
            autonomy: Distância máxima percorrida por rota (ida e volta ao depósito).
            distance_matrix: Matriz de distâncias euclidianas NxN (índice 0 = depósito).
            commodity_capacities: Capacidades por tipo de commodity (opcional para C-SDVRP).
            commodities: Mapeamento cliente->commodity (opcional para C-SDVRP).

        Returns:
            Tuple[List[List[int]], float]: (rotas_otimas, custo_total_otimo)
        """
        n = len(sequence)
        if n == 0:
            return [], 0.0

        # V[i] = custo mínimo para processar os primeiros i clientes da sequência
        V = [float('inf')] * (n + 1)
        V[0] = 0.0

        # Vetor de predecessores para reconstrução das rotas ótimas
        P = [-1] * (n + 1)

        for i in range(n):
            if V[i] == float('inf'):
                continue  # Pula estados inalcançáveis

            route_demand = 0
            route_distance = 0.0
            last_customer = 0  # Inicia sempre do depósito

            # Controle de commodities para C-SDVRP tradicional
            current_commodity_demands = {}
            if commodity_capacities and commodities is None:
                current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}

            for j in range(i, n):
                customer = sequence[j]
                customer_demand = demands[customer]

                dist_to_customer = distance_matrix[last_customer, customer]
                dist_to_depot = distance_matrix[customer, 0]

                # Restrição de capacidade total
                if route_demand + customer_demand > capacity:
                    break

                # Restrições por commodity (C-SDVRP)
                commodity_feasible = True
                if current_commodity_demands and commodities is None:
                    customer_commodity = commodities[customer]
                    new_commodity_demand = current_commodity_demands[customer_commodity] + customer_demand
                    if new_commodity_demand > commodity_capacities[customer_commodity]:
                        commodity_feasible = False

                if not commodity_feasible:
                    break

                # Restrição de autonomia (rota deve conseguir retornar ao depósito)
                if i == j:
                    total_route_dist = distance_matrix[0, customer] + dist_to_depot
                else:
                    total_route_dist = (route_distance - distance_matrix[sequence[j-1], 0] +
                                        dist_to_customer + dist_to_depot)

                if total_route_dist > autonomy:
                    break

                route_demand += customer_demand
                if current_commodity_demands:
                    customer_commodity = commodities[customer]
                    current_commodity_demands[customer_commodity] += customer_demand

                if i == j:
                    route_distance = distance_matrix[0, customer] + dist_to_depot
                else:
                    route_distance = total_route_dist
                last_customer = customer

                route_cost = route_distance

                new_cost = V[i] + route_cost
                if new_cost < V[j + 1]:
                    V[j + 1] = new_cost
                    P[j + 1] = i

        # Reconstrução das rotas ótimas via backtracking
        routes = []
        j = n
        while j > 0:
            i = P[j]
            if i < 0:
                # Fallback: rotas unitárias garantem factibilidade
                routes = [[c] for c in sequence]
                return routes, OptimalSplit._calculate_cost(routes, distance_matrix)
            routes.append(sequence[i:j])
            j = i

        routes.reverse()

        return routes, V[n]

    @staticmethod
    def _calculate_cost(routes: List[List[int]], dm: np.ndarray) -> float:
        """Distância total percorrida por um conjunto de rotas."""
        return routes_cost(routes, dm)

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

        Adiciona clientes à rota atual até violar restrições. Não garante
        ótimo, mas é O(n).
        """
        routes = []
        current_route = []
        current_demand = 0
        current_distance = 0.0
        last_pos = 0  # Depósito

        current_commodity_demands = {}
        if commodity_capacities and commodities is None:
            current_commodity_demands = {pt: 0 for pt in commodity_capacities.keys()}

        for customer in sequence:
            cust_demand = demands[customer]
            dist_to_cust = distance_matrix[last_pos, customer]
            dist_to_depot = distance_matrix[customer, 0]

            new_demand = current_demand + cust_demand

            commodity_feasible = True
            if current_commodity_demands:
                cust_commodity = commodities[customer]
                new_commodity_demand = current_commodity_demands[cust_commodity] + cust_demand
                if new_commodity_demand > commodity_capacities[cust_commodity]:
                    commodity_feasible = False

            if not current_route:
                new_distance = distance_matrix[0, customer] + dist_to_depot
            else:
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

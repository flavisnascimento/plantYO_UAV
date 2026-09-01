#!/usr/bin/env python3
"""
Utilitários de rota compartilhados pelo pacote solver_benchmark.

Reúne num único lugar contas que antes estavam copiadas em várias classes
(distância de rota, custo de tour, demanda de rota, construção de tour por
Nearest Neighbor). O cálculo é idêntico ao das versões originais; apenas
deixou de estar duplicado.
"""

from typing import List
import numpy as np


def route_distance(route: List[int], dm: np.ndarray) -> float:
    """Distância de uma rota fechada: base -> primeiro -> ... -> último -> base.

    Índice 0 da matriz de distâncias é sempre o depósito/base.
    """
    if not route:
        return 0.0
    d = dm[0, route[0]]                      # base -> primeiro
    for i in range(len(route) - 1):
        d += dm[route[i], route[i + 1]]      # entre clientes consecutivos
    d += dm[route[-1], 0]                    # último -> base
    return d


# Um tour fechado tem exatamente o mesmo cálculo de uma rota fechada.
tour_cost = route_distance


def route_demand(route: List[int], demands: List[int]) -> int:
    """Soma das demandas dos clientes de uma rota."""
    return sum(demands[c] for c in route)


def routes_cost(routes: List[List[int]], dm: np.ndarray) -> float:
    """Distância total percorrida por um conjunto de rotas."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += route_distance(route, dm)
    return total


def build_nn_tour(dm: np.ndarray, n: int) -> List[int]:
    """Constrói um tour gigante por Nearest Neighbor a partir do depósito.

    Args:
        dm: Matriz de distâncias NxN (índice 0 = depósito).
        n: Número de clientes (exclui o depósito).

    Returns:
        Lista de IDs de clientes na ordem de visita (não inclui o depósito).
    """
    unvisited = set(range(1, n + 1))
    tour = []
    current = 0  # Começa no depósito
    while unvisited:
        nearest = min(unvisited, key=lambda x: dm[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return tour

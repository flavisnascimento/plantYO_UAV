#!/usr/bin/env python3
"""
Estrutura de instância de benchmark C-SDVRP e construtores de instância,
"""

import os
import sys
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

# grid_generator é um módulo irmão, um nível acima deste pacote (pasta scripts/).
# Garante que essa pasta esteja no sys.path para o import funcionar em qualquer cwd.
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

try:
    from grid_generator import GridGenerator, GridConfig, CommodityCapacity, PlantType
    GRID_GENERATOR_AVAILABLE = True
except ImportError:
    GRID_GENERATOR_AVAILABLE = False
    print("[WARNING] grid_generator não disponível. Usando instâncias genéricas.")


@dataclass
class BenchmarkInstance:
    """
    Instância de benchmark em C-SDVRP.

    Encapsula tudo o que define uma instância do C-SDVRP, incluindo a
    original em CVRP puro para avaliação justa dos solvers.
    """
    name: str
    distance_matrix: np.ndarray
    demands: List[int]
    capacity: int
    autonomy: float  # Distância máxima por rota (metros)
    num_waypoints: int
    description: str = ""
    optimal_known: Optional[float] = None
    commodity_capacities: Optional[Dict] = None
    commodities: Optional[List] = None

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
        """Cria instância a partir de um grid regular (como usado no plantio)."""
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

        Aplica a transformação C-SDVRP -> CVRP por clientes virtuais. Cada
        commodity (erva, arbusto, árvore) é representada por clientes virtuais
        que respeitam internamente a capacidade por compartimento.
        """
        if not GRID_GENERATOR_AVAILABLE:
            raise ImportError("GridGenerator não disponível. Use from_grid() como alternativa.")

        if base_x is None:
            base_x = grid_size_x / 2

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

        generator = GridGenerator(config)
        generator.generate()

        if use_virtual_clients:
            # Clientes virtuais (transformação de clientes virtuais 2024)
            distance_matrix = generator.get_distance_matrix()
            demands = generator.get_demands()
            commodities = generator.get_commodities()  # Mantém para validação, mas não passa para solvers
            num_clients = len(generator.virtual_clients)
            capacity = config.commodity_capacity.total  # 300 sementes totais
            commodity_capacities = None  # Solvers resolvem CVRP puro

            desc = (f"C-SDVRP Grid {grid_size_x}x{grid_size_y}m, "
                    f"padrão E-A-Á-A-E")
        else:
            # Waypoints individuais
            distance_matrix = generator.get_individual_distance_matrix()
            demands = generator.get_individual_demands()
            waypoints = generator.get_all_waypoints()
            commodities = [None] + [wp.plant_type for wp in waypoints]
            num_clients = len(waypoints)
            capacity = generator.get_effective_capacity()
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

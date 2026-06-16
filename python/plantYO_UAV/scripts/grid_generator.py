#!/usr/bin/env python3
"""
Grid Generator para Reflorestamento com Drone
C-SDVRP - Commoditized Split Delivery VRP

Implementa a transformação de Petris (2024):
- Cada linha é dividida em clientes virtuais por COMMODITY
- Cada commodity (Erva, Arbusto, Árvore) tem capacidade separada
- Cliente virtual = porção de uma commodity que cabe no drone

Referência:
  Petris et al. (2024) - Heurística baseada em Restricted Master para C-SDVRP
  Demonstrou que transformação C-SDVRP → CVRP + HGS supera métodos especializados
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class PlantType(Enum):
    """Tipos de plantas no padrão de reflorestamento"""
    ERVA = "erva"
    ARBUSTO = "arbusto"
    ARVORE = "arvore"


@dataclass
class Waypoint:
    """Representa um waypoint de plantio"""
    id: int
    x: float
    y: float
    line: int
    position_in_line: int
    plant_type: PlantType
    seeds_required: int
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'line': self.line,
            'position_in_line': self.position_in_line,
            'plant_type': self.plant_type.value,
            'seeds_required': self.seeds_required,
            'name': f"{self.plant_type.value}_{self.id}"
        }


@dataclass
class VirtualClient:
    """
    Cliente virtual para transformação C-SDVRP → CVRP (Petris 2024).
    
    Representa uma porção de uma linha que pode ser atendida em uma viagem,
    considerando a capacidade específica de cada commodity.
    
    Exemplo: Uma linha pode gerar:
    - 3 clientes virtuais de Erva (255 sementes / 100 cap = 3)
    - 3 clientes virtuais de Arbusto (240 / 100 = 3)
    - 2 clientes virtuais de Árvore (120 / 100 = 2)
    Total: 8 clientes virtuais para uma linha
    """
    id: int
    line_id: int           # Linha original
    commodity: PlantType   # Tipo de semente
    split_idx: int         # Índice do split (0, 1, 2, ...)
    waypoints: List[Waypoint]
    demand: int            # Sementes desta commodity neste split
    
    @property
    def y(self) -> float:
        return self.waypoints[0].y if self.waypoints else 0.0
    
    @property
    def start_x(self) -> float:
        return min(wp.x for wp in self.waypoints) if self.waypoints else 0.0
    
    @property
    def end_x(self) -> float:
        return max(wp.x for wp in self.waypoints) if self.waypoints else 0.0
    
    @property
    def center_x(self) -> float:
        return (self.start_x + self.end_x) / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.y)
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    @property
    def total_seeds(self) -> int:
        return self.demand
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'line_id': self.line_id,
            'commodity': self.commodity.value,
            'split_idx': self.split_idx,
            'num_waypoints': self.num_waypoints,
            'demand': self.demand,
            'y': self.y,
            'start_x': self.start_x,
            'end_x': self.end_x
        }


# Alias para compatibilidade
PlantingSegment = VirtualClient
PlantingLine = VirtualClient


@dataclass
class CommodityCapacity:
    """Capacidade do drone por commodity"""
    erva: int = 100
    arbusto: int = 100
    arvore: int = 100
    
    def get(self, plant_type: PlantType) -> int:
        if plant_type == PlantType.ERVA:
            return self.erva
        elif plant_type == PlantType.ARBUSTO:
            return self.arbusto
        elif plant_type == PlantType.ARVORE:
            return self.arvore
        return 0
    
    @property
    def total(self) -> int:
        return self.erva + self.arbusto + self.arvore


@dataclass
class GridConfig:
    """Configuração do grid de plantio"""
    grid_size_x: float = 100.0
    grid_size_y: float = 100.0
    waypoint_spacing: float = 2.5   # metros entre waypoints
    line_spacing: float = 2.5        # metros entre linhas
    margin: float = 2.5              # margem das bordas do talhão
    base_x: float = None
    base_y: float = None
    seeds_per_waypoint: int = 15
    commodity_capacity: CommodityCapacity = None
    
    def __post_init__(self):
        if self.base_x is None:
            self.base_x = self.grid_size_x / 2
        if self.base_y is None:
            self.base_y = 0.0
        if self.commodity_capacity is None:
            self.commodity_capacity = CommodityCapacity()
    
    @property
    def vehicle_capacity(self) -> int:
        """Capacidade total (para compatibilidade)"""
        return self.commodity_capacity.total


class GridGenerator:
    """
    Gera grid com transformação C-SDVRP → CVRP (Petris 2024).
    
    Cada linha é transformada em múltiplos clientes virtuais,
    um para cada porção de cada commodity que cabe no drone.
    """
    
    # Padrão E-A-Á-A-E (índices 0,1,2,3,4)
    PLANT_PATTERN = [
        PlantType.ERVA,      # 0
        PlantType.ARBUSTO,   # 1
        PlantType.ARVORE,    # 2
        PlantType.ARBUSTO,   # 3
        PlantType.ERVA       # 4
    ]
    
    def __init__(self, config: GridConfig = None):
        self.config = config or GridConfig()
        self.virtual_clients: List[VirtualClient] = []
        self.waypoints: List[Waypoint] = []
        self.num_lines = 0
        self.waypoints_per_line = 0
        
    def generate(self) -> List[VirtualClient]:
        """
        Gera clientes virtuais usando transformação Petris.
        
        Para cada linha:
        1. Agrupa waypoints por commodity
        2. Divide cada commodity em splits que cabem na capacidade
        3. Cria cliente virtual para cada split
        """
        self.virtual_clients = []
        self.waypoints = []
        
        # Calcular dimensões do grid COM MARGEM
        margin = self.config.margin
        effective_x = self.config.grid_size_x - 2 * margin
        effective_y = self.config.grid_size_y - 2 * margin
        
        self.num_lines = int(effective_y / self.config.line_spacing) + 1
        self.waypoints_per_line = int(effective_x / self.config.waypoint_spacing) + 1
        
        waypoint_id = 0
        client_id = 0
        
        for line_idx in range(self.num_lines):
            y = margin + line_idx * self.config.line_spacing  # Começa com margem
            
            # Gerar waypoints e agrupar por commodity
            line_waypoints = {pt: [] for pt in PlantType}
            
            for pos_idx in range(self.waypoints_per_line):
                x = margin + pos_idx * self.config.waypoint_spacing  # Começa com margem
                pattern_idx = pos_idx % len(self.PLANT_PATTERN)
                plant_type = self.PLANT_PATTERN[pattern_idx]
                
                waypoint = Waypoint(
                    id=waypoint_id,
                    x=x,
                    y=y,
                    line=line_idx,
                    position_in_line=pos_idx,
                    plant_type=plant_type,
                    seeds_required=self.config.seeds_per_waypoint
                )
                
                line_waypoints[plant_type].append(waypoint)
                self.waypoints.append(waypoint)
                waypoint_id += 1
            
            # Para cada commodity, criar splits
            for plant_type in PlantType:
                wps = line_waypoints[plant_type]
                if not wps:
                    continue
                
                capacity = self.config.commodity_capacity.get(plant_type)
                if capacity == 0:
                    continue
                total_demand = len(wps) * self.config.seeds_per_waypoint
                
                # Quantos splits necessários?
                num_splits = int(np.ceil(total_demand / capacity))
                wps_per_split = int(np.ceil(len(wps) / num_splits))
                
                for split_idx in range(num_splits):
                    start_wp = split_idx * wps_per_split
                    end_wp = min((split_idx + 1) * wps_per_split, len(wps))
                    split_wps = wps[start_wp:end_wp]
                    
                    if not split_wps:
                        continue
                    
                    split_demand = len(split_wps) * self.config.seeds_per_waypoint
                    
                    client = VirtualClient(
                        id=client_id,
                        line_id=line_idx,
                        commodity=plant_type,
                        split_idx=split_idx,
                        waypoints=split_wps,
                        demand=split_demand
                    )
                    
                    self.virtual_clients.append(client)
                    client_id += 1
        
        return self.virtual_clients
    
    # Alias para compatibilidade
    @property
    def segments(self) -> List[VirtualClient]:
        return self.virtual_clients
    
    @property
    def lines(self) -> List[VirtualClient]:
        return self.virtual_clients
    
    def get_base_position(self) -> Tuple[float, float]:
        return (self.config.base_x, self.config.base_y)
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Matriz de distâncias entre clientes virtuais.
        Índice 0 = base (depósito), Índices 1..N = clientes virtuais
        """
        if not self.virtual_clients:
            self.generate()
        
        n = len(self.virtual_clients) + 1
        distances = np.zeros((n, n))
        
        base_x, base_y = self.get_base_position()
        
        # Distância da base para cada cliente
        for i, client in enumerate(self.virtual_clients):
            dist = np.sqrt((client.center_x - base_x)**2 + (client.y - base_y)**2)
            distances[0, i + 1] = dist
            distances[i + 1, 0] = dist
        
        # Distância entre clientes
        for i, c1 in enumerate(self.virtual_clients):
            for j, c2 in enumerate(self.virtual_clients):
                if i != j:
                    dist = np.sqrt((c1.center_x - c2.center_x)**2 + 
                                  (c1.y - c2.y)**2)
                    distances[i + 1, j + 1] = dist
        
        return distances
    
    # Aliases para compatibilidade
    def get_segment_distance_matrix(self) -> np.ndarray:
        return self.get_distance_matrix()
    
    def get_line_distance_matrix(self) -> np.ndarray:
        return self.get_distance_matrix()
    
    def get_demands(self) -> List[int]:
        """
        Retorna demandas dos clientes virtuais.
        
        IMPORTANTE: No C-SDVRP transformado, a demanda é por commodity.
        Mas como cada cliente virtual é de uma só commodity e cabe
        na capacidade dessa commodity, usamos a demanda diretamente.
        """
        if not self.virtual_clients:
            self.generate()
        
        demands = [0]  # Base tem demanda 0
        for client in self.virtual_clients:
            demands.append(client.demand)
        
        return demands
    
    def get_commodities(self) -> List[PlantType]:
        """
        Retorna tipos de commodity para cada cliente virtual.
        """
        if not self.virtual_clients:
            self.generate()
        
        commodities = [None]  # Base não tem commodity
        for client in self.virtual_clients:
            commodities.append(client.commodity)
        
        return commodities
    
    # Aliases para compatibilidade
    def get_segment_demands(self) -> List[int]:
        return self.get_demands()
    
    def get_line_demands(self) -> List[int]:
        return self.get_demands()
    
    def get_client_waypoints(self, client_id: int, reverse: bool = False) -> List[Waypoint]:
        """
        Retorna waypoints de um cliente virtual específico.
        
        NOTA: Não ordena os waypoints - deixa o solver HGS decidir a ordem.
        Isso garante comparação justa entre diferentes algoritmos.
        """
        if not self.virtual_clients:
            self.generate()
        
        client = self.virtual_clients[client_id]
        # Retorna na ordem original - solver decide a ordem de visitação
        wps = list(client.waypoints)
        
        if reverse:
            wps.reverse()
        
        return wps
    
    # Aliases para compatibilidade
    def get_segment_waypoints(self, segment_id: int, reverse: bool = False) -> List[Waypoint]:
        return self.get_client_waypoints(segment_id, reverse)
    
    def get_line_waypoints(self, line_id: int, reverse: bool = False) -> List[Waypoint]:
        return self.get_client_waypoints(line_id, reverse)
    
    def get_all_waypoints(self) -> List[Waypoint]:
        """Retorna todos os waypoints do grid"""
        if not self.waypoints:
            self.generate()
        return self.waypoints
    
    # ========================================
    # MÉTODOS PARA WAYPOINTS INDIVIDUAIS (HGS)
    # ========================================
    
    def get_individual_distance_matrix(self) -> np.ndarray:
        """
        Matriz de distâncias para waypoints INDIVIDUAIS.
        Índice 0 = base (depósito), Índices 1..N = waypoints individuais
        
        Use este método para HGS com waypoints individuais.
        """
        if not self.waypoints:
            self.generate()
        
        n = len(self.waypoints) + 1
        distances = np.zeros((n, n))
        
        base_x, base_y = self.get_base_position()
        
        # Distância da base para cada waypoint
        for i, wp in enumerate(self.waypoints):
            dist = np.sqrt((wp.x - base_x)**2 + (wp.y - base_y)**2)
            distances[0, i + 1] = dist
            distances[i + 1, 0] = dist
        
        # Distância entre waypoints
        for i, wp1 in enumerate(self.waypoints):
            for j, wp2 in enumerate(self.waypoints):
                if i != j:
                    dist = np.sqrt((wp1.x - wp2.x)**2 + (wp1.y - wp2.y)**2)
                    distances[i + 1, j + 1] = dist
        
        return distances
    
    def get_individual_demands(self) -> List[int]:
        """
        Demandas para waypoints individuais.
        Cada waypoint tem demanda = seeds_per_waypoint (ex: 15)
        """
        if not self.waypoints:
            self.generate()
        
        demands = [0]  # Base tem demanda 0
        for wp in self.waypoints:
            demands.append(wp.seeds_required)
        
        return demands
    
    def get_effective_capacity(self) -> int:
        """
        Calcula capacidade efetiva baseada na proporção E-A-Á-A-E.
        
        Proporção no padrão: 2 erva : 2 arbusto : 1 árvore (por 5 waypoints)
        
        Capacidade efetiva = min(cap_tipo / proporção_tipo) * sementes_por_wp
        """
        # Contar proporção no padrão
        pattern_counts = {pt: 0 for pt in PlantType}
        for pt in self.PLANT_PATTERN:
            pattern_counts[pt] += 1
        
        pattern_len = len(self.PLANT_PATTERN)
        
        # Calcular quantos ciclos completos cabem
        cycles_per_type = {}
        for pt in PlantType:
            if pattern_counts[pt] > 0:
                capacity = self.config.commodity_capacity.get(pt)
                seeds_per_cycle = pattern_counts[pt] * self.config.seeds_per_waypoint
                cycles_per_type[pt] = capacity // seeds_per_cycle
            else:
                cycles_per_type[pt] = float('inf')
        
        # Limitante é o tipo que acaba primeiro
        min_cycles = min(cycles_per_type.values())
        
        # Waypoints por viagem = ciclos * waypoints_por_ciclo
        waypoints_per_trip = min_cycles * pattern_len
        
        # Capacidade em sementes
        effective_capacity = waypoints_per_trip * self.config.seeds_per_waypoint
        
        return int(effective_capacity)
    
    def get_waypoint_by_id(self, wp_id: int) -> Waypoint:
        """Retorna waypoint pelo ID (índice na lista)"""
        if not self.waypoints:
            self.generate()
        return self.waypoints[wp_id]

    def print_summary(self):
        """Imprime resumo do grid"""
        if not self.virtual_clients:
            self.generate()
        
        total_wps = len(self.waypoints)
        total_seeds = sum(c.demand for c in self.virtual_clients)
        
        # Contar por commodity
        by_commodity = {pt: [] for pt in PlantType}
        for c in self.virtual_clients:
            by_commodity[c.commodity].append(c)
        
        print("=" * 60)
        print("GRID DE PLANTIO - C-SDVRP (Petris 2024)")
        print("=" * 60)
        print(f"Talhão: {self.config.grid_size_x}m x {self.config.grid_size_y}m")
        print(f"Espaçamento: {self.config.waypoint_spacing}m x {self.config.line_spacing}m")
        print(f"Base (depósito): ({self.config.base_x}, {self.config.base_y})")
        print()
        print(f"Capacidade por commodity:")
        print(f"  Erva:    {self.config.commodity_capacity.erva} sementes")
        print(f"  Arbusto: {self.config.commodity_capacity.arbusto} sementes")
        print(f"  Árvore:  {self.config.commodity_capacity.arvore} sementes")
        print(f"  Total:   {self.config.commodity_capacity.total} sementes")
        print()
        print(f"Linhas originais: {self.num_lines}")
        print(f"Waypoints por linha: {self.waypoints_per_line}")
        print(f"Total waypoints: {total_wps}")
        print(f"Total sementes: {total_seeds}")
        print()
        print(f"CLIENTES VIRTUAIS (transformação Petris):")
        print(f"  Total: {len(self.virtual_clients)}")
        for pt in PlantType:
            clients = by_commodity[pt]
            total_demand = sum(c.demand for c in clients)
            print(f"  {pt.value.capitalize():8s}: {len(clients):3d} clientes, {total_demand:5d} sementes")
        print()
        
        # Mostrar algumas linhas de exemplo
        print("Exemplo (primeiras 3 linhas):")
        for line_id in range(min(3, self.num_lines)):
            line_clients = [c for c in self.virtual_clients if c.line_id == line_id]
            info = []
            for pt in PlantType:
                pt_clients = [c for c in line_clients if c.commodity == pt]
                if pt_clients:
                    info.append(f"{pt.value[0].upper()}×{len(pt_clients)}")
            print(f"  Linha {line_id}: {', '.join(info)}")


if __name__ == "__main__":
    # Teste com 2.5m x 2.5m
    config = GridConfig(
        grid_size_x=100.0,
        grid_size_y=100.0,
        waypoint_spacing=2.5,
        line_spacing=2.5,
        base_x=50.0,
        base_y=0.0,
        seeds_per_waypoint=15,
        commodity_capacity=CommodityCapacity(erva=100, arbusto=100, arvore=100)
    )
    
    gen = GridGenerator(config)
    clients = gen.generate()
    gen.print_summary()
    
    print()
    print("=" * 60)
    print("VERIFICAÇÃO")
    print("=" * 60)
    
    demands = gen.get_demands()
    max_demand = max(demands[1:])  # Ignorar base
    
    print(f"Maior demanda: {max_demand}")
    print(f"Capacidade mínima (por commodity): {min(config.commodity_capacity.erva, config.commodity_capacity.arbusto, config.commodity_capacity.arvore)}")
    
    if max_demand <= 100:
        print("✅ Todas as demandas cabem na capacidade da commodity!")
    else:
        print("❌ ERRO: Alguma demanda excede capacidade!")

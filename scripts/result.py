#!/usr/bin/env python3
"""Estrutura de resultado de um solver."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class SolverResult:
    """Resultado de um solver."""
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

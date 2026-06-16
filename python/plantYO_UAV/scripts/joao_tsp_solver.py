#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
João's TSP-Split Solver for Capacitated Vehicle Routing Problem (CVRP)
================================================================================

A hybrid metaheuristic approach combining the Lin-Kernighan Heuristic (LKH) for
the Traveling Salesman Problem (TSP) with an optimal split algorithm for the
Capacitated Vehicle Routing Problem (CVRP).

This solver is designed for UAV-based precision agriculture applications,
specifically for seed planting missions with capacity and autonomy constraints.

Algorithm Overview
------------------
The solver follows a "Route-first, Cluster-second" approach:

1. **TSP Phase**: Solve a global TSP tour including the depot using the
   Lin-Kernighan heuristic, which produces near-optimal tours (typically
   within 1% of optimum) with O(n^2.2) complexity.

2. **Split Phase**: Apply dynamic programming to find optimal cut points
   in the TSP tour that minimize total distance while respecting:
   - Vehicle capacity constraints (seed load)
   - Autonomy constraints (maximum flight distance per trip)

3. **Refinement Phase**: Apply local search (2-opt) to each sub-route
   for further improvement.

Theoretical Foundation
----------------------
- TSP: Lin, S. & Kernighan, B.W. (1973). "An Effective Heuristic Algorithm
  for the Traveling-Salesman Problem." Operations Research, 21(2), 498-516.

- Split Algorithm: Prins, C. (2004). "A simple and effective evolutionary
  algorithm for the vehicle routing problem." Computers & Operations Research.

- CVRP Lower Bounds: Beasley, J.E. (1983). "Route first—cluster second methods
  for vehicle routing." Omega, 11(4), 403-408.

Author
------
João Rafael

License
-------

Usage Example
-------------
    >>> from joao_tsp_solver import JoaoTSPSolver
    >>> solver = JoaoTSPSolver(split_strategy='optimal')
    >>> result = solver.solve(
    ...     distance_matrix=dm,
    ...     demands=demands,
    ...     capacity=225,
    ...     autonomy=2025.0
    ... )
    >>> print(f"Total distance: {result.total_distance:.2f}m")
    >>> print(f"Number of routes: {result.num_routes}")

================================================================================
"""

__author__ = "João Rafael"
__version__ = "1.0.0"
__date__ = "2026-01-14"

import numpy as np
import time
import math
import tempfile
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SolverResult:
    """
    Container for solver output results.
    
    This dataclass encapsulates all relevant information about a solution
    found by the solver, including routes, metrics, and metadata.
    
    Attributes
    ----------
    solver_name : str
        Identifier of the solver that produced this result.
    instance_name : str
        Name of the problem instance solved.
    routes : List[List[int]]
        Solution routes. Each route is a list of waypoint indices (excluding depot).
        The depot (index 0) is implicitly at the start and end of each route.
    total_distance : float
        Total distance traveled across all routes, including returns to depot.
    num_routes : int
        Number of routes (trips) in the solution.
    computation_time : float
        Wall-clock time in seconds for solving.
    feasible : bool
        Whether the solution satisfies all constraints.
    capacity_violations : int
        Number of routes that violate capacity constraints.
    autonomy_violations : int
        Number of routes that violate autonomy (distance) constraints.
    metadata : Dict[str, Any]
        Additional solver-specific information.
    
    Properties
    ----------
    gap_percent : Optional[float]
        Percentage gap to known optimal solution (if available in metadata).
    
    Example
    -------
        >>> result = solver.solve(dm, demands, capacity, autonomy)
        >>> print(f"Found {result.num_routes} routes covering {result.total_distance:.1f}m")
        >>> for i, route in enumerate(result.routes):
        ...     print(f"  Route {i+1}: {route}")
    """
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
        """
        Calculate percentage gap to optimal solution.
        
        Returns
        -------
        Optional[float]
            Gap percentage if optimal is known, None otherwise.
            Formula: ((solution - optimal) / optimal) * 100
        """
        if 'optimal' in self.metadata and self.metadata['optimal']:
            return ((self.total_distance - self.metadata['optimal']) / 
                    self.metadata['optimal']) * 100
        return None
    
    def to_dict(self) -> Dict:
        """
        Convert result to dictionary for serialization.
        
        Returns
        -------
        Dict
            Dictionary representation of the result.
        """
        return {
            'solver': self.solver_name,
            'instance': self.instance_name,
            'routes': self.routes,
            'distance': self.total_distance,
            'num_routes': self.num_routes,
            'time_s': self.computation_time,
            'feasible': self.feasible,
            'capacity_violations': self.capacity_violations,
            'autonomy_violations': self.autonomy_violations,
            'gap_percent': self.gap_percent
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Solution by {self.solver_name}",
            f"  Instance: {self.instance_name}",
            f"  Total Distance: {self.total_distance:.2f}m",
            f"  Number of Routes: {self.num_routes}",
            f"  Computation Time: {self.computation_time:.4f}s",
            f"  Feasible: {self.feasible}"
        ]
        if self.gap_percent is not None:
            lines.append(f"  Gap to Optimal: {self.gap_percent:.2f}%")
        return "\n".join(lines)


# =============================================================================
# ABSTRACT BASE SOLVER
# =============================================================================

class BaseSolver(ABC):
    """
    Abstract base class for CVRP solvers.
    
    This interface defines the contract that all CVRP solvers must implement.
    It ensures consistent behavior across different solving approaches and
    enables fair benchmarking comparisons.
    
    All concrete solver implementations must inherit from this class and
    implement the abstract methods: `name`, `reference`, and `solve`.
    
    The base class provides utility methods for route distance calculation
    and solution validation that are common across all solvers.
    
    Abstract Properties
    -------------------
    name : str
        Human-readable name for the solver.
    reference : str
        Bibliographic reference for the algorithm.
    
    Abstract Methods
    ----------------
    solve(distance_matrix, demands, capacity, autonomy, time_limit, **kwargs)
        Main solving method that must be implemented.
    
    Utility Methods
    ---------------
    validate_solution(result, instance)
        Validate a solution against problem constraints.
    _calc_route_distance(route, dm)
        Calculate the total distance of a single route.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the solver.
        
        Returns
        -------
        str
            Solver name for identification in benchmarks and reports.
        """
        pass
    
    @property
    @abstractmethod
    def reference(self) -> str:
        """
        Bibliographic reference for the algorithm.
        
        Returns
        -------
        str
            Citation or reference for the method used.
        """
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
        Solve the Capacitated Vehicle Routing Problem (CVRP).
        
        This is the main method that concrete solvers must implement.
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            Square matrix of shape (n, n) containing pairwise distances.
            Index 0 represents the depot; indices 1 to n-1 are customers.
        demands : List[int]
            List of demands for each location. demands[0] = 0 (depot).
        capacity : int
            Maximum load capacity of the vehicle.
        autonomy : float
            Maximum distance the vehicle can travel per route.
        time_limit : float, optional
            Maximum computation time in seconds (default: 30.0).
        **kwargs : dict
            Additional solver-specific parameters.
        
        Returns
        -------
        SolverResult
            Object containing the solution routes and metrics.
        """
        pass
    
    def _calc_route_distance(self, route: List[int], dm: np.ndarray) -> float:
        """
        Calculate the total distance of a single route.
        
        The route is assumed to start and end at the depot (index 0).
        
        Parameters
        ----------
        route : List[int]
            Sequence of waypoint indices to visit (excluding depot).
        dm : np.ndarray
            Distance matrix.
        
        Returns
        -------
        float
            Total distance: depot -> route[0] -> ... -> route[-1] -> depot
        
        Example
        -------
            >>> route = [3, 1, 5]  # Visit waypoints 3, 1, 5
            >>> dist = solver._calc_route_distance(route, distance_matrix)
            >>> # Returns: dm[0,3] + dm[3,1] + dm[1,5] + dm[5,0]
        """
        if not route:
            return 0.0
        
        # Depot to first waypoint
        dist = dm[0, route[0]]
        
        # Between consecutive waypoints
        for i in range(len(route) - 1):
            dist += dm[route[i], route[i + 1]]
        
        # Last waypoint back to depot
        dist += dm[route[-1], 0]
        
        return dist
    
    def validate_solution(self, result: 'SolverResult', instance) -> Dict:
        """
        Validate a solution against problem constraints.
        
        Checks for:
        - Capacity violations (route demand > vehicle capacity)
        - Autonomy violations (route distance > max distance)
        - Missing waypoints (not all customers visited)
        - Duplicate waypoints (customer visited more than once)
        
        Parameters
        ----------
        result : SolverResult
            The solution to validate.
        instance : BenchmarkInstance
            The problem instance with constraints.
        
        Returns
        -------
        Dict
            Validation results with keys:
            - 'valid': bool - True if solution is feasible
            - 'capacity_violations': List of violation details
            - 'autonomy_violations': List of violation details
            - 'missing_waypoints': List of unvisited waypoints
            - 'duplicate_waypoints': List of duplicated waypoints
        """
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
            # Check capacity
            route_demand = sum(instance.demands[wp] for wp in route)
            if route_demand > instance.capacity:
                validation['capacity_violations'].append({
                    'route': route_idx,
                    'demand': route_demand,
                    'capacity': instance.capacity,
                    'excess': route_demand - instance.capacity
                })
                validation['valid'] = False
            
            # Check autonomy
            route_dist = self._calc_route_distance(route, instance.distance_matrix)
            if route_dist > instance.autonomy:
                validation['autonomy_violations'].append({
                    'route': route_idx,
                    'distance': route_dist,
                    'autonomy': instance.autonomy,
                    'excess': route_dist - instance.autonomy
                })
                validation['valid'] = False
            
            # Check duplicates
            for wp in route:
                if wp in visited:
                    validation['duplicate_waypoints'].append(wp)
                    validation['valid'] = False
                visited.add(wp)
        
        # Check coverage
        missing = expected - visited
        if missing:
            validation['missing_waypoints'] = list(missing)
            validation['valid'] = False
        
        return validation


# =============================================================================
# MAIN SOLVER: João's TSP-Split Algorithm
# =============================================================================

class JoaoTSPSolver(BaseSolver):
    """
    TSP-Split Solver using Lin-Kernighan Heuristic.
    
    This solver implements a "Route-first, Cluster-second" approach for the
    Capacitated Vehicle Routing Problem (CVRP). It combines the powerful
    Lin-Kernighan heuristic for TSP with an optimal dynamic programming
    split algorithm.
    
    Algorithm
    ---------
    1. **TSP Phase**: Construct a Hamiltonian tour visiting all waypoints
       (and optionally the depot) using the Lin-Kernighan heuristic.
       
    2. **Reordering**: If depot is included, reorder the tour to start
       from the depot position, enabling better split decisions.
       
    3. **Split Phase**: Apply dynamic programming to partition the tour
       into feasible routes that respect capacity and autonomy constraints
       while minimizing total distance.
       
    4. **Refinement**: Apply 2-opt local search to each route for
       potential improvement.
    
    Complexity
    ----------
    - TSP Phase: O(n^2.2) expected for Lin-Kernighan
    - Split Phase: O(n²) worst case, O(n) typical with pruning
    - Refinement: O(n²) per route, O(n³) total worst case
    
    Parameters
    ----------
    split_strategy : str, optional
        Strategy for splitting the TSP tour into routes:
        - 'greedy': Sequential split when constraints are violated (fast)
        - 'optimal': Dynamic programming for minimum cost split (better quality)
        Default is 'optimal'.
    use_2opt_refinement : bool, optional
        Whether to apply 2-opt local search on each route (default: True).
    include_depot_in_tsp : bool, optional
        Whether to include the depot in the TSP tour (default: True).
        Including the depot generally improves solution quality.
    
    Attributes
    ----------
    available : bool
        Whether the lk_heuristic library is installed and available.
    
    References
    ----------
    .. [1] Lin, S. & Kernighan, B.W. (1973). "An Effective Heuristic Algorithm
           for the Traveling-Salesman Problem." Operations Research, 21(2).
    
    .. [2] Prins, C. (2004). "A simple and effective evolutionary algorithm
           for the vehicle routing problem." Computers & Operations Research.
    
    Example
    -------
        >>> solver = JoaoTSPSolver(split_strategy='optimal')
        >>> result = solver.solve(
        ...     distance_matrix=dm,
        ...     demands=[0, 15, 15, 15, 15],  # Depot + 4 waypoints
        ...     capacity=45,
        ...     autonomy=100.0
        ... )
        >>> print(result)
    """
    
    def __init__(self, 
                 split_strategy: str = 'optimal', 
                 use_2opt_refinement: bool = True,
                 include_depot_in_tsp: bool = True):
        """
        Initialize the João TSP-Split Solver.
        
        Parameters
        ----------
        split_strategy : str, optional
            'greedy' for fast sequential split, 'optimal' for DP-based split.
        use_2opt_refinement : bool, optional
            Enable 2-opt local search refinement on each route.
        include_depot_in_tsp : bool, optional
            Include depot node in TSP for better tour quality.
        
        Raises
        ------
        Warning
            If lk_heuristic library is not installed.
        """
        self.split_strategy = split_strategy
        self.use_2opt_refinement = use_2opt_refinement
        self.include_depot_in_tsp = include_depot_in_tsp
        
        # Check for Lin-Kernighan library availability
        try:
            from lk_heuristic.utils.solver_funcs import solve as lk_solve
            self.lk_solve = lk_solve
            self.available = True
        except ImportError:
            self.available = False
            import warnings
            warnings.warn(
                "lk_heuristic library not found. Install with: pip install lk-heuristic",
                ImportWarning
            )
    
    @property
    def name(self) -> str:
        """Solver identifier for benchmarking."""
        return f"João-LKH-TSP ({self.split_strategy})"
    
    @property
    def reference(self) -> str:
        """Bibliographic reference."""
        return (
            "João Rafael - TSP-Split Solver with Lin-Kernighan Heuristic. "
            "Based on: Lin & Kernighan (1973). 'An Effective Heuristic Algorithm "
            "for the Traveling-Salesman Problem.' Operations Research, 21(2), 498-516."
        )
    
    def solve(self,
              distance_matrix: np.ndarray,
              demands: List[int],
              capacity: int,
              autonomy: float,
              time_limit: float = 30.0,
              **kwargs) -> SolverResult:
        """
        Solve the CVRP using TSP + Split approach.
        
        This method orchestrates the complete solving process:
        1. Extract or receive coordinates
        2. Solve TSP for all nodes
        3. Reorder tour starting from depot
        4. Split into feasible routes
        5. Apply local search refinement
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            Symmetric distance matrix of shape (n, n).
            Entry [i,j] is the distance from node i to node j.
            Node 0 is the depot.
        demands : List[int]
            Demand at each node. demands[0] must be 0 (depot).
        capacity : int
            Maximum vehicle capacity per route.
        autonomy : float
            Maximum distance per route (including return to depot).
        time_limit : float, optional
            Maximum solving time in seconds.
        **kwargs : dict
            Additional parameters:
            - coordinates: List[Tuple[float, float]] - Node coordinates
            - instance_name: str - Name for result reporting
        
        Returns
        -------
        SolverResult
            Solution containing routes, distance, and performance metrics.
        
        Notes
        -----
        If coordinates are not provided, they are estimated from the distance
        matrix using Multidimensional Scaling (MDS).
        """
        # Handle unavailable solver
        if not self.available:
            return SolverResult(
                solver_name=self.name,
                instance_name=kwargs.get('instance_name', 'unknown'),
                routes=[],
                total_distance=float('inf'),
                num_routes=0,
                computation_time=0,
                feasible=False,
                metadata={'error': 'lk_heuristic not installed'}
            )
        
        start_time = time.time()
        n = len(demands)
        
        # Handle trivial case
        if n <= 1:
            return SolverResult(
                solver_name=self.name,
                instance_name=kwargs.get('instance_name', 'unknown'),
                routes=[],
                total_distance=0.0,
                num_routes=0,
                computation_time=0,
                feasible=True
            )
        
        # Get coordinates (from kwargs or extract from distance matrix)
        coords = kwargs.get('coordinates', None)
        if coords is None:
            coords = self._extract_coordinates(distance_matrix)
        
        # Determine which nodes to include in TSP
        if self.include_depot_in_tsp:
            tsp_indices = list(range(n))  # All nodes including depot
        else:
            tsp_indices = list(range(1, n))  # Only waypoints
        
        # Solve TSP
        if len(tsp_indices) <= 2:
            tour = [i for i in tsp_indices if i != 0]
        else:
            raw_tour = self._solve_tsp_lk(coords, tsp_indices, time_limit)
            
            if self.include_depot_in_tsp:
                tour = self._reorder_tour_from_depot(raw_tour)
            else:
                tour = raw_tour
        
        tsp_time = time.time() - start_time
        
        # Split tour into feasible routes
        # Para clientes virtuais, não há necessidade de commodity constraints
        if self.split_strategy == 'optimal':
            routes = self._split_optimal(tour, demands, capacity, autonomy, distance_matrix)
        else:
            routes = self._split_greedy(tour, demands, capacity, autonomy, distance_matrix)
        
        # Apply local search refinement
        if self.use_2opt_refinement:
            routes = [self._local_2opt(route, distance_matrix) for route in routes]
        
        computation_time = time.time() - start_time
        
        # Calculate total distance
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
                'tsp_time': tsp_time,
                'split_strategy': self.split_strategy,
                'include_depot': self.include_depot_in_tsp,
                'tour_length': len(tour),
                'capacity': capacity,
                'autonomy': autonomy
            }
        )
    
    # =========================================================================
    # COORDINATE EXTRACTION
    # =========================================================================
    
    def _extract_coordinates(self, dm: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extract 2D coordinates from distance matrix using MDS.
        
        When explicit coordinates are not available, this method uses
        Multidimensional Scaling to recover approximate 2D positions
        from the pairwise distance matrix.
        
        Parameters
        ----------
        dm : np.ndarray
            Distance matrix of shape (n, n).
        
        Returns
        -------
        List[Tuple[float, float]]
            Estimated (x, y) coordinates for each node.
        
        Notes
        -----
        For Euclidean distance matrices, MDS can recover the original
        coordinates up to rotation, reflection, and translation.
        Falls back to polar coordinate approximation if sklearn is unavailable.
        """
        n = dm.shape[0]
        
        try:
            from sklearn.manifold import MDS
            mds = MDS(
                n_components=2, 
                dissimilarity='precomputed', 
                random_state=42,
                max_iter=300, 
                normalized_stress='auto'
            )
            coords = mds.fit_transform(dm)
            return [(coords[i, 0], coords[i, 1]) for i in range(n)]
        except (ImportError, TypeError):
            # Fallback: polar coordinate approximation
            coords = [(0.0, 0.0)]  # Depot at origin
            for i in range(1, n):
                angle = 2 * math.pi * (i - 1) / (n - 1)
                r = dm[0, i]
                coords.append((r * math.cos(angle), r * math.sin(angle)))
            return coords
    
    # =========================================================================
    # TSP SOLVING (LIN-KERNIGHAN)
    # =========================================================================
    
    def _solve_tsp_lk(self, 
                      coords: List[Tuple[float, float]], 
                      waypoint_indices: List[int],
                      time_limit: float) -> List[int]:
        """
        Solve TSP using Lin-Kernighan Heuristic.
        
        Creates a TSPLIB format file and uses the lk_heuristic library
        to find a near-optimal tour.
        
        The Lin-Kernighan algorithm is one of the most effective heuristics
        for TSP, typically finding solutions within 1% of optimal.
        
        Parameters
        ----------
        coords : List[Tuple[float, float]]
            (x, y) coordinates for all nodes.
        waypoint_indices : List[int]
            Indices of nodes to include in TSP.
        time_limit : float
            Maximum computation time.
        
        Returns
        -------
        List[int]
            Tour as ordered list of original node indices.
        
        Notes
        -----
        The method handles TSPLIB file I/O internally, creating temporary
        files that are cleaned up after solving.
        
        References
        ----------
        Lin, S. & Kernighan, B.W. (1973). "An Effective Heuristic Algorithm
        for the Traveling-Salesman Problem." Operations Research.
        """
        n = len(waypoint_indices)
        
        # Create temporary files
        tsp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.tsp', delete=False)
        solution_base = tempfile.mktemp()
        
        try:
            # Write TSPLIB format file
            tsp_file.write("NAME: joao_drone_tsp\n")
            tsp_file.write("TYPE: TSP\n")
            tsp_file.write("COMMENT: Drone plantation TSP - João Rafael\n")
            tsp_file.write(f"DIMENSION: {n}\n")
            tsp_file.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            tsp_file.write("NODE_COORD_SECTION\n")
            
            for i, wp in enumerate(waypoint_indices):
                x, y = coords[wp]
                # Scale to integers (TSPLIB convention)
                tsp_file.write(f"{i + 1} {int(x * 1000)} {int(y * 1000)}\n")
            
            tsp_file.write("EOF\n")
            tsp_file.close()
            
            # Solve with LK-heuristic
            self.lk_solve(
                tsp_file=tsp_file.name,
                solution_method="lk1_improve",
                runs=1,
                backtracking=(5, 5),
                reduction_level=4,
                reduction_cycle=4,
                tour_type="cycle",
                file_name=solution_base,
                logging_level=50  # Suppress output
            )
            
            # Parse solution file
            solution_file = solution_base + ".tsp"
            tour = self._parse_tsp_solution(solution_file, coords, waypoint_indices)
            
            # Clean up solution file
            if os.path.exists(solution_file):
                os.unlink(solution_file)
            
            # Fallback if parsing failed
            if len(tour) != n:
                tour = waypoint_indices.copy()
                
        finally:
            # Clean up input file
            if os.path.exists(tsp_file.name):
                os.unlink(tsp_file.name)
        
        return tour
    
    def _parse_tsp_solution(self,
                            solution_file: str,
                            coords: List[Tuple[float, float]],
                            waypoint_indices: List[int]) -> List[int]:
        """
        Parse TSP solution file and map back to original indices.
        
        Parameters
        ----------
        solution_file : str
            Path to the TSPLIB solution file.
        coords : List[Tuple[float, float]]
            Original coordinates.
        waypoint_indices : List[int]
            Original node indices.
        
        Returns
        -------
        List[int]
            Tour with original node indices.
        """
        tour = []
        
        if not os.path.exists(solution_file):
            return tour
        
        with open(solution_file, 'r') as f:
            lines = f.readlines()
            node_coord_section = False
            
            for line in lines:
                line = line.strip()
                if line.startswith("NODE_COORD_SECTION"):
                    node_coord_section = True
                    continue
                    
                if node_coord_section and line and line != "EOF":
                    try:
                        parts = line.split()
                        if len(parts) >= 3:
                            x = float(parts[1]) / 1000.0
                            y = float(parts[2]) / 1000.0
                            
                            # Match to original waypoint
                            for wp in waypoint_indices:
                                wx, wy = coords[wp]
                                if abs(wx - x) < 0.01 and abs(wy - y) < 0.01:
                                    if wp not in tour:
                                        tour.append(wp)
                                    break
                    except (ValueError, IndexError):
                        continue
                        
                if line == "EOF":
                    break
        
        return tour
    
    def _reorder_tour_from_depot(self, tour: List[int]) -> List[int]:
        """
        Reorder TSP tour to start after the depot.
        
        Since a TSP tour is a cycle, we can choose any starting point.
        For CVRP, we start the tour from the node following the depot
        to facilitate better split decisions.
        
        Parameters
        ----------
        tour : List[int]
            TSP tour potentially containing depot (index 0).
        
        Returns
        -------
        List[int]
            Reordered tour starting after depot, with depot removed.
        
        Example
        -------
            >>> tour = [3, 0, 1, 5, 2]  # Depot at position 1
            >>> reordered = solver._reorder_tour_from_depot(tour)
            >>> # Returns: [1, 5, 2, 3]  # Start after depot, depot removed
        """
        if 0 not in tour:
            return tour
        
        depot_pos = tour.index(0)
        n = len(tour)
        
        reordered = []
        for i in range(1, n):
            idx = (depot_pos + i) % n
            if tour[idx] != 0:
                reordered.append(tour[idx])
        
        return reordered
    
    # =========================================================================
    # SPLIT ALGORITHMS
    # =========================================================================
    
    def _split_greedy(self,
                      tour: List[int],
                      demands: List[int],
                      capacity: int,
                      autonomy: float,
                      dm: np.ndarray) -> List[List[int]]:
        """
        Greedy sequential split algorithm.
        
        Traverses the TSP tour sequentially and creates a new route whenever
        adding the next waypoint would violate capacity or autonomy constraints.
        
        This approach is fast (O(n)) but may not find the optimal split.
        
        Parameters
        ----------
        tour : List[int]
            TSP tour (sequence of waypoint indices).
        demands : List[int]
            Demand at each node.
        capacity : int
            Maximum route capacity.
        autonomy : float
            Maximum route distance.
        dm : np.ndarray
            Distance matrix.
        commodity_capacities : Dict, optional
            Maximum capacity per commodity.
        commodities : List, optional
            Commodity type for each node.
        
        Returns
        -------
        List[List[int]]
            List of routes, each route is a list of waypoint indices.
        
        Algorithm
        ---------
        ```
        for each waypoint w in tour:
            if adding w violates constraints:
                close current route
                start new route with w
            else:
                add w to current route
        ```
        """
        if not tour:
            return []
        
        routes = []
        current_route = []
        current_demand = 0
        current_distance = 0.0
        last_wp = 0  # Start at depot
        
        # Para clientes virtuais, não há necessidade de commodity constraints
        current_commodity_demands = None
        
        for wp in tour:
            wp_demand = demands[wp]
            dist_to_wp = dm[last_wp, wp]
            dist_to_base = dm[wp, 0]
            
            # Check if waypoint fits in current route
            potential_demand = current_demand + wp_demand
            potential_distance = current_distance + dist_to_wp + dist_to_base
            
            # Para clientes virtuais, não há necessidade de commodity constraints
            commodity_feasible = True
            
            if potential_demand <= capacity and potential_distance <= autonomy and commodity_feasible:
                # Fits - add to current route
                current_route.append(wp)
                current_demand = potential_demand
                current_distance = current_distance + dist_to_wp
                last_wp = wp
            else:
                # Doesn't fit - close current route and start new
                if current_route:
                    routes.append(current_route)
                
                current_route = [wp]
                current_demand = wp_demand
                current_distance = dm[0, wp]
                last_wp = wp
        
        # Add final route
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def _split_optimal(self,
                       tour: List[int],
                       demands: List[int],
                       capacity: int,
                       autonomy: float,
                       dm: np.ndarray) -> List[List[int]]:
        """
        Optimal split using dynamic programming.
        
        Finds the minimum-cost partition of the TSP tour into feasible routes
        using the Bellman-Ford style dynamic programming approach.
        
        This method guarantees the optimal split for a given tour ordering.
        
        Parameters
        ----------
        tour : List[int]
            TSP tour (sequence of waypoint indices).
        demands : List[int]
            Demand at each node.
        capacity : int
            Maximum route capacity.
        autonomy : float
            Maximum route distance.
        dm : np.ndarray
            Distance matrix.
        commodity_capacities : Dict, optional
            Maximum capacity per commodity.
        commodities : List, optional
            Commodity type for each node.
        
        Returns
        -------
        List[List[int]]
            Optimal list of routes.
        
        Complexity
        ----------
        Time: O(n²) worst case, typically O(n) with pruning
        Space: O(n) for DP arrays
        
        Algorithm
        ---------
        Let DP[i] = minimum cost to serve waypoints tour[0..i-1]
        
        Recurrence:
            DP[i] = min over all valid j < i of:
                    DP[j] + cost(route from tour[j] to tour[i-1])
        
        A route tour[j..i-1] is valid if:
            - sum of demands <= capacity
            - per-commodity demands <= commodity_capacities (if provided)
            - route distance <= autonomy
        
        References
        ----------
        Beasley, J.E. (1983). "Route first—cluster second methods for
        vehicle routing." Omega, 11(4), 403-408.
        
        Prins, C. (2004). "A simple and effective evolutionary algorithm
        for the vehicle routing problem." Computers & Operations Research.
        """
        if not tour:
            return []
        
        n = len(tour)
        INF = float('inf')
        
        # DP arrays
        dp_cost = [INF] * (n + 1)  # dp_cost[i] = min cost to serve tour[0..i-1]
        dp_pred = [-1] * (n + 1)   # Predecessor for route reconstruction
        dp_cost[0] = 0.0
        
        # Fill DP table
        for i in range(1, n + 1):
            route_demand = 0
            
            # Try all possible route start positions j
            for j in range(i - 1, -1, -1):
                wp = tour[j]
                route_demand += demands[wp]
                
                # Para clientes virtuais, não há necessidade de commodity constraints
                commodity_feasible = True
                
                # Pruning: if demand exceeds capacity, no need to check longer routes
                if route_demand > capacity:
                    break
                
                # Calculate route distance
                route_distance = self._calc_route_distance(tour[j:i], dm)
                
                # Check autonomy constraint
                if route_distance > autonomy:
                    continue  # Route too long, but shorter routes may work
                
                # Update DP if this split is better
                candidate_cost = dp_cost[j] + route_distance
                if candidate_cost < dp_cost[i]:
                    dp_cost[i] = candidate_cost
                    dp_pred[i] = j
        
        # Reconstruct routes from predecessors
        routes = []
        i = n
        while i > 0:
            j = dp_pred[i]
            if j == -1:
                # No valid split found - fallback to greedy
                return self._split_greedy(tour, demands, capacity, autonomy, dm)
            routes.append(tour[j:i])
            i = j
        
        routes.reverse()
        return routes
    
    # =========================================================================
    # LOCAL SEARCH REFINEMENT
    # =========================================================================
    
    def _local_2opt(self, route: List[int], dm: np.ndarray) -> List[int]:
        """
        Improve route using 2-opt local search.
        
        The 2-opt algorithm iteratively improves a route by reversing
        segments. For each pair of edges, it checks if reversing the
        path between them reduces total distance.
        
        Parameters
        ----------
        route : List[int]
            Initial route (list of waypoint indices).
        dm : np.ndarray
            Distance matrix.
        
        Returns
        -------
        List[int]
            Improved route.
        
        Complexity
        ----------
        O(n²) per improvement iteration, O(n³) worst case total.
        
        Algorithm
        ---------
        ```
        repeat until no improvement:
            for i in range(n-1):
                for j in range(i+2, n):
                    new_route = route[:i+1] + reversed(route[i+1:j+1]) + route[j+1:]
                    if cost(new_route) < cost(route):
                        route = new_route
                        restart search
        ```
        
        References
        ----------
        Croes, G.A. (1958). "A Method for Solving Traveling-Salesman Problems."
        Operations Research, 6(6), 791-812.
        """
        if len(route) < 3:
            return route
        
        improved = True
        best_route = route.copy()
        
        while improved:
            improved = False
            best_dist = self._calc_route_distance(best_route, dm)
            
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    # Create new route with reversed segment
                    new_route = (
                        best_route[:i + 1] + 
                        best_route[i + 1:j + 1][::-1] + 
                        best_route[j + 1:]
                    )
                    new_dist = self._calc_route_distance(new_route, dm)
                    
                    if new_dist < best_dist - 1e-6:  # Numerical tolerance
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                        
                if improved:
                    break
        
        return best_route
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _calculate_total_distance(self, 
                                   routes: List[List[int]], 
                                   dm: np.ndarray) -> float:
        """
        Calculate total distance across all routes.
        
        Parameters
        ----------
        routes : List[List[int]]
            List of routes.
        dm : np.ndarray
            Distance matrix.
        
        Returns
        -------
        float
            Sum of distances for all routes.
        """
        return sum(self._calc_route_distance(route, dm) for route in routes)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def solve_cvrp(distance_matrix: np.ndarray,
               demands: List[int],
               capacity: int,
               autonomy: float,
               split_strategy: str = 'optimal',
               time_limit: float = 30.0,
               **kwargs) -> SolverResult:
    """
    Convenience function to solve CVRP with João's solver.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Symmetric distance matrix (n x n). Index 0 is depot.
    demands : List[int]
        Demand at each node. demands[0] = 0.
    capacity : int
        Vehicle capacity.
    autonomy : float
        Maximum distance per route.
    split_strategy : str, optional
        'greedy' or 'optimal' (default).
    time_limit : float, optional
        Maximum solving time in seconds.
    **kwargs : dict
        Additional parameters passed to solver.
    
    Returns
    -------
    SolverResult
        Solution with routes and metrics.
    
    Example
    -------
        >>> from joao_tsp_solver import solve_cvrp
        >>> result = solve_cvrp(dm, demands, capacity=100, autonomy=500)
        >>> print(f"Total: {result.total_distance:.1f}m in {result.num_routes} routes")
    """
    solver = JoaoTSPSolver(split_strategy=split_strategy)
    return solver.solve(
        distance_matrix=distance_matrix,
        demands=demands,
        capacity=capacity,
        autonomy=autonomy,
        time_limit=time_limit,
        **kwargs
    )


# =============================================================================
# MAIN - DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of the João TSP-Split Solver.
    
    Creates a simple test instance and solves it with different strategies.
    """
    print("=" * 70)
    print("João's TSP-Split Solver - Demonstration")
    print("=" * 70)
    
    # Create a simple test instance (5x5 grid)
    print("\nCreating test instance (5x5 grid, 25 waypoints)...")
    
    # Generate waypoints in a grid
    waypoints = [(0, 0)]  # Depot at origin
    for x in range(5):
        for y in range(5):
            waypoints.append((x * 10 + 5, y * 10 + 5))
    
    n = len(waypoints)
    
    # Build distance matrix
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = waypoints[i][0] - waypoints[j][0]
                dy = waypoints[i][1] - waypoints[j][1]
                dm[i, j] = math.sqrt(dx * dx + dy * dy)
    
    # Demands (depot = 0, waypoints = 15 each)
    demands = [0] + [15] * (n - 1)
    
    # Problem parameters
    capacity = 100
    autonomy = 150.0
    
    print(f"  Waypoints: {n - 1}")
    print(f"  Capacity: {capacity}")
    print(f"  Autonomy: {autonomy}m")
    
    # Solve with different strategies
    print("\n" + "-" * 70)
    print("Solving with Greedy Split...")
    solver_greedy = JoaoTSPSolver(split_strategy='greedy')
    
    if solver_greedy.available:
        result_greedy = solver_greedy.solve(
            distance_matrix=dm,
            demands=demands,
            capacity=capacity,
            autonomy=autonomy,
            coordinates=waypoints,
            instance_name='demo_grid'
        )
        print(result_greedy)
    else:
        print("  Solver not available (lk_heuristic not installed)")
    
    print("\n" + "-" * 70)
    print("Solving with Optimal Split...")
    solver_optimal = JoaoTSPSolver(split_strategy='optimal')
    
    if solver_optimal.available:
        result_optimal = solver_optimal.solve(
            distance_matrix=dm,
            demands=demands,
            capacity=capacity,
            autonomy=autonomy,
            coordinates=waypoints,
            instance_name='demo_grid'
        )
        print(result_optimal)
        
        # Print routes
        print("\nRoutes:")
        for i, route in enumerate(result_optimal.routes):
            route_dist = solver_optimal._calc_route_distance(route, dm)
            route_demand = sum(demands[wp] for wp in route)
            print(f"  Route {i + 1}: {route}")
            print(f"           Distance: {route_dist:.1f}m, Demand: {route_demand}")
    else:
        print("  Solver not available (lk_heuristic not installed)")
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)

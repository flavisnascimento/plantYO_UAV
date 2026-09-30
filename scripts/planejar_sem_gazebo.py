#!/usr/bin/env python3
"""
Planeja as missoes sem Gazebo e confere os compartimentos.

Para cada configuracao, talhao e solver: numero de rotas, estouros de
compartimento, cobertura dos clientes e distancia planejada. Serve para checar
a correcao e ter o numero de rotas antes de voar.

Uso:  python3 planejar_sem_gazebo.py [tempo_por_solver_s] [solvers...]
Ex.:  python3 planejar_sem_gazebo.py 30 HGS NN LKH DAHA
"""
import contextlib
import io
import sys

sys.path.insert(0, '.')
from grid_generator import GridGenerator, GridConfig, CommodityCapacity
from compartimentos import dados_compartimentos, violacoes
from NN import NearestNeighborSolver
from DAHA import AHASolverBenchmark
from lkh_tsp_solver import LKHTSPSolver
from hgs_solver import HGSSolver, DroneConfig

TL = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
SOLVERS = [s.upper() for s in sys.argv[2:]] or ['HGS', 'NN', 'LKH', 'DAHA']
CAMPOS = {'G75': 75, 'G100': 100, 'G150': 150}
CONFIGS = {'Eq-Multi': (100, 100, 100), 'Rt-Multi': (120, 120, 60)}
AUTONOMIA = 1500.0


def distancia(routes, dm):
    return sum(dm[0, r[0]] + sum(dm[r[i], r[i + 1]] for i in range(len(r) - 1)) + dm[r[-1], 0]
               for r in routes if r)


print(f"{'config':9} {'talhao':6} {'solver':6} {'rotas':>5} {'estouros':>8} {'cobertura':>9} {'dist_planejada':>14}")
for nome, (e, a, t) in CONFIGS.items():
    for campo, L in CAMPOS.items():
        g = GridGenerator(GridConfig(grid_size_x=L, grid_size_y=L, waypoint_spacing=2.0,
                                     line_spacing=3.0, margin=2.5, base_x=L / 2, base_y=0.0,
                                     seeds_per_waypoint=15,
                                     commodity_capacity=CommodityCapacity(erva=e, arbusto=a, arvore=t)))
        g.generate()
        dm, dem = g.get_distance_matrix(), g.get_demands()
        comm, caps = dados_compartimentos(g)
        cap = sum(caps.values())
        kw = dict(commodities=comm, commodity_capacities=caps)
        coords = [g.get_base_position()] + [(c.center_x, c.waypoints[0].y) for c in g.virtual_clients]
        for s in SOLVERS:
            if s == 'NN':
                routes = NearestNeighborSolver().solve(dm, dem, cap, AUTONOMIA, TL, **kw).routes
            elif s == 'LKH':
                routes = LKHTSPSolver().solve(dm, dem, cap, AUTONOMIA, TL, coordinates=coords, **kw).routes
            elif s in ('DAHA', 'D-AHA'):
                routes = AHASolverBenchmark().solve(dm, dem, cap, AUTONOMIA, TL, seed=0, **kw).routes
            elif s == 'HGS':
                with contextlib.redirect_stdout(io.StringIO()):
                    routes = HGSSolver(DroneConfig(dispenser_capacity=cap, autonomy_meters=AUTONOMIA)
                                       ).solve_with_autonomy(dm, dem, time_limit=TL, verbose=True, **kw).routes
            else:
                continue
            cobertura = sorted(c for r in routes for c in r) == list(range(1, len(dem)))
            print(f"{nome:9} {campo:6} {s:6} {len(routes):5d} {len(violacoes(routes, dem, comm, caps)):8d} "
                  f"{str(cobertura):>9} {distancia(routes, dm):14.0f}", flush=True)

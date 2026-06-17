#!/usr/bin/env python3
"""
Dispensor Planter - 1comp vs 3comp + escolha de solver (NN/HGS/DAHA).
Mede tempo de missao e distancia no Gazebo.
"""
import rospy
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgs_planter_node import HGSPlanterNode
from grid_generator import GridGenerator, GridConfig, CommodityCapacity
from hgs_solver import HGSSolver, DroneConfig
from solver_benchmark import AHASolverBenchmark, NearestNeighborSolver


# guarda o init_node real
_real_init_node = rospy.init_node

def _noop_init_node(*args, **kwargs):
    pass  # ja inicializado, nao faz nada


# --- escolhe o solver ---
_orig_generate = HGSPlanterNode.generate_and_optimize

def _generate_multi_solver(self):
    solver_name = getattr(self, "_solver_name", "HGS").upper()
    if solver_name == "HGS":
        return _orig_generate(self)

    config = GridConfig(
        grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y,
        waypoint_spacing=self.waypoint_spacing, line_spacing=self.line_spacing,
        margin=2.5, base_x=self.base_x, base_y=self.base_y,
        seeds_per_waypoint=self.seeds_per_waypoint,
        commodity_capacity=CommodityCapacity(
            erva=self.capacity_erva, arbusto=self.capacity_arbusto,
            arvore=self.capacity_arvore))
    generator = GridGenerator(config)
    generator.generate()
    effective_capacity = generator.get_effective_capacity()
    dm = generator.get_individual_distance_matrix()
    demands = generator.get_individual_demands()

    if solver_name == "NN":
        solver = NearestNeighborSolver()
    elif solver_name in ("DAHA", "D-AHA", "AHA"):
        solver = AHASolverBenchmark()
    else:
        return _orig_generate(self)

    rospy.loginfo(f"[SOLVER] Resolvendo com {solver_name}...")
    result = solver.solve(
        distance_matrix=dm, demands=demands,
        capacity=effective_capacity, autonomy=self.drone_autonomy,
        time_limit=self.solver_time_limit, instance_name="gazebo")
    self._validate_solution(result, demands)
    rospy.loginfo(f"[SOLUCAO] Distancia: {result.total_distance:.2f}m, Rotas: {result.num_routes}")
    return generator, result

HGSPlanterNode.generate_and_optimize = _generate_multi_solver


class DispensorPlanter:
    def __init__(self):
        _real_init_node('dispensor_planter', anonymous=True)
        self.modo = rospy.get_param("~modo", "3comp")
        self._cap = rospy.get_param("~cap_por_guilda", 100)
        self._solver_name = rospy.get_param("~solver", "HGS")
        rospy.loginfo("="*60)
        rospy.loginfo(f"[DISPENSOR] MODO: {self.modo} | SOLVER: {self._solver_name}")
        rospy.loginfo("="*60)

        # bloqueia init_node duplicado durante o super
        rospy.init_node = _noop_init_node

        if self.modo == "3comp":
            rospy.set_param("~capacity_erva", self._cap)
            rospy.set_param("~capacity_arbusto", self._cap)
            rospy.set_param("~capacity_arvore", self._cap)
            t0 = time.time()
            self._planter = self._make_planter()
            dur = time.time() - t0
            rospy.loginfo("="*60)
            rospy.loginfo(f"[RESULTADO-3comp] SOLVER={self._solver_name} TEMPO_MISSAO={dur:.1f}s")
            if getattr(self._planter, "solution", None):
                rospy.loginfo(f"[RESULTADO-3comp] DISTANCIA={self._planter.solution.total_distance:.1f}m")
            rospy.loginfo("="*60)
        else:
            self._run_1comp()

    def _make_planter(self):
        p = HGSPlanterNode.__new__(HGSPlanterNode)
        p._solver_name = self._solver_name
        p.__init__()
        return p

    def _run_1comp(self):
        campanhas = [("erva", self._cap,0,0), ("arvore",0,0,self._cap), ("arbusto",0,self._cap,0)]
        tempo_total = 0.0; dist_total = 0.0
        for nome, ce, cab, car in campanhas:
            rospy.set_param("~capacity_erva", ce)
            rospy.set_param("~capacity_arbusto", cab)
            rospy.set_param("~capacity_arvore", car)
            rospy.loginfo("="*60)
            rospy.loginfo(f"[CAMPANHA] {nome.upper()} | SOLVER {self._solver_name}")
            rospy.loginfo("="*60)
            t0 = time.time()
            p = self._make_planter()
            dur = time.time() - t0
            tempo_total += dur
            if getattr(p, "solution", None):
                dist_total += p.solution.total_distance
            rospy.loginfo(f"[CAMPANHA {nome}] tempo: {dur:.1f}s")
        rospy.loginfo("="*60)
        rospy.loginfo(f"[RESULTADO-1comp] SOLVER={self._solver_name} TEMPO_MISSAO={tempo_total:.1f}s")
        rospy.loginfo(f"[RESULTADO-1comp] DISTANCIA={dist_total:.1f}m")
        rospy.loginfo("="*60)


if __name__ == "__main__":
    try:
        DispensorPlanter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

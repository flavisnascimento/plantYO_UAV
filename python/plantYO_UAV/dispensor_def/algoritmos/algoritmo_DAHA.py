#!/usr/bin/env python3
"""Algoritmo: D-AHA - modos 1comp / 3comp"""

import rospy, json, time, os, sys, argparse
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/plantyo_ws/src/mrs_computer_vision_examples/python/plantYO_UAV/scripts'))
sys.path.insert(0, _BASE)

from grid_generator import GridGenerator, GridConfig, CommodityCapacity
from solver_benchmark import AHASolverBenchmark
from comp_mode import solve_with_mode
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String

SPACING = 2.5; MARGIN = 2.5; SEEDS_WP = 15; Q_TOTAL = 300; AUTONOMY = 2025.0

class SolverDAHA:
    def __init__(self, modo, time_limit=30.0, pop=30, iters=100):
        rospy.init_node('algoritmo_daha', anonymous=True)
        self.modo = modo; self.time_limit = time_limit
        self.solver = AHASolverBenchmark(population_size=pop, max_iterations=iters)
        self.pub_routes = rospy.Publisher('/planter/routes', MarkerArray, queue_size=10, latch=True)
        self.pub_waypoints = rospy.Publisher('/planter/waypoints', MarkerArray, queue_size=10, latch=True)
        self.pub_log = rospy.Publisher('/planter/log', String, queue_size=10)
        self.frame_id = "uav1/gps_origin"; self.color = (1.0, 1.0, 0.0); self.results = []

    def log(self, m):
        rospy.loginfo(f"[DAHA-{self.modo}] {m}")
        self.pub_log.publish(String(m))

    def _route(self, dm, demands, tl):
        r = self.solver.solve(distance_matrix=dm, demands=demands,
                              capacity=Q_TOTAL, autonomy=AUTONOMY,
                              time_limit=tl, instance_name='x')
        return r.routes

    def _publish_waypoints(self, waypoints):
        markers = MarkerArray()
        colors = {'erva': (0.4, 0.9, 0.2), 'arbusto': (1.0, 0.7, 0.0), 'arvore': (0.0, 0.5, 0.1)}
        for wp in waypoints:
            m = Marker(); m.header.frame_id = self.frame_id; m.header.stamp = rospy.Time.now()
            m.ns = "waypoints"; m.id = wp.id; m.type = Marker.SPHERE
            m.pose.position = Point(wp.x, wp.y, 0.2); m.pose.orientation.w = 1.0
            c = colors.get(wp.plant_type.value, (0.5, 0.5, 0.5))
            m.color.r, m.color.g, m.color.b = c; m.color.a = 0.6
            m.scale.x = m.scale.y = m.scale.z = 0.4; m.lifetime = rospy.Duration(0)
            markers.markers.append(m)
        self.pub_waypoints.publish(markers)

    def _publish_routes(self, routes, gen):
        markers = MarkerArray()
        base_x, base_y = gen.get_base_position()
        for idx, route in enumerate(routes):
            m = Marker(); m.header.frame_id = self.frame_id; m.header.stamp = rospy.Time.now()
            m.ns = "routes"; m.id = idx; m.type = Marker.LINE_STRIP; m.pose.orientation.w = 1.0
            m.points.append(Point(base_x, base_y, 0.5))
            for wp_id in route:
                vc = gen.virtual_clients[wp_id - 1]
                m.points.append(Point(vc.center_x, vc.y, 0.5))
            m.points.append(Point(base_x, base_y, 0.5))
            m.color.r, m.color.g, m.color.b = self.color; m.color.a = 1.0
            m.scale.x = 0.3; m.lifetime = rospy.Duration(0)
            markers.markers.append(m)
        self.pub_routes.publish(markers)

    def run_test(self, grid_size):
        self.log(f"{'='*70}")
        self.log(f"TESTE D-AHA [{self.modo}] - Grid {grid_size}m")
        self.log(f"{'='*70}")
        config = GridConfig(
            grid_size_x=grid_size, grid_size_y=grid_size,
            waypoint_spacing=SPACING, line_spacing=SPACING, margin=MARGIN,
            base_x=grid_size/2, base_y=0.0, seeds_per_waypoint=SEEDS_WP,
            commodity_capacity=CommodityCapacity(erva=100, arbusto=100, arvore=100))
        gen = GridGenerator(config); gen.generate()
        self.log(f"Grid: {len(gen.waypoints)} waypoints, {len(gen.virtual_clients)} clientes virtuais")
        self._publish_waypoints(gen.waypoints)
        dm = gen.get_distance_matrix(); demands = gen.get_demands()

        tl = self.time_limit / 3.0 if self.modo == '1comp' else self.time_limit
        route_fn = lambda d, dem: self._route(d, dem, tl)
        start = time.time()
        routes, total, campanhas = solve_with_mode(self.modo, route_fn, gen, dm, demands)
        elapsed = time.time() - start

        self.log(f"Resultado: {len(routes)} rotas, {total:.2f}m em {elapsed:.4f}s ({campanhas} campanha(s))")
        self._publish_routes(routes, gen)
        self.results.append({
            'grid_size': grid_size, 'modo': self.modo, 'campanhas': campanhas,
            'num_waypoints': len(gen.waypoints),
            'num_virtual_clients': len(gen.virtual_clients),
            'num_routes': len(routes), 'total_distance': total,
            'solve_time_s': elapsed, 'solver': 'D-AHA'})

    def save_results(self):
        out = os.path.normpath(os.path.join(_BASE, '..', 'results'))
        os.makedirs(out, exist_ok=True)
        fn = os.path.join(out, f'DAHA_{self.modo}_{int(time.time())}.json')
        with open(fn, 'w') as f:
            json.dump({'solver': 'D-AHA', 'modo': self.modo, 'results': self.results}, f, indent=2)
        self.log(f"Salvo: {fn}")
        self.log("="*70); self.log(f"RESUMO D-AHA [{self.modo}]")
        for r in self.results:
            self.log(f"Grid {r['grid_size']:6.1f}m: {r['num_virtual_clients']:4d} vc, {r['num_routes']:3d} rotas, {r['total_distance']:8.1f}m, {r['solve_time_s']:.4f}s")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modo', choices=['1comp', '3comp'], default='3comp')
    p.add_argument('--grids', nargs='+', type=float, default=[25.0, 50.0, 75.0])
    p.add_argument('--time', type=float, default=30.0)
    args = p.parse_args()
    try:
        s = SolverDAHA(modo=args.modo, time_limit=args.time)
        for size in args.grids:
            s.run_test(size); rospy.sleep(1)
        s.save_results()
    except rospy.ROSInterruptException:
        pass

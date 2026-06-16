#!/usr/bin/env python3
"""Algoritmo: Nearest Neighbor (NN) - modos 1comp / 3comp"""

import rospy, json, time, os, sys, argparse
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/plantyo_ws/src/mrs_computer_vision_examples/python/plantYO_UAV/scripts'))
sys.path.insert(0, _BASE)

from grid_generator import GridGenerator, GridConfig, CommodityCapacity
from comp_mode import solve_with_mode
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String

SPACING = 2.5; MARGIN = 2.5; SEEDS_WP = 15; Q_TOTAL = 300; AUTONOMY = 2025.0

def nearest_neighbor(dm, demands, capacity, autonomy):
    n = len(demands)
    unvisited = set(range(1, n))
    routes = []
    while unvisited:
        route = []; rd = 0; cur_dist = 0.0; current = 0
        while True:
            best = None; bd = float('inf')
            for nx in unvisited:
                if rd + demands[nx] > capacity:
                    continue
                d_to = dm[current, nx]
                if cur_dist + d_to + dm[nx, 0] > autonomy:
                    continue
                if d_to < bd:
                    bd = d_to; best = nx
            if best is None:
                break
            route.append(best); rd += demands[best]; cur_dist += bd; current = best
            unvisited.remove(best)
        if route:
            routes.append(route)
        else:
            routes.append([unvisited.pop()])
    return routes

class SolverNN:
    def __init__(self, modo):
        rospy.init_node('algoritmo_nn', anonymous=True)
        self.modo = modo
        self.pub_routes = rospy.Publisher('/planter/routes', MarkerArray, queue_size=10, latch=True)
        self.pub_waypoints = rospy.Publisher('/planter/waypoints', MarkerArray, queue_size=10, latch=True)
        self.pub_log = rospy.Publisher('/planter/log', String, queue_size=10)
        self.frame_id = "uav1/gps_origin"
        self.color = (1.0, 0.0, 0.0)
        self.results = []

    def log(self, m):
        rospy.loginfo(f"[NN-{self.modo}] {m}")
        self.pub_log.publish(String(m))

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
        self.log(f"TESTE NN [{self.modo}] - Grid {grid_size}m")
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

        route_fn = lambda d, dem: nearest_neighbor(d, dem, Q_TOTAL, AUTONOMY)
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
            'solve_time_s': elapsed, 'solver': 'NN'})

    def save_results(self):
        out = os.path.join(_BASE, '..', 'results')
        out = os.path.normpath(out)
        os.makedirs(out, exist_ok=True)
        fn = os.path.join(out, f'NN_{self.modo}_{int(time.time())}.json')
        with open(fn, 'w') as f:
            json.dump({'solver': 'NN', 'modo': self.modo, 'results': self.results}, f, indent=2)
        self.log(f"Salvo: {fn}")
        self.log("="*70); self.log(f"RESUMO NN [{self.modo}]")
        for r in self.results:
            self.log(f"Grid {r['grid_size']:6.1f}m: {r['num_virtual_clients']:4d} vc, {r['num_routes']:3d} rotas, {r['total_distance']:8.1f}m, {r['solve_time_s']:.4f}s")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--modo', choices=['1comp', '3comp'], default='3comp')
    p.add_argument('--grids', nargs='+', type=float, default=[25.0, 50.0, 75.0])
    args = p.parse_args()
    try:
        s = SolverNN(modo=args.modo)
        for size in args.grids:
            s.run_test(size); rospy.sleep(1)
        s.save_results()
    except rospy.ROSInterruptException:
        pass

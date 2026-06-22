#!/usr/bin/env python3
"""
HGS Planter - Sistema de Plantio com Drone usando C-SDVRP
VERSÃO FINAL: Commoditized Split Delivery VRP

Baseado em:
- Vidal (2022): Hybrid Genetic Search for CVRP
- Petris (2024): Transformação C-SDVRP → CVRP

Características:
- Multi-commodity: 3 compartimentos (Erva, Arbusto, Árvore) com 100 cada
- Split Delivery: Pode voltar à base no meio do caminho
- Autonomia: Respeita limite de distância por viagem
- Padrão de plantio: E-A-Á-A-E (repete a cada 5 waypoints)
"""

import rospy
import math
import os
import time
from threading import Lock
from gazebo_msgs.srv import SpawnModel
from mrs_msgs.srv import Vec4, PathSrv, PathSrvRequest
from mrs_msgs.msg import Reference
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String, Float64
from visualization_msgs.msg import Marker, MarkerArray
from typing import List, Dict, Tuple

from grid_generator import GridGenerator, GridConfig, CommodityCapacity
from hgs_solver import HGSSolver, DroneConfig, CVRPSolution
from mission_logger import MissionLogger, BatteryModel

CRUISING_ALT = 2.0   # Altitude de cruzeiro (voo entre pontos)
PLANTING_ALT = 1.5   # Altitude de plantio (desce para dispensar)
TOLERANCE = 2.0      # Tolerância horizontal para chegada

# Cores por tipo de planta (RGB)
PLANT_COLORS = {
    "erva": (0.4, 0.9, 0.2),      # Verde claro
    "arbusto": (1.0, 0.7, 0.0),   # Amarelo/Laranja
    "arvore": (0.0, 0.5, 0.1),    # Verde escuro
}


class RouteVisualizer:
    """Visualizador de rotas e waypoints no RViz"""
    
    def __init__(self):
        # latch=True mantém a última mensagem para novos subscribers
        self.pub = rospy.Publisher("/planter/visualization", MarkerArray, queue_size=10, latch=True)
        self.markers = MarkerArray()
        self.id_counter = 0
        self.planted_ids = set()  # IDs dos waypoints já plantados
        # Frame do MRS - origem do mundo
        self.frame_id = "uav1/world_origin"
    
    def clear(self):
        """Limpa todos os markers"""
        msg = Marker()
        msg.header.frame_id = self.frame_id
        msg.action = Marker.DELETEALL
        self.markers.markers = [msg]
        self.pub.publish(self.markers)
        self.markers = MarkerArray()
        self.id_counter = 0
        self.planted_ids = set()
    
    def _create_marker(self, x, y, z, color, ns="planter"):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time(0)  # Usa último TF disponível
        marker.ns = ns
        marker.id = self.id_counter
        self.id_counter += 1
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        return marker
    
    def draw_waypoints(self, waypoints: List, show_labels: bool = False):
        """Desenha todos os waypoints como esferas coloridas"""
        for wp in waypoints:
            pt = wp.to_dict() if hasattr(wp, 'to_dict') else wp
            color = PLANT_COLORS.get(pt['plant_type'], (0.5, 0.5, 0.5))
            
            # Esfera para o waypoint
            marker = self._create_marker(pt['x'], pt['y'], 0.3, color, "waypoints")
            marker.type = Marker.SPHERE
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            self.markers.markers.append(marker)
    
    def draw_route(self, route_points: List[Tuple[float, float]], color=(0.0, 0.5, 1.0), route_id=0):
        """Desenha uma rota como linha"""
        if len(route_points) < 2:
            return
        
        marker = self._create_marker(0, 0, 0, color, f"route_{route_id}")
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.08  # Espessura da linha
        
        for x, y in route_points:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.5  # Altura da linha
            marker.points.append(pt)
        
        self.markers.markers.append(marker)
    
    def draw_base(self, x, y):
        """Desenha a base como um cilindro azul"""
        marker = self._create_marker(x, y, 0.5, (0.0, 0.3, 1.0), "base")
        marker.type = Marker.CYLINDER
        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 1.0
        self.markers.markers.append(marker)
        
        # Texto "BASE"
        text = self._create_marker(x, y, 2.0, (1.0, 1.0, 1.0), "base_label")
        text.type = Marker.TEXT_VIEW_FACING
        text.text = "BASE"
        text.scale.z = 1.0
        self.markers.markers.append(text)
    
    def draw_field_boundary(self, size_x, size_y, margin=0):
        """Desenha os limites externos do talhão"""
        marker = self._create_marker(0, 0, 0, (1.0, 0.5, 0.0), "boundary")
        marker.type = Marker.LINE_STRIP
        marker.scale.x = 0.15
        
        # Cantos do talhão (borda externa)
        corners = [
            (0, 0),
            (size_x, 0),
            (size_x, size_y),
            (0, size_y),
            (0, 0)  # Fecha o quadrado
        ]
        
        for x, y in corners:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.1
            marker.points.append(pt)
        
        self.markers.markers.append(marker)
    
    def mark_planted(self, x, y, plant_type):
        """Marca um waypoint como plantado"""
        color = PLANT_COLORS.get(plant_type, (0.5, 0.5, 0.5))
        # Cor mais brilhante
        bright_color = (min(1.0, color[0] + 0.3), min(1.0, color[1] + 0.3), min(1.0, color[2] + 0.3))
        
        marker = self._create_marker(x, y, 0.5, bright_color, "planted")
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.8
        self.markers.markers.append(marker)
        self.publish()
    
    def publish(self):
        """Publica todos os markers"""
        self.pub.publish(self.markers)


class MissionStats:
    """Estatísticas da missão em tempo real"""
    def __init__(self):
        self.start_time = None
        self.total_plants = 0
        self.planted = 0
        self.total_distance = 0.0
        self.current_route = 0
        self.total_routes = 0
        self.current_speed = 0.0
        self.last_pos = None
        self.last_time = None
        self.battery_percent = 100.0  # Bateria
        
        # Publishers para métricas
        self.pub_planted = rospy.Publisher("/planter/planted", Float64, queue_size=1)
        self.pub_progress = rospy.Publisher("/planter/progress", Float64, queue_size=1)
        self.pub_speed = rospy.Publisher("/planter/speed", Float64, queue_size=1)
        self.pub_distance = rospy.Publisher("/planter/distance", Float64, queue_size=1)
        self.pub_eta = rospy.Publisher("/planter/eta_minutes", Float64, queue_size=1)
        self.pub_marker = rospy.Publisher("/planter/status_marker", Marker, queue_size=1)
        self.pub_status = rospy.Publisher("/planter/status", MarkerArray, queue_size=1, latch=True)
        self.pub_drone = rospy.Publisher("/planter/drone_marker", Marker, queue_size=1)
    
    def start(self, total_plants: int, total_routes: int):
        self.start_time = time.time()
        self.total_plants = total_plants
        self.total_routes = total_routes
        self.planted = 0
        self.total_distance = 0.0
    
    def update_position(self, x: float, y: float):
        """Atualiza posição e calcula velocidade/distância"""
        now = time.time()
        if self.last_pos is not None and self.last_time is not None:
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            dist = math.hypot(dx, dy)
            dt = now - self.last_time
            if dt > 0:
                self.current_speed = dist / dt
                self.total_distance += dist
        self.last_pos = (x, y)
        self.last_time = now
    
    def plant(self):
        """Registra uma planta"""
        self.planted += 1
        self.publish()
    
    def set_route(self, route: int):
        self.current_route = route
    
    def set_battery(self, percent: float):
        """Atualiza percentual de bateria"""
        self.battery_percent = percent
    
    def elapsed(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def eta_minutes(self) -> float:
        """Tempo estimado restante em minutos"""
        if self.planted > 0:
            avg_time = self.elapsed() / self.planted
            remaining = self.total_plants - self.planted
            return (remaining * avg_time) / 60.0
        return 0.0
    
    def progress_percent(self) -> float:
        if self.total_plants > 0:
            return (self.planted / self.total_plants) * 100.0
        return 0.0
    
    def publish(self):
        """Publica todas as métricas"""
        self.pub_planted.publish(Float64(self.planted))
        self.pub_progress.publish(Float64(self.progress_percent()))
        self.pub_speed.publish(Float64(self.current_speed))
        self.pub_distance.publish(Float64(self.total_distance))
        self.pub_eta.publish(Float64(self.eta_minutes()))
        
        # Marker de texto para RViz/Gazebo
        self.publish_status_marker()
    
    def publish_status_marker(self):
        """Publica marker de texto com status"""
        marker = Marker()
        marker.header.frame_id = "uav1/fcu"
        marker.header.stamp = rospy.Time(0)  # Usa último TF disponível
        marker.ns = "planter_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Posição acima do drone
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 2.0
        marker.pose.orientation.w = 1.0
        
        # Tamanho do texto
        marker.scale.z = 0.5
        
        # Cor verde
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Texto com estatísticas
        elapsed = self.elapsed()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        marker.text = (
            f"Plantado: {self.planted}/{self.total_plants} ({self.progress_percent():.1f}%)\n"
            f"Rota: {self.current_route}/{self.total_routes}\n"
            f"Vel: {self.current_speed:.1f} m/s | Dist: {self.total_distance:.0f}m\n"
            f"Tempo: {mins:02d}:{secs:02d} | ETA: {self.eta_minutes():.1f} min"
        )
        
        marker.lifetime = rospy.Duration(1.0)
        self.pub_marker.publish(marker)
        
        # Também publica painel de status fixo no canto do talhão
        self.publish_status_panel()
    
    def publish_status_panel(self):
        """Publica painel de status fixo no canto do talhão"""
        markers = MarkerArray()
        
        # Painel de fundo (semi-transparente)
        bg = Marker()
        bg.header.frame_id = "uav1/world_origin"
        bg.header.stamp = rospy.Time(0)
        bg.ns = "status_panel"
        bg.id = 0
        bg.type = Marker.CUBE
        bg.action = Marker.ADD
        bg.pose.position.x = 5
        bg.pose.position.y = 5
        bg.pose.position.z = 8
        bg.pose.orientation.w = 1.0
        bg.scale.x = 0.1
        bg.scale.y = 15
        bg.scale.z = 8
        bg.color.r = 0.1
        bg.color.g = 0.1
        bg.color.b = 0.3
        bg.color.a = 0.7
        markers.markers.append(bg)
        
        # Texto de status
        txt = Marker()
        txt.header.frame_id = "uav1/world_origin"
        txt.header.stamp = rospy.Time(0)
        txt.ns = "status_panel"
        txt.id = 1
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = 5
        txt.pose.position.y = 5
        txt.pose.position.z = 10
        txt.pose.orientation.w = 1.0
        txt.scale.z = 1.5
        txt.color.r = 1.0
        txt.color.g = 1.0
        txt.color.b = 1.0
        txt.color.a = 1.0
        
        elapsed = self.elapsed()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        txt.text = (
            f"=== MISSAO ===\n"
            f"Plantado: {self.planted}/{self.total_plants}\n"
            f"Progresso: {self.progress_percent():.1f}%\n"
            f"Rota: {self.current_route}/{self.total_routes}\n"
            f"Dist: {self.total_distance:.0f}m\n"
            f"Bateria: {self.battery_percent:.0f}%\n"
            f"Tempo: {mins:02d}:{secs:02d}"
        )
        markers.markers.append(txt)
        
        self.pub_status.publish(markers)
    
    def print_summary(self):
        """Imprime resumo final"""
        elapsed = self.elapsed()
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"MISSÃO COMPLETA!")
        rospy.loginfo(f"{'='*60}")
        rospy.loginfo(f"Plantas: {self.planted}/{self.total_plants}")
        rospy.loginfo(f"Tempo total: {mins:02d}:{secs:02d}")
        rospy.loginfo(f"Distância total: {self.total_distance:.1f}m")
        rospy.loginfo(f"Velocidade média: {self.total_distance/elapsed:.2f} m/s" if elapsed > 0 else "")
        rospy.loginfo(f"Tempo por planta: {elapsed/self.planted:.2f}s" if self.planted > 0 else "")
        rospy.loginfo(f"{'='*60}")


class HGSPlanterNode:
    """
    Nó ROS para plantio com drone usando C-SDVRP.
    
    Fluxo:
    1. Gera grid de waypoints com padrão E-A-Á-A-E
    2. Transforma em C-SDVRP (clientes virtuais por commodity)
    3. Resolve com HGS + split por autonomia
    4. Executa rotas, voltando à base para recarregar
    """
    
    def __init__(self):
        rospy.init_node("hgs_planter", anonymous=True)
        
        # Parâmetros do talhão - AJUSTADOS para melhor visualização
        self.grid_size_x = rospy.get_param("~grid_size_x", 25.0)
        self.grid_size_y = rospy.get_param("~grid_size_y", 25.0)
        self.waypoint_spacing = rospy.get_param("~waypoint_spacing", 5.0)  
        self.line_spacing = rospy.get_param("~line_spacing", 5.0)      
        self.base_x = rospy.get_param("~base_x", 50.0)  # Base no centro-inferior (50,0)
        self.base_y = rospy.get_param("~base_y", 0.0)
        
        # Parâmetros do drone
        self.capacity_erva = rospy.get_param("~capacity_erva", 100)
        self.capacity_arbusto = rospy.get_param("~capacity_arbusto", 100)
        self.capacity_arvore = rospy.get_param("~capacity_arvore", 100)
        self.drone_autonomy = rospy.get_param("~drone_autonomy", 2250.0)
        self.reserve_percent = rospy.get_param("~reserve_percent", 0.10)
        self.seeds_per_waypoint = rospy.get_param("~seeds_per_waypoint", 15)
        
        # Parâmetros de bateria (valores AUMENTADOS para simulação visível)
        # Com estes valores, cada 100m de voo consome ~5% de bateria
        self.battery_capacity_mah = rospy.get_param("~battery_capacity_mah", 1000)  # Bateria menor
        self.battery_consumption_hover = rospy.get_param("~battery_consumption_hover", 500.0)  # 500A simulado
        self.battery_consumption_flight = rospy.get_param("~battery_consumption_flight", 1000.0)  # 1000A simulado
        self.battery_consumption_plant = rospy.get_param("~battery_consumption_plant", 600.0)  # 600A simulado
        self.battery_reserve = rospy.get_param("~battery_reserve", 20.0)
        
        # Parâmetros do solver
        self.solver_time_limit = rospy.get_param("~solver_time_limit", 10.0)
        
        # Inicializa modelo de bateria
        self.battery = BatteryModel(
            capacity_mah=self.battery_capacity_mah,
            consumption_hover_amps=self.battery_consumption_hover,
            consumption_flight_amps=self.battery_consumption_flight,
            consumption_plant_amps=self.battery_consumption_plant,
            reserve_percent=self.battery_reserve
        )
        
        # Inicializa logger para missão (CSV/JSON export)
        log_output_dir = rospy.get_param("~log_output_dir", os.path.expanduser("~/plantyo_logs"))
        mission_name = rospy.get_param("~mission_name", f"mission_{int(time.time())}")
        self.logger = MissionLogger(output_dir=log_output_dir, mission_name=mission_name)
        rospy.loginfo(f"[LOGGER] Logs serão salvos em: {log_output_dir}/{mission_name}")
        
        # Publisher para alertas
        self.muda_pub = rospy.Publisher("/muda_alert", String, queue_size=1)
        self.battery_pub = rospy.Publisher("/planter/battery", Float64, queue_size=1)
        
        # ========================================
        # 1) AGUARDA DRONE ESTAR PRONTO
        # ========================================
        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("AGUARDANDO DRONE INICIALIZAR...")
        rospy.loginfo("=" * 60)
        
        # Aguarda serviços ROS do drone
        rospy.loginfo("[DRONE] Aguardando serviço trajectory_generation/path...")
        rospy.wait_for_service('/uav1/trajectory_generation/path')
        self.path_srv = rospy.ServiceProxy('/uav1/trajectory_generation/path', PathSrv)
        
        rospy.loginfo("[DRONE] Aguardando serviço control_manager/goto...")
        rospy.wait_for_service('/uav1/control_manager/goto')
        self.goto_srv = rospy.ServiceProxy('/uav1/control_manager/goto', Vec4)
        
        rospy.loginfo("[DRONE] Aguardando serviço gazebo/spawn_sdf_model...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        rospy.loginfo("[DRONE] ✓ Todos os serviços disponíveis!")
        
        # Carrega modelo de planta
        self.load_plant_model()
        
        # ========================================
        # 2) DESENHA LIMITES DO TALHÃO
        # ========================================
        self.spawn_field_boundaries()
        
        # ========================================
        # 3) RODA O SOLVER HGS
        # ========================================
        rospy.loginfo("\n[SOLVER] Iniciando otimização C-SDVRP...")
        self.generator, self.solution = self.generate_and_optimize()
        
        # Estado da missão
        self.current_route_idx = 0
        self.current_client_in_route = 0
        self.current_wp_in_client = 0
        self.current_client_waypoints = []
        self.planting = False
        self.lock = Lock()
        self.base_position = self.generator.get_base_position()
        
        # ========================================
        # 3.5) VISUALIZAÇÃO NO RVIZ
        # ========================================
        rospy.loginfo("\n[VIZ] Criando visualização no RViz...")
        self.visualizer = RouteVisualizer()
        rospy.sleep(0.5)  # Espera publisher conectar
        self.visualize_mission()
        rospy.loginfo("[VIZ] ✓ Visualização publicada em /planter/visualization")
        
        # ========================================
        # 4) AGUARDA DRONE DECOLAR
        # ========================================
        rospy.loginfo("\n[DRONE] Aguardando drone decolar...")
        rospy.loginfo("[DRONE] Verificando altitude...")
        
        # Aguarda até o drone estar no ar (altitude > 1m)
        self.current_altitude = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        
        # Subscriber de odometria (método da classe)
        self.odom_sub = rospy.Subscriber(
            '/uav1/estimation_manager/odom_main',
            Odometry,
            self._odom_callback,
            queue_size=1
        )
        
        # Espera até 60 segundos para o drone decolar
        timeout = 60
        rate = rospy.Rate(2)
        for i in range(timeout * 2):
            if self.current_altitude > 1.0:
                rospy.loginfo(f"[DRONE] ✓ Drone no ar! Altitude: {self.current_altitude:.1f}m")
                rospy.loginfo(f"[DRONE] Posição: ({self.current_x:.1f}, {self.current_y:.1f})")
                break
            if i % 10 == 0:
                rospy.loginfo(f"[DRONE] Altitude atual: {self.current_altitude:.1f}m (aguardando > 1.0m)")
            rate.sleep()
        else:
            rospy.logwarn("[DRONE] Timeout! Drone pode não ter decolado. Continuando mesmo assim...")
        
        rospy.sleep(2.0)  # Tempo extra para estabilizar
        
        # ========================================
        # 5) EXECUTA MISSÃO (LOOP SÍNCRONO)
        # ========================================
        rospy.loginfo("\n[MISSÃO] Iniciando plantio...")
        rospy.sleep(1.0)
        self.execute_mission()
    
    def _validate_solution(self, solution, demands):
        """
        Valida a solução do HGS para garantir que:
        1. Não há waypoints duplicados (mesmo waypoint em múltiplas rotas)
        2. Todos os waypoints são visitados exatamente uma vez
        3. A solução é válida para execução
        
        Args:
            solution: CVRPSolution retornada pelo solver
            demands: Lista de demandas (índice 0 = depósito)
        """
        rospy.loginfo("\n[VALIDAÇÃO] Verificando solução HGS...")
        
        visited = set()
        duplicates = []
        all_waypoints = []
        
        # Verifica duplicatas entre rotas
        for route_idx, route in enumerate(solution.routes):
            for wp_id in route:
                all_waypoints.append((wp_id, route_idx))
                if wp_id in visited:
                    # Encontrar em qual rota estava antes
                    prev_route = next(r for w, r in all_waypoints[:-1] if w == wp_id)
                    duplicates.append((wp_id, prev_route, route_idx))
                    rospy.logerr(f"[VALIDAÇÃO] DUPLICATA: Waypoint {wp_id} nas rotas {prev_route+1} e {route_idx+1}")
                visited.add(wp_id)
        
        # Verifica cobertura (todos os waypoints exceto depósito)
        expected = set(range(1, len(demands)))  # Depósito é 0
        missing = expected - visited
        extra = visited - expected
        
        if missing:
            rospy.logwarn(f"[VALIDAÇÃO] Waypoints NÃO visitados: {sorted(missing)}")
        
        if extra:
            rospy.logwarn(f"[VALIDAÇÃO] Waypoints inválidos (fora do range): {sorted(extra)}")
        
        # Resumo
        if duplicates:
            rospy.logerr(f"[VALIDAÇÃO] ❌ ERRO: {len(duplicates)} waypoints duplicados!")
            rospy.logerr(f"[VALIDAÇÃO] Duplicatas: {duplicates}")
        elif missing:
            rospy.logwarn(f"[VALIDAÇÃO] ⚠ AVISO: {len(missing)} waypoints não serão visitados")
        else:
            rospy.loginfo(f"[VALIDAÇÃO] ✓ Solução válida: {len(visited)} waypoints únicos em {len(solution.routes)} rotas")
        
        return len(duplicates) == 0 and len(missing) == 0
    
    def generate_and_optimize(self) -> Tuple[GridGenerator, CVRPSolution]:
        """Gera grid e otimiza com C-SDVRP + HGS"""
        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("HGS-CVRP PLANTER - Vidal (2022)")
        rospy.loginfo("=" * 60)
        
        # Configuração do grid (com margem de 2.5m das bordas)
        config = GridConfig(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            waypoint_spacing=self.waypoint_spacing,
            line_spacing=self.line_spacing,
            margin=2.5,  # Margem das bordas do talhão
            base_x=self.base_x,
            base_y=self.base_y,
            seeds_per_waypoint=self.seeds_per_waypoint,
            commodity_capacity=CommodityCapacity(
                erva=self.capacity_erva,
                arbusto=self.capacity_arbusto,
                arvore=self.capacity_arvore
            )
        )
        
        generator = GridGenerator(config)
        generator.generate()
        
        rospy.loginfo(f"\n[GRID] Talhão: {config.grid_size_x}m x {config.grid_size_y}m (margem: {config.margin}m)")
        rospy.loginfo(f"[GRID] Área efetiva: {config.grid_size_x - 2*config.margin}m x {config.grid_size_y - 2*config.margin}m")
        rospy.loginfo(f"[GRID] Base: ({config.base_x}, {config.base_y})")
        rospy.loginfo(f"[GRID] Total waypoints: {len(generator.waypoints)}")
        
        # Debug: mostrar range de coordenadas
        if generator.waypoints:
            xs = [wp.x for wp in generator.waypoints]
            ys = [wp.y for wp in generator.waypoints]
            rospy.loginfo(f"[GRID] Range X: {min(xs):.1f} a {max(xs):.1f}")
            rospy.loginfo(f"[GRID] Range Y: {min(ys):.1f} a {max(ys):.1f}")
        
        # Capacidade efetiva baseada na proporção E-A-Á-A-E
        effective_capacity = generator.get_effective_capacity()
        wps_per_trip = effective_capacity // self.seeds_per_waypoint
        
        rospy.loginfo(f"\n[DRONE] Capacidade por tipo: {self.capacity_erva}E + {self.capacity_arbusto}A + {self.capacity_arvore}Á")
        rospy.loginfo(f"[DRONE] Capacidade efetiva: {effective_capacity} sementes ({wps_per_trip} waypoints/viagem)")
        rospy.loginfo(f"[DRONE] Autonomia: {self.drone_autonomy}m")
        
        # Configuração do drone para o solver
        drone_config = DroneConfig(
            dispenser_capacity=effective_capacity,
            autonomy_meters=self.drone_autonomy,
            reserve_percent=self.reserve_percent
        )
        
        # Resolver CVRP com waypoints individuais
        rospy.loginfo(f"\n[SOLVER] Resolvendo CVRP com HGS (waypoints individuais)...")
        rospy.loginfo(f"[SOLVER] Time limit: {self.solver_time_limit}s")
        
        solver = HGSSolver(drone_config)
        distance_matrix = generator.get_individual_distance_matrix()
        demands = generator.get_individual_demands()
        
        solution = solver.solve_with_autonomy(
            distance_matrix=distance_matrix,
            demands=demands,
            time_limit=self.solver_time_limit,
            verbose=True
        )
        
        # VALIDAÇÃO: Verificar duplicatas na solução
        self._validate_solution(solution, demands)
        
        # Resumo da solução
        
        rospy.loginfo(f"\n[SOLUÇÃO] Distância total: {solution.total_distance:.2f}m")
        rospy.loginfo(f"[SOLUÇÃO] Número de rotas: {solution.num_routes}")
        
        # Mostrar resumo das rotas
        for i, route in enumerate(solution.routes[:5]):
            seeds = sum(demands[wp_id] for wp_id in route)
            rospy.loginfo(f"[ROTA {i+1}] {len(route)} waypoints, {seeds} sementes")
        if len(solution.routes) > 5:
            rospy.loginfo(f"... e mais {len(solution.routes) - 5} rotas")
        
        return generator, solution
    
    def load_plant_model(self):
        """Carrega modelos de plantas (agora usa arquivos SDF)"""
        self.load_plant_models()
        rospy.loginfo("[MODEL] ✓ Modelos 3D prontos para uso")
    
    def visualize_mission(self):
        """Desenha waypoints e rotas no RViz"""
        # Limpa visualização anterior
        self.visualizer.clear()
        rospy.sleep(0.1)
        
        # DEBUG: Mostra coordenadas do campo e base
        rospy.loginfo(f"[VIZ] Campo: (0,0) a ({self.grid_size_x}, {self.grid_size_y})")
        rospy.loginfo(f"[VIZ] Base: {self.base_position}")
        
        # Mostra alguns waypoints para debug
        if self.generator.waypoints:
            wp_first = self.generator.waypoints[0]
            wp_last = self.generator.waypoints[-1]
            rospy.loginfo(f"[VIZ] Primeiro waypoint: ({wp_first.x}, {wp_first.y})")
            rospy.loginfo(f"[VIZ] Último waypoint: ({wp_last.x}, {wp_last.y})")
        
        # Desenha limites do talhão (borda externa)
        self.visualizer.draw_field_boundary(self.grid_size_x, self.grid_size_y)
        
        # Desenha base
        self.visualizer.draw_base(self.base_position[0], self.base_position[1])
        
        # Desenha todos os waypoints
        self.visualizer.draw_waypoints(self.generator.get_all_waypoints())
        
        # Desenha rotas (primeiras 10 para não poluir)
        colors = [
            (1.0, 0.0, 0.0),  # Vermelho
            (0.0, 1.0, 0.0),  # Verde
            (0.0, 0.0, 1.0),  # Azul
            (1.0, 1.0, 0.0),  # Amarelo
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Ciano
            (1.0, 0.5, 0.0),  # Laranja
            (0.5, 0.0, 1.0),  # Roxo
            (0.0, 0.5, 0.5),  # Teal
            (0.5, 0.5, 0.0),  # Oliva
        ]
        
        for route_idx, route in enumerate(self.solution.routes[:10]):
            route_points = [self.base_position]  # Começa na base
            
            # Agora route contém IDs de waypoints individuais
            for wp_matrix_id in route:
                wp_id = wp_matrix_id - 1
                wp = self.generator.waypoints[wp_id]
                route_points.append((wp.x, wp.y))
            
            route_points.append(self.base_position)  # Volta à base
            
            color = colors[route_idx % len(colors)]
            self.visualizer.draw_route(route_points, color, route_idx)
        
        self.visualizer.publish()
        rospy.loginfo(f"[VIZ] Desenhadas {min(10, len(self.solution.routes))} rotas no RViz")

    def execute_mission(self):
        """Executa toda a missão de forma síncrona com estatísticas"""
        # Calcula total de plantas (agora cada waypoint = 1 planta)
        total_plants = sum(len(route) for route in self.solution.routes)
        
        # TRACKING: Set de waypoints já plantados para evitar duplicatas
        self.planted_waypoint_ids = set()
        self.planted_coords = {}  # Dict: (x, y) -> wp_id que plantou lá
        
        # Inicializa estatísticas
        self.stats = MissionStats()
        self.stats.start(total_plants, len(self.solution.routes))
        
        # Reset da bateria no início da missão
        self.battery.reset()
        
        rospy.loginfo(f"\n[MISSÃO] Total de plantas: {total_plants}")
        rospy.loginfo(f"[MISSÃO] Total de rotas: {len(self.solution.routes)}")
        rospy.loginfo(f"[BATERIA] Capacidade: {self.battery.capacity_mah}mAh | Reserva: {self.battery.reserve_percent}%")
        
        for route_idx, route in enumerate(self.solution.routes):
            self.stats.set_route(route_idx + 1)
            route_start = time.time()
            
            # Inicia log da rota (agora route é lista de waypoint IDs)
            self.logger.log_route_start(route_idx + 1, len(route), len(route))
            
            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"ROTA {route_idx + 1}/{len(self.solution.routes)} - {len(route)} waypoints")
            rospy.loginfo(f"[BATERIA] {self.battery.get_percent():.1f}% | {self.battery.voltage():.1f}V")
            rospy.loginfo(f"{'='*60}")
            
            # Agora route contém IDs de waypoints individuais (1-indexed da matriz)
            for wp_idx, wp_matrix_id in enumerate(route):
                wp_start = time.time()
                
                # Converte ID da matriz (1-indexed) para índice do waypoint (0-indexed)
                waypoint_id = wp_matrix_id - 1
                
                # VERIFICAÇÃO DE DUPLICATA: Pula se já foi plantado
                if wp_matrix_id in self.planted_waypoint_ids:
                    rospy.logwarn(f"[SKIP] Waypoint {wp_matrix_id} já foi plantado! Pulando...")
                    continue
                
                # Validação de índice
                if waypoint_id < 0 or waypoint_id >= len(self.generator.waypoints):
                    rospy.logerr(f"[ERRO] ID inválido: matrix_id={wp_matrix_id}, wp_id={waypoint_id}, max={len(self.generator.waypoints)-1}")
                    continue
                
                # Obtém o waypoint
                wp = self.generator.waypoints[waypoint_id]
                pt = wp.to_dict()
                
                # Verifica se já passou por essa coordenada física (arredondada para 0.5m)
                coord_key = (round(pt["x"] * 2) / 2, round(pt["y"] * 2) / 2)
                if coord_key in self.planted_coords:
                    rospy.logwarn(f"[COORD DUPLICADA] Posição ({pt['x']:.1f}, {pt['y']:.1f}) já visitada! ID atual: {wp_matrix_id}, ID anterior: {self.planted_coords[coord_key]}")
                
                # Calcula distância para consumo de bateria
                dist_to_wp = math.sqrt((pt["x"] - self.current_x)**2 + (pt["y"] - self.current_y)**2)
                
                # Move para o waypoint em altitude de cruzeiro
                arrived = self.goto_and_wait(pt["x"], pt["y"], CRUISING_ALT)
                
                # Consome bateria pelo voo (usando distância, 2 m/s)
                self.battery.consume_flight(dist_to_wp, speed_ms=2.0)
                self.battery_pub.publish(self.battery.get_percent())
                self.stats.set_battery(self.battery.get_percent())
                
                # Atualiza estatísticas de posição
                self.stats.update_position(self.current_x, self.current_y)
                
                # SÓ PLANTA SE CHEGOU AO PONTO!
                if arrived:
                    # Desce para plantar (simula acionamento do dispenser)
                    self.descend_and_plant(pt["x"], pt["y"], pt["plant_type"], pt["id"])
                    self.muda_pub.publish(pt["plant_type"])
                    
                    # MARCA COMO PLANTADO para evitar revisita
                    self.planted_waypoint_ids.add(wp_matrix_id)
                    self.planted_coords[coord_key] = wp_matrix_id  # Registra coordenada física
                    
                    # Consome bateria pelo plantio (~3s hover + dispenser)
                    self.battery.consume_plant(3.0)
                    self.battery_pub.publish(self.battery.get_percent())
                    self.stats.set_battery(self.battery.get_percent())
                    
                    # Marca no visualizador RViz
                    self.visualizer.mark_planted(pt["x"], pt["y"], pt["plant_type"])
                    
                    # Registra planta e publica stats
                    self.stats.plant()
                    
                    wp_time = time.time() - wp_start
                    progress = self.stats.progress_percent()
                    
                    # Log do waypoint para CSV
                    self.logger.log_waypoint(
                        waypoint_id=pt["id"],
                        x=pt["x"],
                        y=pt["y"],
                        plant_type=pt["plant_type"],
                        route_id=route_idx + 1,
                        client_id=waypoint_id,
                        success=True,
                        duration_s=wp_time,
                        battery_percent=self.battery.get_percent()
                    )
                    
                    rospy.loginfo(
                        f"  WP {wp_idx+1}/{len(route)} | {pt['plant_type']} "
                        f"({pt['x']:.0f},{pt['y']:.0f}) | {wp_time:.1f}s | "
                        f"{progress:.1f}% | BAT: {self.battery.get_percent():.0f}%"
                    )
                else:
                    # Não chegou - pula este waypoint
                    wp_time = time.time() - wp_start
                    self.logger.log_waypoint(
                        waypoint_id=pt["id"],
                        x=pt["x"],
                        y=pt["y"],
                        plant_type=pt["plant_type"],
                        route_id=route_idx + 1,
                        client_id=waypoint_id,
                        success=False,
                        duration_s=wp_time,
                        battery_percent=self.battery.get_percent(),
                        error="Drone não chegou ao waypoint"
                    )
                    rospy.logerr(
                        f"  WP {wp_idx+1}/{len(route)} | {pt['plant_type']} "
                        f"({pt['x']:.0f},{pt['y']:.0f}) | PULADO - drone não chegou!"
                    )
            
            # Volta à base após cada rota
            rospy.loginfo(f"\n[BASE] Retornando...")
            base_start = time.time()
            
            # Calcula distância de retorno para consumo de bateria
            dist_to_base = math.sqrt(
                (self.base_position[0] - self.current_x)**2 + 
                (self.base_position[1] - self.current_y)**2
            )
            
            self.goto_and_wait(self.base_position[0], self.base_position[1], CRUISING_ALT)
            self.battery.consume_flight(dist_to_base, speed_ms=2.0)
            self.stats.update_position(self.base_position[0], self.base_position[1])
            base_time = time.time() - base_start
            
            route_time = time.time() - route_start
            
            # Finaliza log da rota
            battery_before_recharge = self.battery.get_percent()
            self.logger.log_route_end(
                route_id=route_idx + 1,
                total_time_s=route_time,
                distance_m=self.stats.total_distance,
                battery_used_percent=100 - battery_before_recharge
            )
            
            rospy.loginfo(f"[ROTA {route_idx+1}] ✓ Completa em {route_time:.1f}s (volta base: {base_time:.1f}s)")
            rospy.loginfo(f"[BATERIA] Antes recarga: {battery_before_recharge:.1f}%")
            rospy.loginfo(f"[BASE] Recarregando (3s)...")
            
            # Simula recarga (bateria cheia novamente)
            self.battery.recharge()
            rospy.sleep(3.0)
            rospy.loginfo(f"[BATERIA] Após recarga: {self.battery.get_percent():.1f}%")
        
        # Estatísticas de duplicatas
        unique_planted = len(self.planted_waypoint_ids)
        skipped_duplicates = total_plants - unique_planted
        if skipped_duplicates > 0:
            rospy.logwarn(f"\n[DUPLICATAS] {skipped_duplicates} waypoints foram pulados por já terem sido plantados!")
            rospy.logwarn(f"[DUPLICATAS] Isso indica que o HGS gerou duplicatas na solução.")
        else:
            rospy.loginfo(f"\n[VALIDAÇÃO] ✓ Nenhuma duplicata detectada durante execução")
        
        rospy.loginfo(f"[RESULTADO] Waypoints únicos plantados: {unique_planted}")
        
        # Imprime resumo final
        self.stats.print_summary()
        
        # Salva logs para CSV e JSON
        rospy.loginfo("\n[LOGGER] Salvando dados da missão...")
        self.logger.set_summary(
            total_waypoints=total_plants,
            successful_plants=self.stats.planted,
            failed_plants=total_plants - self.stats.planted,
            total_routes=len(self.solution.routes),
            total_distance_m=self.stats.total_distance,
            total_time_s=time.time() - self.stats.start_time,
            solver_time_s=getattr(self, 'solver_time', 0),
            field_config={
                "width_m": self.grid_size_x,
                "height_m": self.grid_size_y,
                "line_spacing_m": self.line_spacing,
                "waypoint_spacing_m": self.waypoint_spacing,
                "commodities": {
                    "erva": self.capacity_erva,
                    "arbusto": self.capacity_arbusto,
                    "arvore": self.capacity_arvore
                }
            },
            solver_name=getattr(self, '_solver_name', 'HGS')
        )
        csv_files = self.logger.save_to_csv()
        json_file = self.logger.save_to_json()
        rospy.loginfo(f"[LOGGER] ✓ CSVs: {csv_files}")
        rospy.loginfo(f"[LOGGER] ✓ JSON: {json_file}")
    
    def goto_and_wait(self, x: float, y: float, z: float, tolerance: float = 1.5) -> bool:
        """
        Move o drone para (x,y,z) e espera chegar E ESTABILIZAR.
        Com velocidade 2 m/s, tolerância 1.5m é adequada.
        """
        
        # Calcula distância
        dist = math.hypot(self.current_x - x, self.current_y - y)
        
        # Se já está perto o suficiente, retorna sucesso imediato
        if dist < tolerance:
            return True
        
        # Usa GOTO simples
        try:
            response = self.goto_srv([x, y, z, 0.0])
            if not response.success:
                rospy.logwarn(f"[GOTO] Rejeitado: {response.message}")
                rospy.sleep(0.5)
                response = self.goto_srv([x, y, z, 0.0])
                if not response.success:
                    return False
        except rospy.ServiceException as e:
            rospy.logwarn(f"[GOTO] Erro: {e}")
            return False
        
        # Timeout proporcional à distância (velocidade ~1.5 m/s considerando aceleração)
        # Damos mais tempo: dist / 1.5 + 5 segundos de margem
        timeout = min(90.0, max(10.0, dist / 1.5 + 5.0))
        
        # Espera chegar E ESTABILIZAR
        rate = rospy.Rate(20)  # 20 Hz para detecção mais precisa
        start = time.time()
        stable_count = 0
        STABLE_REQUIRED = 10  # 0.5 segundos estável (10 ciclos a 20Hz)
        
        while time.time() - start < timeout:
            current_dist = math.hypot(self.current_x - x, self.current_y - y)
            
            if current_dist < tolerance:
                stable_count += 1
                if stable_count >= STABLE_REQUIRED:
                    # Drone estável por tempo suficiente
                    return True
            else:
                stable_count = 0  # Reset se saiu da tolerância
            
            rate.sleep()
        
        # Timeout - aceita se estiver razoavelmente perto
        final_dist = math.hypot(self.current_x - x, self.current_y - y)
        if final_dist < tolerance * 2.0:
            rospy.logwarn(f"[GOTO] Aceitando timeout, dist={final_dist:.1f}m")
            return True
        
        rospy.logwarn(f"[GOTO] Timeout ({x:.0f},{y:.0f}), dist={final_dist:.1f}m")
        return False
    
    def _odom_callback(self, msg):
        """Callback de odometria - atualiza posição atual"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_altitude = msg.pose.pose.position.z
    
    def descend_and_plant(self, x: float, y: float, plant_type: str, plant_id: int):
        """
        Desce para plantar e depois sobe.
        Padrão: CRUZEIRO (3m) → DESCIDA (1.5m) → ESTABILIZA → PLANTA → SUBIDA (3m)
        """
        # 0) PAUSA para estabilização horizontal
        rospy.sleep(0.3)
        
        # 1) DESCE para altitude de plantio
        self._change_altitude(PLANTING_ALT)
        
        # 2) PAUSA antes de plantar (garante estabilidade)
        rospy.sleep(0.2)
        
        # 3) PLANTA (spawna no Gazebo)
        self.spawn_plant(f"plant_{plant_id}", x, y, plant_type)
        
        # 4) SOBE de volta para cruzeiro
        self._change_altitude(CRUISING_ALT)
    
    def _change_altitude(self, target_z: float, timeout: float = 8.0):
        """Muda altitude e espera chegar E estabilizar (movimento vertical apenas)"""
        try:
            # Mantém X,Y atuais, só muda Z
            self.goto_srv([self.current_x, self.current_y, target_z, 0.0])
        except rospy.ServiceException:
            return
        
        # Espera atingir altitude (com tolerância de 0.3m)
        rate = rospy.Rate(20)
        start = time.time()
        stable_count = 0
        STABLE_REQUIRED = 6  # 0.3s estável
        
        while time.time() - start < timeout:
            if abs(self.current_altitude - target_z) < 0.3:
                stable_count += 1
                if stable_count >= STABLE_REQUIRED:
                    return
            else:
                stable_count = 0
            rate.sleep()
    
    def spawn_field_boundaries(self):
        """Desenha os limites do talhão no Gazebo com postes e linhas"""
        rospy.loginfo("[TALHÃO] Desenhando limites do campo...")
        
        size_x = self.grid_size_x
        size_y = self.grid_size_y
        
        # Cantos do talhão
        corners = [
            (0, 0, "canto_SW"),
            (size_x, 0, "canto_SE"),
            (size_x, size_y, "canto_NE"),
            (0, size_y, "canto_NW")
        ]
        
        # Spawna postes nos cantos (cilindros vermelhos altos)
        post_height = 3.0
        post_radius = 0.15
        
        for x, y, name in corners:
            post_sdf = f"""
            <?xml version="1.0"?>
            <sdf version="1.6">
              <model name="{name}">
                <static>true</static>
                <link name="link">
                  <visual name="visual">
                    <pose>0 0 {post_height/2} 0 0 0</pose>
                    <geometry>
                      <cylinder>
                        <radius>{post_radius}</radius>
                        <length>{post_height}</length>
                      </cylinder>
                    </geometry>
                    <material>
                      <ambient>1 0 0 1</ambient>
                      <diffuse>1 0 0 1</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """
            self._spawn_model(name, post_sdf, x, y)
        
        # Spawna linhas de borda (caixas finas no chão)
        line_height = 0.05
        line_width = 0.1
        
        # Bordas horizontais (X)
        for y, name_prefix in [(0, "borda_sul"), (size_y, "borda_norte")]:
            line_sdf = f"""
            <?xml version="1.0"?>
            <sdf version="1.6">
              <model name="{name_prefix}">
                <static>true</static>
                <link name="link">
                  <visual name="visual">
                    <pose>{size_x/2} 0 {line_height/2} 0 0 0</pose>
                    <geometry>
                      <box>
                        <size>{size_x} {line_width} {line_height}</size>
                      </box>
                    </geometry>
                    <material>
                      <ambient>1 0.5 0 1</ambient>
                      <diffuse>1 0.5 0 1</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """
            self._spawn_model(name_prefix, line_sdf, 0, y)
        
        # Bordas verticais (Y)
        for x, name_prefix in [(0, "borda_oeste"), (size_x, "borda_leste")]:
            line_sdf = f"""
            <?xml version="1.0"?>
            <sdf version="1.6">
              <model name="{name_prefix}">
                <static>true</static>
                <link name="link">
                  <visual name="visual">
                    <pose>0 {size_y/2} {line_height/2} 0 0 0</pose>
                    <geometry>
                      <box>
                        <size>{line_width} {size_y} {line_height}</size>
                      </box>
                    </geometry>
                    <material>
                      <ambient>1 0.5 0 1</ambient>
                      <diffuse>1 0.5 0 1</diffuse>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """
            self._spawn_model(name_prefix, line_sdf, x, 0)
        
        # Marca a base com um círculo azul
        base_x = self.base_x
        base_y = self.base_y
        base_sdf = f"""
        <?xml version="1.0"?>
        <sdf version="1.6">
          <model name="base_marker">
            <static>true</static>
            <link name="link">
              <visual name="visual">
                <pose>0 0 0.02 0 0 0</pose>
                <geometry>
                  <cylinder>
                    <radius>1.0</radius>
                    <length>0.04</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>0 0 1 1</ambient>
                  <diffuse>0 0 1 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """
        self._spawn_model("base_marker", base_sdf, base_x, base_y)
        
        rospy.loginfo(f"[TALHÃO] Limites: (0,0) até ({size_x},{size_y})")
        rospy.loginfo(f"[TALHÃO] Base: ({base_x}, {base_y})")
    
    def _spawn_model(self, name: str, sdf: str, x: float, y: float):
        """Spawna um modelo SDF no Gazebo"""
        pose = Pose()
        pose.position = Point(x, y, 0.0)
        pose.orientation = Quaternion(0, 0, 0, 1)
        
        try:
            self.spawn_srv(
                model_name=name,
                model_xml=sdf,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )
        except rospy.ServiceException:
            pass  # Ignora se já existe
    
    def load_plant_models(self):
        """Carrega modelos SDF dos arquivos"""
        import rospkg
        rospack = rospkg.RosPack()
        
        try:
            pkg_path = rospack.get_path('plantyo_uav')
        except:
            # Fallback para caminho absoluto
            pkg_path = "/home/flanascimento/rma2025_ws/src/mrs_computer_vision_examples/python/plantYO_UAV"
        
        self.plant_models = {}
        
        for plant_type in ["erva", "arbusto", "arvore"]:
            model_path = f"{pkg_path}/models/{plant_type}_model/model.sdf"
            try:
                with open(model_path, 'r') as f:
                    self.plant_models[plant_type] = f.read()
                rospy.loginfo(f"[MODEL] ✓ Carregado: {plant_type}")
            except Exception as e:
                rospy.logwarn(f"[MODEL] Erro ao carregar {plant_type}: {e}")
                self.plant_models[plant_type] = self._get_fallback_model(plant_type)
        
        rospy.loginfo("[MODEL] Modelos 3D carregados!")
    
    def _get_fallback_model(self, plant_type: str) -> str:
        """Modelo de fallback caso o arquivo não exista"""
        colors = {
            "erva": ("0.3 0.8 0.2", "0.4 0.9 0.3", 0.12),
            "arbusto": ("0.8 0.7 0.0", "1.0 0.85 0.1", 0.30),
            "arvore": ("0.0 0.4 0.1", "0.0 0.5 0.15", 0.50)
        }
        ambient, diffuse, height = colors.get(plant_type, ("0.5 0.5 0.5", "0.6 0.6 0.6", 0.25))
        
        return f"""<?xml version="1.0"?>
        <sdf version="1.6">
          <model name="{plant_type}">
            <static>true</static>
            <link name="link">
              <visual name="visual">
                <pose>0 0 {height/2} 0 0 0</pose>
                <geometry>
                  <cylinder>
                    <radius>{height*0.3}</radius>
                    <length>{height}</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>{ambient} 1</ambient>
                  <diffuse>{diffuse} 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>"""
    
    def get_plant_model(self, plant_type: str) -> str:
        """Retorna modelo SDF para o tipo de planta"""
        if not hasattr(self, 'plant_models'):
            self.load_plant_models()
        
        return self.plant_models.get(plant_type, self.plant_models.get("arbusto", ""))
    
    def spawn_plant(self, name: str, x: float, y: float, plant_type: str = "plant"):
        """Spawna modelo de planta no Gazebo com cor por tipo"""
        pose = Pose()
        pose.position = Point(x, y, 0.0)
        pose.orientation = Quaternion(0, 0, 0, 1)
        
        # Gera modelo com cor específica
        model_xml = self.get_plant_model(plant_type)
        
        try:
            self.spawn_srv(
                model_name=name,
                model_xml=model_xml,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )
        except rospy.ServiceException:
            pass  # Ignora se já existe


if __name__ == "__main__":
    try:
        HGSPlanterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

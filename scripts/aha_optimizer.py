#!/usr/bin/env python3
import rospy
import math
import numpy as np
import random
from threading import Lock
import os
from gazebo_msgs.srv import SpawnModel
from mrs_msgs.srv import Vec4, PathSrv, PathSrvRequest
from mrs_msgs.msg import Reference
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
from typing import List, Dict, Tuple
CRUISING_ALT = 2.5
PLANTING_ALT = 1.0
TOLERANCE = 1  # m
class AdvancedAHA_Optimizer:
    """Artificial Hummingbird Algorithm com restrições agronômicas"""
    
    def __init__(self, population_size=30, max_iterations=100):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Definições agronômicas das plantas
        self.plant_constraints = {
            "Pequi": {
                "type": "large_tree",
                "min_spacing": 10.0,  # Árvore grande precisa muito espaço
                "max_spacing": 15.0,
                "soil_preference": "clay",
                "sun_requirement": "full",
                "growth_months": 36,
                "planting_priority": 1  # Plantar primeiro (bloqueia menos)
            },
            "Baru": {
                "type": "large_tree", 
                "min_spacing": 12.0,  # Ainda maio
                "max_spacing": 18.0,
                "soil_preference": "clay",
                "sun_requirement": "full", 
                "growth_months": 48,
                "planting_priority": 1
            },
            "Cagaita": {
                "type": "medium_tree",
                "min_spacing": 6.0,
                "max_spacing": 10.0,
                "soil_preference": "sandy",
                "sun_requirement": "partial",
                "growth_months": 24,
                "planting_priority": 2
            },
            "Mangaba": {
                "type": "medium_tree",
                "min_spacing": 5.0,
                "max_spacing": 8.0,
                "soil_preference": "sandy",
                "sun_requirement": "partial",
                "growth_months": 18,
                "planting_priority": 2
            },
            "Araticum": {
                "type": "medium_tree",
                "min_spacing": 7.0,
                "max_spacing": 10.0,
                "soil_preference": "mixed",
                "sun_requirement": "full",
                "growth_months": 30,
                "planting_priority": 2
            },
            "Murici": {
                "type": "shrub",
                "min_spacing": 3.0,
                "max_spacing": 5.0,
                "soil_preference": "mixed",
                "sun_requirement": "full",
                "growth_months": 12,
                "planting_priority": 3  # Plantar por último
            },
            "Jenipapo": {
                "type": "shrub",
                "min_spacing": 4.0,
                "max_spacing": 6.0,
                "soil_preference": "mixed",
                "sun_requirement": "partial",
                "growth_months": 15,
                "planting_priority": 3
            },
            "Bacupari": {
                "type": "shrub",
                "min_spacing": 2.5,
                "max_spacing": 4.0,
                "soil_preference": "sandy",
                "sun_requirement": "partial",
                "growth_months": 10,
                "planting_priority": 3
            },
            "Guabiroba": {
                "type": "shrub",
                "min_spacing": 3.5,
                "max_spacing": 5.0,
                "soil_preference": "clay",
                "sun_requirement": "full",
                "growth_months": 14,
                "planting_priority": 3
            },
            "Gabiroba-do-campo": {
                "type": "shrub",
                "min_spacing": 3.0,
                "max_spacing": 4.5,
                "soil_preference": "sandy",
                "sun_requirement": "full",
                "growth_months": 11,
                "planting_priority": 3
            },
            "Capim": {
                "type": "grass",
                "min_spacing": 2.0,  # Espaçamento fixo reduzido conforme texto
                "max_spacing": 2.0,  # Fixo
                "soil_preference": "mixed",
                "sun_requirement": "full",
                "growth_months": 6,  # Cobertura rápida
                "planting_priority": 3  # Plantado por último
            }
        }
        
    def calculate_distance_between_points(self, p1: Dict, p2: Dict) -> float:
        """Calcula distância euclidiana entre dois pontos"""
        return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
        
    def calculate_total_distance(self, points: List[Dict], start_pos=(0, 0)) -> float:
        """Calcula a distância total da trajetória"""
        if len(points) == 0:
            return 0
        
        total_distance = 0
        current_pos = start_pos
        
        for point in points:
            next_pos = (point['x'], point['y'])
            distance = math.sqrt((next_pos[0] - current_pos[0])**2 + 
                               (next_pos[1] - current_pos[1])**2)
            total_distance += distance
            current_pos = next_pos
            
        return total_distance
    
    def calculate_spacing_violations(self, ordered_points: List[Dict]) -> float:
        """Calcula penalidades por violações de espaçamento entre plantas"""
        spacing_penalty = 0
        
        for i in range(len(ordered_points)):
            current_plant = ordered_points[i]
            current_constraints = self.plant_constraints[current_plant['name']]
            
            # Verifica distância para todas as outras plantas (não só adjacentes)
            for j in range(len(ordered_points)):
                if i != j:
                    other_plant = ordered_points[j]
                    other_constraints = self.plant_constraints[other_plant['name']]
                    
                    distance = self.calculate_distance_between_points(current_plant, other_plant)
                    
                    # Distância mínima necessária (raio de cada planta)
                    min_required = (current_constraints['min_spacing'] + 
                                  other_constraints['min_spacing']) / 2
                    
                    if distance < min_required:
                        # Penalidade proporcional à violação
                        violation = min_required - distance
                        spacing_penalty += violation * 15  # Penalidade alta
        
        return spacing_penalty
    
    def calculate_soil_compatibility_penalty(self, ordered_points: List[Dict]) -> float:
        """Penaliza plantas com necessidades de solo incompatíveis próximas"""
        soil_penalty = 0
        
        for i in range(len(ordered_points) - 1):
            current_plant = ordered_points[i]
            next_plant = ordered_points[i + 1]
            
            current_soil = self.plant_constraints[current_plant['name']]['soil_preference']
            next_soil = self.plant_constraints[next_plant['name']]['soil_preference']
            
            distance = self.calculate_distance_between_points(current_plant, next_plant)
            
            # Se solos muito diferentes estão próximos
            if current_soil != next_soil and current_soil != "mixed" and next_soil != "mixed":
                if distance < 8.0:  # Threshold para compatibilidade de solo
                    soil_penalty += (8.0 - distance) * 3
                    
        return soil_penalty
    
    def calculate_planting_sequence_penalty(self, ordered_points: List[Dict]) -> float:
        """Penaliza sequências de plantio ineficientes"""
        sequence_penalty = 0
        
        for i in range(len(ordered_points) - 1):
            current_priority = self.plant_constraints[ordered_points[i]['name']]['planting_priority']
            next_priority = self.plant_constraints[ordered_points[i + 1]['name']]['planting_priority']
            
            # Penaliza se planta de prioridade baixa vem antes de alta prioridade
            if current_priority > next_priority:
                sequence_penalty += (current_priority - next_priority) * 5
                
        return sequence_penalty
    
    def calculate_growth_efficiency_bonus(self, ordered_points: List[Dict]) -> float:
        """Bônus por plantar espécies de crescimento rápido estrategicamente"""
        bonus = 0
        
        # Bônus se plantas de crescimento rápido estão no final
        for i, point in enumerate(ordered_points):
            growth_months = self.plant_constraints[point['name']]['growth_months']
            
            # Plantas rápidas (< 15 meses) no final ganham bônus
            if growth_months < 15 and i > len(ordered_points) * 0.6:
                bonus += 2
                
        return bonus
    
    def fitness_function(self, solution: List[int], points: List[Dict]) -> float:
        """Função de fitness multi-objetivo avançada"""
        ordered_points = [points[i] for i in solution]
        
        # 1. Distância de viagem (objetivo principal)
        travel_distance = self.calculate_total_distance(ordered_points)
        
        # 2. Violações de espaçamento (crítico)
        spacing_penalty = self.calculate_spacing_violations(ordered_points)
        
        # 3. Compatibilidade de solo
        soil_penalty = self.calculate_soil_compatibility_penalty(ordered_points)
        
        # 4. Sequência de plantio eficiente
        sequence_penalty = self.calculate_planting_sequence_penalty(ordered_points)
        
        # 5. Eficiência de crescimento
        growth_bonus = self.calculate_growth_efficiency_bonus(ordered_points)
        
        # Fitness final (minimizar - valores menores são melhores)
        total_fitness = (travel_distance + 
                        spacing_penalty * 3.0 +      # Peso muito alto para espaçamento
                        soil_penalty * 1.5 +
                        sequence_penalty * 2.0 -     # Penalidade média para sequência
                        growth_bonus)                # Subtrai bônus
        
        return total_fitness
    
    def initialize_population(self, num_points: int) -> List[List[int]]:
        """Inicializa população com permutações aleatórias"""
        population = []
        base_sequence = list(range(num_points))
        
        for _ in range(self.population_size):
            sequence = base_sequence.copy()
            random.shuffle(sequence)
            population.append(sequence)
            
        return population
    
    def guided_foraging(self, hummingbird: List[int], best_solution: List[int], 
                       iteration: int, max_iter: int) -> List[int]:
        """Comportamento de forrageamento guiado"""
        new_hummingbird = hummingbird.copy()
        decay_factor = 1 - (iteration / max_iter)
        
        if random.random() < 0.5 * decay_factor:
            cut_point = random.randint(1, len(hummingbird) - 1)
            new_hummingbird = best_solution[:cut_point] + [x for x in hummingbird if x not in best_solution[:cut_point]]
        
        if random.random() < 0.3:
            new_hummingbird = self.two_opt_mutation(new_hummingbird)
            
        return new_hummingbird
    
    def territorial_foraging(self, hummingbird: List[int]) -> List[int]:
        """Comportamento de forrageamento territorial"""
        new_hummingbird = hummingbird.copy()
        
        if len(new_hummingbird) > 1:
            idx1, idx2 = random.sample(range(len(new_hummingbird)), 2)
            new_hummingbird[idx1], new_hummingbird[idx2] = new_hummingbird[idx2], new_hummingbird[idx1]
        
        return new_hummingbird
    
    def migration_foraging(self, hummingbird: List[int]) -> List[int]:
        """Comportamento de forrageamento por migração"""
        new_hummingbird = hummingbird.copy()
        
        if len(new_hummingbird) > 2:
            start = random.randint(0, len(new_hummingbird) - 2)
            end = random.randint(start + 1, len(new_hummingbird) - 1)
            new_hummingbird[start:end+1] = list(reversed(new_hummingbird[start:end+1]))
        
        return new_hummingbird
    
    def two_opt_mutation(self, solution: List[int]) -> List[int]:
        """Mutação 2-opt para melhoramento local"""
        if len(solution) < 4:
            return solution
            
        new_solution = solution.copy()
        i = random.randint(0, len(solution) - 2)
        j = random.randint(i + 1, len(solution) - 1)
        
        new_solution[i:j+1] = list(reversed(new_solution[i:j+1]))
        return new_solution
    
    def optimize_trajectory(self, points: List[Dict]) -> Tuple[List[Dict], float, Dict]:
        """Executa o algoritmo AHA para otimizar a trajetória"""
        if len(points) <= 1:
            return points, 0, {}
            
        num_points = len(points)
        population = self.initialize_population(num_points)
        
        # Avalia população inicial
        fitness_values = [self.fitness_function(ind, points) for ind in population]
        
        # Encontra melhor solução inicial
        best_idx = np.argmin(fitness_values)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        
        rospy.loginfo(f"[AHA-ADV] Iniciando otimização avançada. Fitness inicial: {self.best_fitness:.2f}")
        
        # Loop principal do algoritmo
        for iteration in range(self.max_iterations):
            new_population = []
            
            for i, hummingbird in enumerate(population):
                behavior_prob = random.random()
                
                if behavior_prob < 0.4:
                    new_hummingbird = self.guided_foraging(hummingbird, self.best_solution, 
                                                         iteration, self.max_iterations)
                elif behavior_prob < 0.7:
                    new_hummingbird = self.territorial_foraging(hummingbird)
                else:
                    new_hummingbird = self.migration_foraging(hummingbird)
                
                new_fitness = self.fitness_function(new_hummingbird, points)
                
                if new_fitness < fitness_values[i]:
                    new_population.append(new_hummingbird)
                    fitness_values[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_hummingbird.copy()
                        self.best_fitness = new_fitness
                else:
                    new_population.append(hummingbird)
            
            population = new_population
            
            if iteration % 25 == 0:
                rospy.loginfo(f"[AHA-ADV] Iteração {iteration}: Fitness = {self.best_fitness:.2f}")
        
        optimized_points = [points[i] for i in self.best_solution]
        
        # Calcula métricas detalhadas da solução final
        final_metrics = self.calculate_detailed_metrics(optimized_points)
        
        rospy.loginfo(f"[AHA-ADV] Otimização concluída. Fitness final: {self.best_fitness:.2f}")
        return optimized_points, self.best_fitness, final_metrics
    
    def calculate_detailed_metrics(self, optimized_points: List[Dict]) -> Dict:
        """Calcula métricas detalhadas da solução otimizada"""
        travel_dist = self.calculate_total_distance(optimized_points)
        spacing_violations = self.calculate_spacing_violations(optimized_points)
        soil_issues = self.calculate_soil_compatibility_penalty(optimized_points)
        sequence_issues = self.calculate_planting_sequence_penalty(optimized_points)
        
        return {
            'travel_distance': travel_dist,
            'spacing_violations': spacing_violations,
            'soil_compatibility_issues': soil_issues,
            'sequence_inefficiencies': sequence_issues
        }
class OptimizedPlanterNode:
    def __init__(self):
        rospy.init_node("advanced_optimized_drone_planter", anonymous=True)
        
        # Parâmetros ROS
        self.optimize_trajectory = rospy.get_param("~optimize_trajectory", True)
        self.use_line_arrangement = rospy.get_param("~use_line_arrangement", False)
        self.line_spacing = rospy.get_param("~line_spacing", 2.5)
        self.line_direction = rospy.get_param("~line_direction", "horizontal")
        self.line_start_x = rospy.get_param("~line_start_x", 0.0)
        self.line_start_y = rospy.get_param("~line_start_y", 0.0)
        
        self.muda_pub = rospy.Publisher("/muda_alert", String, queue_size=1)
        
        # Pontos com coordenadas que criam conflitos interessantes
        self.original_points = [
            # Prioridade 1: Árvores grandes (plantar primeiro)
            {"name":"Pequi",             "x":8.0,  "y":6.0,  "yaw":0.0},   # Centro-esquerda
            {"name":"Baru",              "x":12.0, "y":8.0, "yaw":0.0},    # Centro-direita
            
            # Prioridade 2: Árvores médias
            {"name":"Cagaita",           "x":5.0,  "y":2.0,  "yaw":0.0},   # Sudoeste
            {"name":"Mangaba",           "x":15.0, "y":12.0, "yaw":0.0},   # Nordeste
            {"name":"Araticum",          "x":3.0,  "y":10.0,  "yaw":0.0},  # Noroeste
            
            # Prioridade 3: Arbustos
            {"name":"Murici",            "x":9.0, "y":7.0,  "yaw":0.0},    # Próximo do Pequi
            {"name":"Jenipapo",          "x":13.0, "y":9.0, "yaw":0.0},    # Próximo do Baru
            {"name":"Bacupari",          "x":6.0,  "y":3.0, "yaw":0.0},    # Próximo da Cagaita
            {"name":"Guabiroba",         "x":4.0,  "y":11.0, "yaw":0.0},   # Próximo do Araticum
            {"name":"Gabiroba-do-campo", "x":14.0, "y":13.0, "yaw":0.0},   # Próximo da Mangaba
            
            # Prioridade 3: Capim (cobertura rápida - plantado por último)
            {"name":"Capim",             "x":10.0, "y":4.0,  "yaw":0.0},   # Espaçamento 2m
            {"name":"Capim",             "x":12.0, "y":4.0,  "yaw":0.0},   # +2m
            {"name":"Capim",             "x":14.0, "y":4.0,  "yaw":0.0},   # +2m
        ]
        
        # Otimiza trajetória
        self.points = self.process_trajectory()
        
        self.current_idx = 0
        self.planting = False
        self.lock = Lock()
        # Serviços
        rospy.wait_for_service('/uav1/trajectory_generation/path')
        self.path_srv = rospy.ServiceProxy('/uav1/trajectory_generation/path', PathSrv)
        rospy.wait_for_service('/uav1/control_manager/goto')
        self.goto_srv = rospy.ServiceProxy('/uav1/control_manager/goto', Vec4)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self.spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        # Subscreve odometria
        self.sub_odom = rospy.Subscriber(
            '/uav1/estimation_manager/odom_main',
            Odometry,
            self.odom_callback,
            queue_size=1
        )
        rospy.sleep(1.0)
        self.send_trajectory(self.points, prepend_current=False)
    def process_trajectory(self) -> List[Dict]:
        """Processa a trajetória baseado nos parâmetros"""
        points = self.original_points.copy()
        
        if self.use_line_arrangement:
            # Arranjo em linha reta
            points = self.arrange_points_in_line(points)
            rospy.loginfo(f"[OPTIMIZER] Pontos organizados em linha {self.line_direction}")
            
        elif self.optimize_trajectory:
            # Otimização avançada com AHA
            optimizer = AdvancedAHA_Optimizer(population_size=40, max_iterations=120)
            
            # Calcula métricas da configuração original
            original_distance = optimizer.calculate_total_distance(points)
            original_spacing_violations = optimizer.calculate_spacing_violations(points)
            
            rospy.loginfo(f"[OPTIMIZER] === ANÁLISE INICIAL ===")
            rospy.loginfo(f"[OPTIMIZER] Distância original: {original_distance:.2f}m")
            rospy.loginfo(f"[OPTIMIZER] Violações de espaçamento: {original_spacing_violations:.2f}")
            
            # Executa otimização
            optimized_points, optimized_fitness, metrics = optimizer.optimize_trajectory(points)
            
            # Calcula melhorias
            improvement_distance = ((original_distance - metrics['travel_distance']) / original_distance) * 100
            improvement_spacing = original_spacing_violations - metrics['spacing_violations']
            
            rospy.loginfo(f"[OPTIMIZER] === RESULTADOS FINAIS ===")
            rospy.loginfo(f"[OPTIMIZER] Distância otimizada: {metrics['travel_distance']:.2f}m")
            rospy.loginfo(f"[OPTIMIZER] Melhoria em distância: {improvement_distance:.1f}%")
            rospy.loginfo(f"[OPTIMIZER] Violações de espaçamento reduzidas: {improvement_spacing:.2f}")
            rospy.loginfo(f"[OPTIMIZER] Problemas de solo: {metrics['soil_compatibility_issues']:.2f}")
            rospy.loginfo(f"[OPTIMIZER] Ineficiências de sequência: {metrics['sequence_inefficiencies']:.2f}")
            
            points = optimized_points
            
        # Log da sequência final detalhada
        rospy.loginfo("[OPTIMIZER] === SEQUÊNCIA DE PLANTIO OTIMIZADA ===")
        for i, point in enumerate(points):
            plant_type = optimizer.plant_constraints[point['name']]['type'] if hasattr(self, 'optimizer') else "unknown"
            rospy.loginfo(f"  {i+1}. {point['name']} ({plant_type}) - ({point['x']:.1f}, {point['y']:.1f})")
            
        return points
    
    def arrange_points_in_line(self, points: List[Dict]) -> List[Dict]:
        """Organiza pontos em linha reta com espaçamento definido"""
        arranged_points = []
        start_point = (self.line_start_x, self.line_start_y)
        
        for i, point in enumerate(points):
            new_point = point.copy()
            
            if self.line_direction == 'horizontal':
                new_point['x'] = start_point[0] + (i * self.line_spacing)
                new_point['y'] = start_point[1]
            elif self.line_direction == 'vertical':
                new_point['x'] = start_point[0]
                new_point['y'] = start_point[1] + (i * self.line_spacing)
            elif self.line_direction == 'diagonal':
                offset = i * self.line_spacing / math.sqrt(2)
                new_point['x'] = start_point[0] + offset
                new_point['y'] = start_point[1] + offset
            
            arranged_points.append(new_point)
        
        return arranged_points
    def send_trajectory(self, pts, prepend_current):
        req = PathSrvRequest()
        req.path.header.frame_id = ""
        req.path.header.stamp = rospy.Time.now()
        req.path.use_heading = True
        req.path.fly_now = True
        req.path.dont_prepend_current_state = prepend_current
        req.path.max_execution_time = 90.0  # Mais tempo para trajetória complexa
        req.path.max_deviation_from_path = 0.0
        for p in pts:
            r = Reference()
            r.position.x = p["x"]
            r.position.y = p["y"]
            r.position.z = CRUISING_ALT
            r.heading = p["yaw"]
            req.path.points.append(r)
            rospy.loginfo(f"[PLAN] → {p['name']} at ({p['x']:.1f},{p['y']:.1f})")
        resp = self.path_srv(req)
        if not resp.success:
            rospy.logerr("[PLAN] failed: " + resp.message)
        else:
            rospy.loginfo("[PLAN] trajectory sent")
    def odom_callback(self, msg: Odometry):
        with self.lock:
            if self.planting or self.current_idx >= len(self.points):
                return
            pos = msg.pose.pose.position
            pt = self.points[self.current_idx]
            dist = math.hypot(pos.x - pt["x"], pos.y - pt["y"])
            if dist < TOLERANCE:
                rospy.loginfo(f"[REACHED] {pt['name']} (idx={self.current_idx})")
                self.planting = True
                rospy.sleep(0.1)
                self.goto([pt["x"], pt["y"], CRUISING_ALT, pt["yaw"]])
                rospy.sleep(0.5)
                self.do_plant(pt)
                self.current_idx += 1
                if self.current_idx < len(self.points):
                    rem = self.points[self.current_idx:]
                    rospy.sleep(0.5)
                    self.send_trajectory(rem, prepend_current=False)
                else:
                    rospy.loginfo("[FINISH] Todas as mudas plantadas com otimização avançada!")
                self.planting = False
    def do_plant(self, pt):
        name, x, y, yaw = pt["name"], pt["x"], pt["y"], pt["yaw"]
        rospy.loginfo(f"[PLANT] descending for {name}")
        self.goto([x, y, PLANTING_ALT, yaw])
        rospy.loginfo(f"[PLANT] spawning '{name}'")
        self.spawn_plant(name, x, y)
        self.muda_pub.publish(name)
        rospy.loginfo(f"[PLANT] ascending after {name}")
        self.goto([x, y, CRUISING_ALT, yaw])
    def goto(self, vec4):
        try:
            self.goto_srv(vec4)
            rospy.sleep(4.0)
        except rospy.ServiceException as e:
            rospy.logwarn(f"[GOTO] failed: {e}")
    def spawn_plant(self, name, x, y):
        # Carrega o modelo SDF externo
        if not hasattr(self, "model_xml"):
            with open('/usr/share/gazebo-11/models/big_plant/model.sdf','r') as f:
                self.model_xml = f.read()
        pose = Pose()
        pose.position = Point(x, y, 0.0)
        pose.orientation = Quaternion(0, 0, 0, 1)
        try:
            self.spawn_srv(
                model_name=name,
                model_xml=self.model_xml,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )
            rospy.loginfo(f"[SPAWN] planted '{name}'")
        except rospy.ServiceException as e:
            rospy.logerr(f"[SPAWN] failed: {e}")
if __name__ == "__main__":
    try:
        OptimizedPlanterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

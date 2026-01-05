#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nome do Arquivo: plantio_drone.py

Sistema Completo de Integração e Simulação para Plantio com Drone.
- Recebe missões de plantio em grid de uma interface web (index.html).
- Comunica-se com o sistema MRS (Multi-Robot System) para controlar o drone.
- Executa a missão em uma simulação no Gazebo, "plantando" objetos.
- Envia telemetria em tempo real de volta para a interface web.
"""

import rospy
import json
import threading
import math

# --- Bibliotecas do Servidor Web ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- Bibliotecas para Cálculos ---
import numpy as np
from shapely.geometry import Point, Polygon

# --- Mensagens e Serviços do ROS ---
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion
from mavros_msgs.msg import State
from mrs_msgs.srv import PathSrv, PathSrvRequest, Vec4
from mrs_msgs.msg import Path, Reference
from gazebo_msgs.srv import SpawnModel

# --- Constantes de Voo ---
CRUISING_ALT = 3.0
PLANTING_ALT = 1.0
TOLERANCE  = 0.5  # Tolerância em metros para chegar no waypoint

class DronePlantingSimulator:
    def __init__(self):
        rospy.init_node('drone_planting_simulator_node', anonymous=True)
        
        # --- Configuração do Flask ---
        self.app = Flask(__name__, template_folder='.')
        CORS(self.app)
        
        # --- Estado e Missão ---
        self.drone_state = {
            'armed': False, 'mode': 'MANUAL', 'gps_fix': False, 'num_sat': 0,
            'position': {'lat': 0.0, 'lon': 0.0},
            'battery': 0.0,
            'mission_status': 'Ocioso'
        }
        self.mission_waypoints = []
        self.current_waypoint_idx = 0
        self.is_planting = False
        self.lock = Lock()
        
        # --- Setup ROS e Flask ---
        self.setup_mrs_services_and_pubs()
        self.setup_mrs_subscribers()
        self.setup_flask_routes()
        
        # Thread para enviar telemetria para a web
        threading.Thread(target=self.telemetry_updater, daemon=True).start()

        rospy.loginfo("Backend de Simulação de Plantio INICIADO.")

    def setup_mrs_services_and_pubs(self):
        self.namespace = rospy.get_param('~namespace', 'uav1')
        
        # Publishers de telemetria para a web (via ROS)
        self.telemetry_pub = rospy.Publisher('/drone/telemetry_status', String, queue_size=1)
        
        # Serviços MRS e Gazebo
        try:
            rospy.wait_for_service(f'/{self.namespace}/control_manager/path', timeout=10)
            self.path_service = rospy.ServiceProxy(f'/{self.namespace}/control_manager/path', PathSrv)
            
            rospy.wait_for_service(f'/{self.namespace}/control_manager/goto', timeout=5)
            self.goto_service = rospy.ServiceProxy(f'/{self.namespace}/control_manager/goto', Vec4)

            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5)
            self.spawn_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            
            rospy.loginfo("Serviços MRS e Gazebo conectados!")
        except rospy.ROSException as e:
            rospy.logwarn(f"Alguns serviços não foram encontrados: {e}. O modo de simulação pode ser limitado.")
            self.path_service = self.goto_service = self.spawn_service = None

    def setup_mrs_subscribers(self):
        rospy.Subscriber(f'/{self.namespace}/estimation_manager/odom_main', Odometry, self.odom_callback)
        rospy.Subscriber(f'/{self.namespace}/mavros/state', State, lambda msg: self.drone_state.update({'armed': msg.armed, 'mode': msg.mode}))
        rospy.Subscriber(f'/{self.namespace}/odometry/gps', Odometry, lambda msg: self.drone_state['position'].update({'lat': msg.pose.pose.position.x, 'lon': msg.pose.pose.position.y}))

    def setup_flask_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/ros/upload_mission', methods=['POST'])
        def upload_mission():
            # A lógica de receber os dados e gerar o grid permanece a mesma
            # ...
            # Após gerar os waypoints:
            # self.mission_waypoints = self.generate_grid_waypoints(...)
            # self.start_mission()
            # return jsonify({'success': True, 'message': 'Missão recebida e iniciada!'})

            data = request.json
            rospy.loginfo("Recebida solicitação de missão de grid via web.")

            try:
                # Extrai os parâmetros enviados pelo seu painel web
                polygon_coords = data['polygon']
                grid_pattern = data['grid_pattern'] 
                spacing = data['spacing'] 
                species_sequence = data['species_sequence']

                # Validação básica
                if not all([polygon_coords, grid_pattern, spacing, species_sequence]):
                    return jsonify({'success': False, 'message': 'Dados da missão incompletos.'})

                # Gera os waypoints e inicia a missão
                self.mission_waypoints = self.generate_grid_waypoints(polygon_coords, grid_pattern, spacing, species_sequence)
                self.start_mission()

                return jsonify({'success': True, 'message': f'Missão gerada com {len(self.mission_waypoints)} pontos e iniciada.'})

            except Exception as e:
                rospy.logerr(f"Erro ao gerar a missão de grid: {e}")
                return jsonify({'success': False, 'message': str(e)})


    def start_mission(self):
        if not self.mission_waypoints:
            rospy.logwarn("Nenhum waypoint na missão para iniciar.")
            self.drone_state['mission_status'] = 'Falha: Sem Waypoints'
            return

        self.current_waypoint_idx = 0
        self.is_planting = False
        self.drone_state['mission_status'] = 'Em andamento'
        rospy.loginfo("Iniciando missão de plantio...")
        self.send_trajectory(self.mission_waypoints)


    def odom_callback(self, msg: Odometry):
        # Este callback agora gerencia a execução da missão
        with self.lock:
            if self.is_planting or self.current_waypoint_idx >= len(self.mission_waypoints):
                return

            pos = msg.pose.pose.position
            target_point = self.mission_waypoints[self.current_waypoint_idx]
            
            # Converte Lat/Lon para a posição local do Gazebo se necessário
            # Para simplificar, vamos assumir que o drone está em um sistema de coordenadas local
            # onde os waypoints do grid são gerados. Se não, uma conversão é necessária aqui.
            target_x, target_y = target_point['x'], target_point['y']

            dist = math.hypot(pos.x - target_x, pos.y - target_y)

            if dist < TOLERANCE:
                self.is_planting = True
                rospy.loginfo(f"[ALVO ALCANÇADO] Ponto {self.current_waypoint_idx + 1}/{len(self.mission_waypoints)}")
                
                # Inicia a rotina de plantio em um novo thread para não bloquear o callback
                threading.Thread(target=self.do_plant_routine, args=(target_point,)).start()

    def do_plant_routine(self, point_data):
        """ Executa a sequência completa de plantio para um ponto. """
        name, x, y = point_data["name"], point_data["x"], point_data["y"]

        rospy.loginfo(f"[PLANTIO] Descendo para plantar {name}...")
        self.goto([x, y, PLANTING_ALT, 0])
        
        rospy.loginfo(f"[PLANTIO] Dispensando semente no Gazebo...")
        self.spawn_plant_in_gazebo(name, x, y)
        rospy.sleep(2.0) # Simula o tempo de plantio

        rospy.loginfo(f"[PLANTIO] Subindo após plantar...")
        self.goto([x, y, CRUISING_ALT, 0])

        with self.lock:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.mission_waypoints):
                rospy.loginfo("[MISSÃO CONCLUÍDA] Todos os pontos foram plantados!")
                self.drone_state['mission_status'] = 'Concluída'
            
            self.is_planting = False


    def spawn_plant_in_gazebo(self, name, x, y):
        # Esta função cria um objeto no Gazebo para simular a planta
        if not self.spawn_service: return

        # Modelo SDF simples para uma planta
        model_xml = """
        <sdf version='1.6'>
          <model name='plant_marker'>
            <pose>{x} {y} 0 0 0 0</pose>
            <link name='link'>
              <visual name='visual'>
                <geometry><cylinder><radius>0.05</radius><length>0.3</length></cylinder></geometry>
                <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Green</name></script></material>
              </visual>
            </link>
            <static>true</static>
          </model>
        </sdf>""".format(x=x, y=y)
        
        try:
            model_name = f"{name.replace(' ', '_')}_{self.current_waypoint_idx}"
            self.spawn_service(model_name=model_name, model_xml=model_xml, robot_namespace='', initial_pose=Pose(), reference_frame='world')
            rospy.loginfo(f"Planta '{model_name}' criada no Gazebo.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Falha ao criar planta no Gazebo: {e}")

    # --- Funções Utilitárias e de Comunicação ---
    
    def telemetry_updater(self):
        """ Envia o status do drone para a interface web a cada 1 segundo. """
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            try:
                # Converte o dicionário de estado para uma string JSON e publica
                status_str = json.dumps(self.drone_state)
                self.telemetry_pub.publish(status_str)
            except Exception as e:
                rospy.logerr(f"Erro no atualizador de telemetria: {e}")
            rate.sleep()
            
    # ... (As funções `generate_grid_waypoints` e `calculate_offset_point` da resposta anterior entram aqui) ...
    # ... (A função `send_trajectory` precisa ser adaptada para usar o serviço do MRS) ...
    def goto(self, vec4_array):
        if not self.goto_service: return
        try:
            self.goto_service(vec4_array)
            rospy.sleep(4.0) # Tempo para o drone chegar
        except rospy.ServiceException as e:
            rospy.logwarn(f"Serviço GOTO falhou: {e}")

    def run(self):
        flask_thread = threading.Thread(target=lambda: self.app.run(host='0.0.0.0', port=5000))
        flask_thread.daemon = True
        flask_thread.start()
        rospy.spin()

if __name__ == '__main__':
    try:
        simulator = DronePlantingSimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        pass

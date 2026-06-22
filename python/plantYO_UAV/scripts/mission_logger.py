#!/usr/bin/env python3
"""
Mission Logger - Sistema de Logging para Análise de Dados

Salva métricas detalhadas de cada missão em formato CSV e JSON
para posterior análise na dissertação.

Métricas coletadas:
- Tempo por waypoint
- Distância percorrida
- Consumo de bateria
- Eficiência do plantio
- Rotas e clientes visitados
"""

import os
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
import rospy


@dataclass
class WaypointLog:
    """Log de um único waypoint"""
    timestamp: float
    waypoint_id: int
    x: float
    y: float
    z: float
    plant_type: str
    route_id: int
    client_id: int
    time_to_reach: float  # segundos
    time_to_plant: float  # segundos
    distance_from_previous: float  # metros
    battery_before: float  # percentual
    battery_after: float  # percentual
    success: bool


@dataclass
class RouteLog:
    """Log de uma rota completa"""
    route_id: int
    start_time: float
    end_time: float
    num_clients: int
    num_waypoints: int
    total_distance: float
    battery_start: float
    battery_end: float
    battery_consumed: float
    waypoints: List[int] = field(default_factory=list)


@dataclass
class MissionSummary:
    """Resumo geral da missão"""
    mission_id: str
    date: str
    start_time: float
    end_time: float
    total_duration: float  # segundos
    
    # Grid config
    grid_size_x: float
    grid_size_y: float
    waypoint_spacing: float
    
    # Resultados
    total_waypoints: int
    planted_waypoints: int
    skipped_waypoints: int
    success_rate: float
    
    # Rotas
    num_routes: int
    total_distance: float
    avg_distance_per_route: float
    
    # Bateria
    battery_capacity_mah: float
    total_battery_consumed: float
    avg_consumption_per_plant: float
    recharges: int
    
    # Eficiência
    plants_per_minute: float
    meters_per_plant: float
    efficiency_score: float  # plantas / (distância * tempo)
    
    # Solver
    solver_time: float
    solver_name: str


class MissionLogger:
    """
    Logger de missão que salva dados em CSV e JSON.
    
    Uso:
        logger = MissionLogger(mission_name="teste_01")
        logger.log_waypoint(...)
        logger.log_route(...)
        logger.save()
    """
    
    def __init__(self, mission_name: Optional[str] = None, output_dir: Optional[str] = None):
        # Gera nome único se não fornecido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mission_id = mission_name or f"mission_{timestamp}"
        
        # Diretório de saída
        if output_dir is None:
            # Usa pasta logs no pacote
            import rospkg
            try:
                rospack = rospkg.RosPack()
                pkg_path = rospack.get_path('plantyo_uav')
            except:
                pkg_path = "/home/flanascimento/rma2025_ws/src/mrs_computer_vision_examples/python/plantYO_UAV"
            output_dir = os.path.join(pkg_path, "logs")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Arquivos
        self.waypoint_csv = os.path.join(self.output_dir, f"{self.mission_id}_waypoints.csv")
        self.routes_csv = os.path.join(self.output_dir, f"{self.mission_id}_routes.csv")
        self.summary_json = os.path.join(self.output_dir, f"{self.mission_id}_summary.json")
        
        # Dados em memória
        self.waypoint_logs: List[WaypointLog] = []
        self.route_logs: List[RouteLog] = []
        self.mission_start = time.time()
        
        # Configuração (será preenchida depois)
        self.config: Dict = {}
        
        # Posição anterior para calcular distância
        self.last_x = 0.0
        self.last_y = 0.0
        
        rospy.loginfo(f"[LOGGER] Iniciado: {self.mission_id}")
        rospy.loginfo(f"[LOGGER] Saída: {self.output_dir}")
    
    def set_config(self, config: Dict):
        """Define configuração da missão"""
        self.config = config
    
    def log_waypoint(
        self,
        waypoint_id: int,
        x: float, y: float,
        plant_type: str,
        route_id: int,
        client_id: int,
        success: bool = True,
        duration_s: float = 0.0,
        battery_percent: float = 100.0,
        error: Optional[str] = None,
        z: float = 0.0
    ):
        """
        Registra um waypoint visitado.
        
        Interface simplificada para fácil uso no código principal.
        """
        import math
        
        # Calcula distância do ponto anterior
        distance = math.hypot(x - self.last_x, y - self.last_y)
        self.last_x = x
        self.last_y = y
        
        log = WaypointLog(
            timestamp=time.time(),
            waypoint_id=waypoint_id,
            x=x, y=y, z=z,
            plant_type=plant_type,
            route_id=route_id,
            client_id=client_id,
            time_to_reach=duration_s * 0.7,  # ~70% para chegar
            time_to_plant=duration_s * 0.3,  # ~30% para plantar
            distance_from_previous=distance,
            battery_before=battery_percent + 1,  # estimativa
            battery_after=battery_percent,
            success=success
        )
        
        if error:
            log.error = error
        
        self.waypoint_logs.append(log)
    
    def log_route_start(self, route_id: int, num_clients: int, num_waypoints: int):
        """
        Inicia o logging de uma rota.
        Deve ser chamado no início de cada rota.
        """
        self._current_route = {
            'route_id': route_id,
            'start_time': time.time(),
            'num_clients': num_clients,
            'num_waypoints': num_waypoints,
            'battery_start': 100.0,  # Será atualizado se necessário
        }
    
    def log_route_end(
        self,
        route_id: int,
        total_time_s: float,
        distance_m: float,
        battery_used_percent: float
    ):
        """
        Finaliza o logging de uma rota.
        Deve ser chamado ao final de cada rota.
        """
        # Recupera dados da rota se existirem
        route_data = getattr(self, '_current_route', None)
        
        if route_data and route_data.get('route_id') == route_id:
            start_time = route_data['start_time']
            num_clients = route_data['num_clients']
            num_waypoints = route_data['num_waypoints']
            battery_start = route_data.get('battery_start', 100.0)
        else:
            start_time = time.time() - total_time_s
            num_clients = 0
            num_waypoints = 0
            battery_start = 100.0
        
        battery_end = battery_start - battery_used_percent
        
        log = RouteLog(
            route_id=route_id,
            start_time=start_time,
            end_time=time.time(),
            num_clients=num_clients,
            num_waypoints=num_waypoints,
            total_distance=distance_m,
            battery_start=battery_start,
            battery_end=battery_end,
            battery_consumed=battery_used_percent,
            waypoints=[]
        )
        
        self.route_logs.append(log)
    
    def log_route(
        self,
        route_id: int,
        start_time: float,
        end_time: float,
        num_clients: int,
        num_waypoints: int,
        total_distance: float,
        battery_start: float,
        battery_end: float,
        waypoint_ids: List[int]
    ):
        """Registra uma rota completa (método legado)"""
        log = RouteLog(
            route_id=route_id,
            start_time=start_time,
            end_time=end_time,
            num_clients=num_clients,
            num_waypoints=num_waypoints,
            total_distance=total_distance,
            battery_start=battery_start,
            battery_end=battery_end,
            battery_consumed=battery_start - battery_end,
            waypoints=waypoint_ids
        )
        
        self.route_logs.append(log)
    
    def generate_summary(self) -> MissionSummary:
        """Gera resumo da missão"""
        end_time = time.time()
        duration = end_time - self.mission_start
        
        # Contagens
        total_waypoints = len(self.waypoint_logs)
        planted = sum(1 for w in self.waypoint_logs if w.success)
        skipped = total_waypoints - planted
        
        # Distância total
        total_distance = sum(w.distance_from_previous for w in self.waypoint_logs)
        
        # Bateria
        total_battery = sum(w.battery_before - w.battery_after for w in self.waypoint_logs)
        
        # Recargas (número de rotas - 1, pois a primeira não conta)
        recharges = max(0, len(self.route_logs) - 1)
        
        return MissionSummary(
            mission_id=self.mission_id,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            start_time=self.mission_start,
            end_time=end_time,
            total_duration=duration,
            
            grid_size_x=self.config.get("grid_size_x", 0),
            grid_size_y=self.config.get("grid_size_y", 0),
            waypoint_spacing=self.config.get("waypoint_spacing", 0),
            
            total_waypoints=total_waypoints,
            planted_waypoints=planted,
            skipped_waypoints=skipped,
            success_rate=(planted / total_waypoints * 100) if total_waypoints > 0 else 0,
            
            num_routes=len(self.route_logs),
            total_distance=total_distance,
            avg_distance_per_route=total_distance / len(self.route_logs) if self.route_logs else 0,
            
            battery_capacity_mah=self.config.get("battery_capacity_mah", 5000),
            total_battery_consumed=total_battery,
            avg_consumption_per_plant=total_battery / planted if planted > 0 else 0,
            recharges=recharges,
            
            plants_per_minute=(planted / (duration / 60)) if duration > 0 else 0,
            meters_per_plant=total_distance / planted if planted > 0 else 0,
            efficiency_score=planted / (total_distance * duration) * 10000 if (total_distance > 0 and duration > 0) else 0,
            
            solver_time=self.config.get("solver_time", 0),
            solver_name=self.config.get("solver_name", "HGS-CVRP")
        )
    
    def set_summary(
        self,
        total_waypoints: int,
        successful_plants: int,
        failed_plants: int,
        total_routes: int,
        total_distance_m: float,
        total_time_s: float,
        solver_time_s: float = 0,
        field_config: Optional[Dict] = None,
        solver_name: str = None
    ):
        """
        Define dados do resumo manualmente.
        Chamado ao final da missão antes de salvar.
        """
        self.config.update({
            "total_waypoints": total_waypoints,
            "successful_plants": successful_plants,
            "failed_plants": failed_plants,
            "total_routes": total_routes,
            "total_distance_m": total_distance_m,
            "total_time_s": total_time_s,
            "solver_time": solver_time_s,
        })
        if solver_name:
            self.config["solver_name"] = solver_name
        
        if field_config:
            self.config["grid_size_x"] = field_config.get("width_m", 0)
            self.config["grid_size_y"] = field_config.get("height_m", 0)
            self.config["waypoint_spacing"] = field_config.get("plant_spacing_m", 0)
            self.config["line_spacing"] = field_config.get("line_spacing_m", 0)
            self.config["commodities"] = field_config.get("commodities", [])
    
    def save_to_csv(self) -> List[str]:
        """
        Salva dados em arquivos CSV.
        Retorna lista de arquivos gerados.
        """
        files = []
        
        # 1. Waypoints CSV
        if self.waypoint_logs:
            with open(self.waypoint_csv, 'w', newline='') as f:
                fieldnames = ['timestamp', 'waypoint_id', 'x', 'y', 'z', 'plant_type',
                             'route_id', 'client_id', 'time_to_reach', 'time_to_plant',
                             'distance_from_previous', 'battery_before', 'battery_after', 'success']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in self.waypoint_logs:
                    row = asdict(log)
                    # Remove campos extras se existirem
                    row = {k: v for k, v in row.items() if k in fieldnames}
                    writer.writerow(row)
            files.append(self.waypoint_csv)
        
        # 2. Routes CSV
        if self.route_logs:
            with open(self.routes_csv, 'w', newline='') as f:
                fieldnames = ['route_id', 'start_time', 'end_time', 'num_clients', 
                              'num_waypoints', 'total_distance', 'battery_start', 
                              'battery_end', 'battery_consumed', 'waypoints']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in self.route_logs:
                    row = asdict(log)
                    row['waypoints'] = str(row['waypoints'])
                    writer.writerow(row)
            files.append(self.routes_csv)
        
        return files
    
    def save_to_json(self) -> str:
        """
        Salva resumo completo em arquivo JSON.
        Retorna caminho do arquivo gerado.
        """
        summary = self.generate_summary()
        
        # Adiciona dados brutos ao JSON
        full_data = {
            "summary": asdict(summary),
            "config": self.config,
            "waypoints": [asdict(w) for w in self.waypoint_logs],
            "routes": [asdict(r) for r in self.route_logs]
        }
        
        with open(self.summary_json, 'w') as f:
            json.dump(full_data, f, indent=2, default=str)
        
        return self.summary_json
    
    def save_all(self):
        """Wrapper para salvar tudo e imprimir resumo"""
        csv_files = self.save_to_csv()
        json_file = self.save_to_json()
        summary = self.generate_summary()
        self._print_summary(summary)
        return csv_files, json_file
    
    def save(self):
        """Salva todos os dados em arquivos (método legado)"""
        rospy.loginfo(f"[LOGGER] Salvando dados...")
        
        # 1. Waypoints CSV
        if self.waypoint_logs:
            with open(self.waypoint_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.waypoint_logs[0]).keys())
                writer.writeheader()
                for log in self.waypoint_logs:
                    writer.writerow(asdict(log))
            rospy.loginfo(f"[LOGGER] ✓ {self.waypoint_csv}")
        
        # 2. Routes CSV
        if self.route_logs:
            with open(self.routes_csv, 'w', newline='') as f:
                fieldnames = ['route_id', 'start_time', 'end_time', 'num_clients', 
                              'num_waypoints', 'total_distance', 'battery_start', 
                              'battery_end', 'battery_consumed', 'waypoints']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in self.route_logs:
                    row = asdict(log)
                    row['waypoints'] = str(row['waypoints'])  # Lista como string
                    writer.writerow(row)
            rospy.loginfo(f"[LOGGER] ✓ {self.routes_csv}")
        
        # 3. Summary JSON
        summary = self.generate_summary()
        with open(self.summary_json, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        rospy.loginfo(f"[LOGGER] ✓ {self.summary_json}")
        
        # 4. Imprime resumo
        self._print_summary(summary)
    
    def _print_summary(self, summary: MissionSummary):
        """Imprime resumo formatado"""
        mins = int(summary.total_duration // 60)
        secs = int(summary.total_duration % 60)
        
        rospy.loginfo(f"\n{'='*70}")
        rospy.loginfo(f"{'RELATÓRIO DA MISSÃO':^70}")
        rospy.loginfo(f"{'='*70}")
        rospy.loginfo(f"ID: {summary.mission_id}")
        rospy.loginfo(f"Data: {summary.date}")
        rospy.loginfo(f"")
        rospy.loginfo(f"{'--- CONFIGURAÇÃO ---':^70}")
        rospy.loginfo(f"Talhão: {summary.grid_size_x}m x {summary.grid_size_y}m")
        rospy.loginfo(f"Espaçamento: {summary.waypoint_spacing}m")
        rospy.loginfo(f"Solver: {summary.solver_name} ({summary.solver_time:.2f}s)")
        rospy.loginfo(f"")
        rospy.loginfo(f"{'--- RESULTADOS ---':^70}")
        rospy.loginfo(f"Plantas: {summary.planted_waypoints}/{summary.total_waypoints} ({summary.success_rate:.1f}%)")
        rospy.loginfo(f"Rotas: {summary.num_routes} | Recargas: {summary.recharges}")
        rospy.loginfo(f"Distância: {summary.total_distance:.1f}m")
        rospy.loginfo(f"Tempo: {mins:02d}:{secs:02d}")
        rospy.loginfo(f"")
        rospy.loginfo(f"{'--- EFICIÊNCIA ---':^70}")
        rospy.loginfo(f"Velocidade: {summary.plants_per_minute:.2f} plantas/min")
        rospy.loginfo(f"Economia: {summary.meters_per_plant:.2f} m/planta")
        rospy.loginfo(f"Bateria: {summary.total_battery_consumed:.1f}% consumido")
        rospy.loginfo(f"Consumo médio: {summary.avg_consumption_per_plant:.3f}%/planta")
        rospy.loginfo(f"Score: {summary.efficiency_score:.4f}")
        rospy.loginfo(f"{'='*70}")
        rospy.loginfo(f"Logs salvos em: {self.output_dir}")
        rospy.loginfo(f"{'='*70}\n")


class BatteryModel:
    """
    Modelo de bateria realista para drones.
    
    Baseado em dados típicos de LiPo 4S/6S:
    - Capacidade: 5000-10000 mAh
    - Tensão: 14.8V (4S) ou 22.2V (6S)
    - Consumo hover: 10-20A
    - Consumo voo: 20-40A
    """
    
    def __init__(
        self,
        capacity_mah: float = 5000,
        voltage: float = 22.2,  # 6S
        consumption_hover_amps: float = 15.0,
        consumption_flight_amps: float = 25.0,
        consumption_plant_amps: float = 18.0,
        reserve_percent: float = 20.0  # Reserva de segurança
    ):
        self.capacity_mah = capacity_mah
        self._nominal_voltage = voltage
        self.consumption_hover = consumption_hover_amps
        self.consumption_flight = consumption_flight_amps
        self.consumption_plant = consumption_plant_amps
        self.reserve_percent = reserve_percent
        
        # Estado
        self.current_mah = capacity_mah
        self.total_consumed_mah = 0.0
        
        # Histórico
        self.consumption_log: List[Dict] = []
    
    @property
    def percentage(self) -> float:
        """Retorna percentual de bateria restante"""
        return (self.current_mah / self.capacity_mah) * 100
    
    def get_percent(self) -> float:
        """Alias para percentage - para compatibilidade"""
        return self.percentage
    
    def voltage(self) -> float:
        """Retorna tensão atual da bateria (varia com carga)"""
        # Modelo simplificado: 4.2V/cell full -> 3.3V/cell empty (6S)
        full_voltage = 25.2  # 6S @ 4.2V/cell
        empty_voltage = 19.8  # 6S @ 3.3V/cell
        pct = self.percentage / 100.0
        return empty_voltage + pct * (full_voltage - empty_voltage)
    
    def reset(self):
        """Reseta bateria para capacidade total"""
        self.current_mah = self.capacity_mah
        self.total_consumed_mah = 0.0
        self.consumption_log = []
    
    @property
    def usable_percentage(self) -> float:
        """Percentual utilizável (descontando reserva)"""
        return max(0, self.percentage - self.reserve_percent)
    
    @property
    def watt_hours(self) -> float:
        """Capacidade em Wh"""
        return (self.capacity_mah / 1000) * self._nominal_voltage
    
    def consume_hover(self, seconds: float) -> float:
        """Consome bateria em hover por N segundos"""
        mah = (self.consumption_hover * seconds) / 3600 * 1000
        return self._consume(mah, "hover", seconds)
    
    def consume_flight(self, time_or_distance: float, speed_ms: float = 2.0, is_time: bool = False) -> float:
        """
        Consome bateria voando.
        
        Args:
            time_or_distance: Pode ser tempo em segundos (is_time=True) ou distância em metros
            speed_ms: Velocidade em m/s (usado quando é distância)
            is_time: Se True, interpreta primeiro arg como tempo
        """
        if is_time:
            time_s = time_or_distance
            distance_m = time_s * speed_ms
        else:
            distance_m = time_or_distance
            time_s = distance_m / speed_ms if speed_ms > 0 else 0
        
        mah = (self.consumption_flight * time_s) / 3600 * 1000
        return self._consume(mah, "flight", time_s, distance_m)
    
    def consume_plant(self, seconds: float = 3.0) -> float:
        """Consome bateria durante plantio (desce, planta, sobe)"""
        mah = (self.consumption_plant * seconds) / 3600 * 1000
        return self._consume(mah, "plant", seconds)
    
    def _consume(self, mah: float, action: str, time_s: float, distance: float = 0) -> float:
        """Registra consumo interno"""
        self.current_mah -= mah
        self.total_consumed_mah += mah
        
        self.consumption_log.append({
            "action": action,
            "mah": mah,
            "time_s": time_s,
            "distance_m": distance,
            "remaining_pct": self.percentage
        })
        
        return mah
    
    def can_fly_distance(self, distance_m: float, speed_ms: float = 2.0) -> bool:
        """Verifica se tem bateria para voar distância + volta à base"""
        time_s = distance_m / speed_ms
        mah_needed = (self.consumption_flight * time_s) / 3600
        
        # Precisa ter margem para a reserva
        min_mah = (self.reserve_percent / 100) * self.capacity_mah
        return (self.current_mah - mah_needed) > min_mah
    
    def estimate_remaining_plants(self, avg_distance: float = 5.0, speed: float = 2.0) -> int:
        """Estima quantas plantas ainda pode fazer"""
        # Consumo médio por planta: voo + plantio
        time_flight = avg_distance / speed
        mah_flight = (self.consumption_flight * time_flight) / 3600
        mah_plant = (self.consumption_plant * 3.0) / 3600  # 3s por plantio
        mah_per_plant = mah_flight + mah_plant
        
        # Bateria disponível (descontando reserva)
        available = self.current_mah - (self.reserve_percent / 100) * self.capacity_mah
        
        if mah_per_plant > 0:
            return max(0, int(available / mah_per_plant))
        return 0
    
    def recharge(self):
        """Recarrega bateria (volta à base)"""
        old = self.percentage
        self.current_mah = self.capacity_mah
        self.consumption_log.append({
            "action": "recharge",
            "from_pct": old,
            "to_pct": 100.0
        })
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de consumo"""
        flight_mah = sum(l['mah'] for l in self.consumption_log if l['action'] == 'flight')
        hover_mah = sum(l['mah'] for l in self.consumption_log if l['action'] == 'hover')
        plant_mah = sum(l['mah'] for l in self.consumption_log if l['action'] == 'plant')
        recharges = sum(1 for l in self.consumption_log if l['action'] == 'recharge')
        
        return {
            "total_consumed_mah": self.total_consumed_mah,
            "total_consumed_pct": (self.total_consumed_mah / self.capacity_mah) * 100,
            "flight_mah": flight_mah,
            "hover_mah": hover_mah,
            "plant_mah": plant_mah,
            "recharges": recharges,
            "current_pct": self.percentage
        }


if __name__ == "__main__":
    # Teste do logger
    print("=== Teste do MissionLogger ===")
    
    logger = MissionLogger(mission_name="teste_local")
    logger.set_config({
        "grid_size_x": 50,
        "grid_size_y": 50,
        "waypoint_spacing": 5,
        "solver_name": "HGS-CVRP",
        "solver_time": 2.5,
        "battery_capacity_mah": 5000
    })
    
    # Simula waypoints
    for i in range(10):
        logger.log_waypoint(
            waypoint_id=i,
            x=i*5, y=i*3, z=3.0,
            plant_type=["erva", "arbusto", "arvore"][i % 3],
            route_id=1,
            client_id=i // 3,
            time_to_reach=2.5,
            time_to_plant=1.5,
            battery_before=100 - i*2,
            battery_after=100 - i*2 - 0.5,
            success=True
        )
    
    # Simula rota
    logger.log_route(
        route_id=1,
        start_time=0,
        end_time=40,
        num_clients=3,
        num_waypoints=10,
        total_distance=150,
        battery_start=100,
        battery_end=75,
        waypoint_ids=list(range(10))
    )
    
    print("\n=== Teste do BatteryModel ===")
    
    battery = BatteryModel(capacity_mah=5000)
    print(f"Bateria inicial: {battery.percentage:.1f}%")
    
    # Simula voo
    battery.consume_flight(100, speed_ms=2.0)
    print(f"Após 100m voo: {battery.percentage:.1f}%")
    
    battery.consume_plant(3.0)
    print(f"Após plantio: {battery.percentage:.1f}%")
    
    print(f"Plantas restantes estimadas: {battery.estimate_remaining_plants()}")
    
    stats = battery.get_stats()
    print(f"Consumo total: {stats['total_consumed_mah']:.2f} mAh")

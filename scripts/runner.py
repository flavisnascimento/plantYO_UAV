#!/usr/bin/env python3
"""Executa benchmarks comparativos entre solvers e gera relatórios."""

import os
import time
import json
from typing import List, Dict
import numpy as np

from base import BaseSolver
from result import SolverResult
from instance import BenchmarkInstance, GRID_GENERATOR_AVAILABLE


class BenchmarkRunner:
    """Executa benchmarks comparativos entre solvers."""

    def __init__(self, output_dir: str = None):
        self.solvers: List[BaseSolver] = []
        self.instances: List[BenchmarkInstance] = []
        self.results: List[SolverResult] = []

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.expanduser("~/plantyo_benchmarks")

        os.makedirs(self.output_dir, exist_ok=True)

    def add_solver(self, solver: BaseSolver):
        """Adiciona solver ao benchmark"""
        self.solvers.append(solver)
        print(f"[BENCHMARK] Solver adicionado: {solver.name}")
        print(f"            Referência: {solver.reference}")

    def add_instance(self, instance: BenchmarkInstance):
        """Adiciona instância de teste"""
        self.instances.append(instance)
        print(f"[BENCHMARK] Instância adicionada: {instance.name}")
        print(f"            {instance.description}")

    def add_standard_instances(self):
        """Adiciona instâncias padrão para testes (grid simples)"""
        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=15.0, grid_size_y=15.0, spacing=5.0,
            base_x=7.5, base_y=0.0, margin=2.5,
            capacity=225, autonomy=1500.0,
            name="small_10x10"
        ))

        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=50.0, grid_size_y=50.0, spacing=5.0,
            base_x=25.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=1500.0,
            name="medium_50x50"
        ))

        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=100.0, grid_size_y=100.0, spacing=5.0,
            base_x=50.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=1500.0,
            name="large_100x100"
        ))

        self.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=200.0, grid_size_y=200.0, spacing=5.0,
            base_x=100.0, base_y=0.0, margin=2.5,
            capacity=225, autonomy=1500.0,
            name="xlarge_200x200"
        ))

    def add_csdvrp_instances(self):
        """
        Adiciona instâncias C-SDVRP reais usando GridGenerator, com a
        transformação de clientes virtuais (2024) e clientes virtuais por commodity.
        """
        if not GRID_GENERATOR_AVAILABLE:
            print("[WARNING] GridGenerator não disponível. Usando instâncias simples.")
            self.add_standard_instances()
            return

        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=25.0, grid_size_y=25.0,
            waypoint_spacing=2.0, line_spacing=3.0,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=1500.0,
            name="csdvrp_25x25"
        ))

        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=50.0, grid_size_y=50.0,
            waypoint_spacing=2.0, line_spacing=3.0,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=1500.0,
            name="csdvrp_50x50"
        ))

        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=75.0, grid_size_y=75.0,
            waypoint_spacing=2.0, line_spacing=3.0,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=1500.0,
            name="csdvrp_75x75"
        ))

        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=100.0, grid_size_y=100.0,
            waypoint_spacing=2.0, line_spacing=3.0,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=1500.0,
            name="csdvrp_100x100"
        ))

        self.add_instance(BenchmarkInstance.from_grid_generator(
            grid_size_x=150.0, grid_size_y=150.0,
            waypoint_spacing=2.0, line_spacing=3.0,
            seeds_per_waypoint=15,
            commodity_capacity=(100, 100, 100),
            autonomy=1500.0,
            name="csdvrp_150x150"
        ))

    def run(self,
            time_limit: float = 30.0,
            num_runs: int = 1,
            verbose: bool = True) -> Dict:
        """Executa benchmark completo."""
        if verbose:
            print("\n" + "="*70)
            print("BENCHMARK DE SOLVERS PARA C-SDVRP")
            print("="*70)
            print(f"Solvers: {[s.name for s in self.solvers]}")
            print(f"Instâncias: {[i.name for i in self.instances]}")
            print(f"Time limit: {time_limit}s | Runs: {num_runs}")
            print("="*70 + "\n")

        all_results = []
        all_individual_results = []

        for instance in self.instances:
            if verbose:
                print(f"\n--- Instância: {instance.name} ---")
                print(f"    Waypoints: {instance.num_waypoints}")
                print(f"    Capacidade: {instance.capacity}")
                print(f"    Autonomia: {instance.autonomy}m")

            for solver in self.solvers:
                run_results = []

                for run in range(num_runs):
                    if verbose:
                        print(f"    [{solver.name}] Run {run+1}/{num_runs}...", end=" ")

                    result = solver.solve(
                        distance_matrix=instance.distance_matrix,
                        demands=instance.demands,
                        capacity=instance.capacity,
                        autonomy=instance.autonomy,
                        time_limit=time_limit,
                        instance_name=instance.name
                    )

                    validation = solver.validate_solution(result, instance)
                    result.feasible = validation['valid']
                    result.capacity_violations = len(validation['capacity_violations'])
                    result.autonomy_violations = len(validation['autonomy_violations'])

                    if instance.optimal_known:
                        result.metadata['optimal'] = instance.optimal_known

                    run_results.append(result)
                    all_individual_results.append(result)

                    if verbose:
                        print(f"Dist: {result.total_distance:.1f}m | "
                              f"Rotas: {result.num_routes} | "
                              f"Tempo: {result.computation_time:.3f}s")

                if num_runs > 1:
                    avg_result = self._average_results(run_results)
                    all_results.append(avg_result)
                else:
                    all_results.append(run_results[0])

        self.results = all_results
        self.individual_results = all_individual_results

        report = self._generate_report()

        return report

    def _average_results(self, results: List[SolverResult]) -> SolverResult:
        """Calcula média de múltiplas execuções com estatísticas completas"""
        distances = [r.total_distance for r in results]
        times = [r.computation_time for r in results]
        routes = [r.num_routes for r in results]

        avg_distance = np.mean(distances)
        avg_time = np.mean(times)
        avg_routes = np.mean(routes)

        return SolverResult(
            solver_name=results[0].solver_name,
            instance_name=results[0].instance_name,
            routes=results[0].routes,
            total_distance=avg_distance,
            num_routes=int(np.round(avg_routes)),
            computation_time=avg_time,
            feasible=all(r.feasible for r in results),
            metadata={
                'num_runs': len(results),
                'distance_mean': float(avg_distance),
                'distance_std': float(np.std(distances)),
                'distance_min': float(np.min(distances)),
                'distance_max': float(np.max(distances)),
                'time_mean': float(avg_time),
                'time_std': float(np.std(times)),
                'time_min': float(np.min(times)),
                'time_max': float(np.max(times)),
                'routes_mean': float(avg_routes),
                'routes_std': float(np.std(routes)),
                'routes_min': float(np.min(routes)),
                'routes_max': float(np.max(routes)),
            }
        )

    def _generate_report(self) -> Dict:
        """Gera relatório comparativo com dados brutos e agregados"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'solvers': [s.name for s in self.solvers],
            'instances': [i.name for i in self.instances],
            'results': [r.to_dict() for r in self.results],
            'summary': {},
            'summary_by_instance': {},
            'individual_runs': [r.to_dict() for r in self.individual_results] if hasattr(self, 'individual_results') else []
        }

        for solver in self.solvers:
            solver_results = [r for r in self.results if r.solver_name == solver.name]

            if solver_results:
                report['summary'][solver.name] = {
                    'avg_distance': np.mean([r.total_distance for r in solver_results]),
                    'std_distance': np.mean([r.metadata.get('distance_std', 0) for r in solver_results]),
                    'avg_time': np.mean([r.computation_time for r in solver_results]),
                    'std_time': np.mean([r.metadata.get('time_std', 0) for r in solver_results]),
                    'avg_routes': np.mean([r.num_routes for r in solver_results]),
                    'feasible_rate': sum(1 for r in solver_results if r.feasible) / len(solver_results)
                }

        for instance in self.instances:
            instance_results = [r for r in self.results if r.instance_name == instance.name]
            report['summary_by_instance'][instance.name] = {}

            for result in instance_results:
                solver_name = result.solver_name
                report['summary_by_instance'][instance.name][solver_name] = {
                    'distance_mean': float(result.total_distance),
                    'distance_std': float(result.metadata.get('distance_std', 0)),
                    'distance_cv': float(result.metadata.get('distance_std', 0) / result.total_distance) if result.total_distance > 0 else 0,
                    'distance_min': float(result.metadata.get('distance_min', 0)),
                    'distance_max': float(result.metadata.get('distance_max', 0)),
                    'time_mean': float(result.computation_time),
                    'time_std': float(result.metadata.get('time_std', 0)),
                    'num_runs': result.metadata.get('num_runs', 1)
                }

        return report

    def print_comparison_table(self):
        """Imprime tabela comparativa formatada"""
        print("\n" + "="*90)
        print("TABELA COMPARATIVA")
        print("="*90)

        header = f"{'Instância':<20}"
        for solver in self.solvers:
            header += f" | {solver.name:>15}"
        print(header)
        print("-"*90)

        print("\nDISTÂNCIA TOTAL (metros):")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            best_dist = float('inf')

            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result and result.total_distance < best_dist:
                    best_dist = result.total_distance

            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    if abs(result.total_distance - best_dist) < 0.01:
                        row += f" | {result.total_distance:>13.1f}*"
                    else:
                        gap = ((result.total_distance - best_dist) / best_dist) * 100
                        row += f" | {result.total_distance:>10.1f}(+{gap:.0f}%)"
                else:
                    row += f" | {'N/A':>15}"
            print(row)

        print("\nTEMPO DE COMPUTAÇÃO (segundos):")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" | {result.computation_time:>15.3f}"
                else:
                    row += f" | {'N/A':>15}"
            print(row)

        print("\nNÚMERO DE ROTAS:")
        for instance in self.instances:
            row = f"{instance.name:<20}"
            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" | {result.num_routes:>15}"
                else:
                    row += f" | {'N/A':>15}"
            print(row)

        print("\n" + "="*90)
        print("* = Melhor resultado para a instância")
        print("="*90)

    def save_results(self, filename: str = None):
        """Salva resultados em JSON"""
        if filename is None:
            filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)

        report = self._generate_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n[BENCHMARK] Resultados salvos em: {filepath}")
        return filepath

    def export_latex_table(self, filename: str = None) -> str:
        """Exporta tabela em formato LaTeX para dissertação"""
        if filename is None:
            filename = f"benchmark_table_{time.strftime('%Y%m%d')}.tex"

        filepath = os.path.join(self.output_dir, filename)

        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Comparação de Solvers para C-SDVRP}")
        latex.append(r"\label{tab:solver_comparison}")

        cols = "l" + "r" * len(self.solvers)
        latex.append(r"\begin{tabular}{" + cols + "}")
        latex.append(r"\toprule")

        header = "Instância"
        for solver in self.solvers:
            header += f" & {solver.name}"
        header += r" \\"
        latex.append(header)
        latex.append(r"\midrule")

        latex.append(r"\multicolumn{" + str(len(self.solvers)+1) + r"}{c}{\textbf{Distância Total (m)}} \\")
        latex.append(r"\midrule")

        for instance in self.instances:
            row = instance.name.replace("_", r"\_")
            best_dist = float('inf')

            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result and result.total_distance < best_dist:
                    best_dist = result.total_distance

            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    if abs(result.total_distance - best_dist) < 0.01:
                        row += f" & \\textbf{{{result.total_distance:.1f}}}"
                    else:
                        row += f" & {result.total_distance:.1f}"
                else:
                    row += " & --"
            row += r" \\"
            latex.append(row)

        latex.append(r"\midrule")

        latex.append(r"\multicolumn{" + str(len(self.solvers)+1) + r"}{c}{\textbf{Tempo de Computação (s)}} \\")
        latex.append(r"\midrule")

        for instance in self.instances:
            row = instance.name.replace("_", r"\_")
            for solver in self.solvers:
                result = next((r for r in self.results
                               if r.solver_name == solver.name and r.instance_name == instance.name), None)
                if result:
                    row += f" & {result.computation_time:.3f}"
                else:
                    row += " & --"
            row += r" \\"
            latex.append(row)

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        latex_content = "\n".join(latex)

        with open(filepath, 'w') as f:
            f.write(latex_content)

        print(f"[BENCHMARK] Tabela LaTeX salva em: {filepath}")
        return latex_content

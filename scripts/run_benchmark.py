#!/usr/bin/env python3
"""
Script de execução rápida do benchmark de solvers

Uso:
    python3 run_benchmark.py                    # Benchmark completo
    python3 run_benchmark.py --quick            # Benchmark rápido (1 instância)
    python3 run_benchmark.py --instance medium  # Instância específica
    python3 run_benchmark.py --runs 5           # 5 execuções para média

Para dissertação, recomenda-se:
    python3 run_benchmark.py --runs 5
"""

import sys
import argparse

# Adiciona path dos scripts
sys.path.insert(0, '/home/flanascimento/rma2025_ws/src/mrs_computer_vision_examples/python/plantYO_UAV/scripts')

from solver_benchmark import (
    BenchmarkRunner, 
    BenchmarkInstance,
    HGSSolverBenchmark, 
    AHASolverBenchmark, 
    NearestNeighborSolver,
    JoaoSolverBenchmark
)


def run_quick_test():
    """Teste rápido com instância pequena"""
    print("\n🚀 BENCHMARK RÁPIDO (teste)")
    print("-" * 50)
    
    runner = BenchmarkRunner()
    
    # Só baseline e HGS para teste rápido
    runner.add_solver(NearestNeighborSolver())
    runner.add_solver(HGSSolverBenchmark())
    
    # Instância pequena
    runner.add_instance(BenchmarkInstance.from_grid(
        grid_size_x=20.0, grid_size_y=20.0, spacing=5.0,
        base_x=10.0, base_y=0.0, margin=2.5,
        capacity=225, autonomy=2025.0,
        name="quick_test"
    ))
    
    runner.run(time_limit=5.0, num_runs=1, verbose=True)
    runner.print_comparison_table()
    
    return runner


def run_full_benchmark(num_runs: int = 1, time_limit: float = 30.0):
    """Benchmark completo para dissertação"""
    print("\n📊 BENCHMARK COMPLETO")
    print("-" * 50)
    
    runner = BenchmarkRunner()
    
    # Todos os solvers disponíveis
    runner.add_solver(NearestNeighborSolver())
    runner.add_solver(AHASolverBenchmark(population_size=30, max_iterations=100))
    runner.add_solver(HGSSolverBenchmark())
    # runner.add_solver(JoaoSolverBenchmark())  # Descomentar quando disponível
    
    # Instâncias padrão
    runner.add_standard_instances()
    
    # Executa
    runner.run(time_limit=time_limit, num_runs=num_runs, verbose=True)
    
    # Resultados
    runner.print_comparison_table()
    runner.save_results()
    runner.export_latex_table()
    
    return runner


def run_specific_instance(instance_name: str, num_runs: int = 1):
    """Benchmark para instância específica"""
    print(f"\n🎯 BENCHMARK INSTÂNCIA: {instance_name}")
    print("-" * 50)
    
    runner = BenchmarkRunner()
    
    runner.add_solver(NearestNeighborSolver())
    runner.add_solver(AHASolverBenchmark(population_size=30, max_iterations=100))
    runner.add_solver(HGSSolverBenchmark())
    
    # Cria instância baseada no nome
    instances = {
        'small': (15.0, 15.0, 5.0),
        'medium': (50.0, 50.0, 5.0),
        'large': (100.0, 100.0, 5.0),
        'xlarge': (200.0, 200.0, 5.0),
    }
    
    if instance_name in instances:
        x, y, sp = instances[instance_name]
        runner.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=x, grid_size_y=y, spacing=sp,
            base_x=x/2, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name=instance_name
        ))
    else:
        print(f"❌ Instância desconhecida: {instance_name}")
        print(f"   Opções: {list(instances.keys())}")
        return None
    
    runner.run(time_limit=30.0, num_runs=num_runs, verbose=True)
    runner.print_comparison_table()
    
    return runner


def compare_hgs_vs_aha():
    """Comparação focada HGS vs AHA para dissertação"""
    print("\n⚔️  COMPARAÇÃO HGS vs AHA")
    print("-" * 50)
    
    runner = BenchmarkRunner()
    
    # Só os dois solvers principais
    runner.add_solver(AHASolverBenchmark(population_size=50, max_iterations=200))
    runner.add_solver(HGSSolverBenchmark())
    
    # Várias instâncias para análise
    sizes = [
        (25, "25x25"),
        (50, "50x50"),
        (75, "75x75"),
        (100, "100x100"),
        (150, "150x150"),
    ]
    
    for size, name in sizes:
        runner.add_instance(BenchmarkInstance.from_grid(
            grid_size_x=float(size), grid_size_y=float(size), spacing=5.0,
            base_x=float(size)/2, base_y=0.0, margin=2.5,
            capacity=225, autonomy=2025.0,
            name=name
        ))
    
    runner.run(time_limit=30.0, num_runs=3, verbose=True)
    runner.print_comparison_table()
    runner.save_results("hgs_vs_aha.json")
    runner.export_latex_table("hgs_vs_aha.tex")
    
    return runner


def main():
    parser = argparse.ArgumentParser(description='Benchmark de Solvers para C-SDVRP')
    parser.add_argument('--quick', action='store_true', help='Teste rápido')
    parser.add_argument('--instance', type=str, help='Instância específica (small/medium/large/xlarge)')
    parser.add_argument('--runs', type=int, default=1, help='Número de execuções')
    parser.add_argument('--time', type=float, default=30.0, help='Tempo limite por solver')
    parser.add_argument('--compare', action='store_true', help='Comparação HGS vs AHA')
    
    args = parser.parse_args()
    
    if args.quick:
        runner = run_quick_test()
    elif args.instance:
        runner = run_specific_instance(args.instance, args.runs)
    elif args.compare:
        runner = compare_hgs_vs_aha()
    else:
        runner = run_full_benchmark(args.runs, args.time)
    
    print("\n✅ Benchmark concluído!")
    print(f"📁 Resultados em: ~/plantyo_benchmarks/")
    
    return runner


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ponto de entrada do benchmark experimental (antigo main() de solver_benchmark.py).

Uso:
    python -m solver_benchmark.run_benchmark
ou:
    python run_benchmark.py     (de dentro da pasta scripts/)
"""

import os
import sys

# Permite rodar tanto como módulo do pacote quanto como script solto.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_HERE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

try:
    from solver_benchmark import (
        BenchmarkRunner,
        NearestNeighborSolver,
        AHASolverBenchmark,
        HGSSolverBenchmark,
    )
except ImportError:
    # Execução direta de dentro do pacote
    from solver_benchmark import (
        BenchmarkRunner,
        NearestNeighborSolver,
        AHASolverBenchmark,
        HGSSolverBenchmark,
    )

# Solvers TSP do João Rafael (opcional, módulo irmão joao_tsp_solver.py)
try:
    from joao_tsp_solver import JoaoTSPSolver
    JOAO_SOLVER_AVAILABLE = True
except ImportError:
    JOAO_SOLVER_AVAILABLE = False
    print("[WARNING] joao_tsp_solver não disponível. Solver João será pulado.")


def main():
    """Executa a avaliação comparativa completa dos algoritmos no C-SDVRP."""
    print("\n" + "="*70)
    print("BENCHMARK DE SOLVERS PARA PLANTIO COM UAV")
    print("Dissertação de Mestrado - C-SDVRP")
    print("="*70)

    runner = BenchmarkRunner()

    # Bateria de solvers
    runner.add_solver(NearestNeighborSolver())     # Baseline de referência
    runner.add_solver(AHASolverBenchmark(population_size=30, max_iterations=100))  # D-AHA proposto
    runner.add_solver(HGSSolverBenchmark())

    # Solvers TSP com Split Ótimo (João Rafael)
    if JOAO_SOLVER_AVAILABLE:
        runner.add_solver(JoaoTSPSolver(split_strategy='optimal', use_2opt_refinement=True))
        runner.add_solver(JoaoTSPSolver(split_strategy='greedy', use_2opt_refinement=True))
    else:
        print("[WARNING] Solvers do João não disponíveis - pulando")

    # Instâncias experimentais via Transformação de Clientes Virtuais
    runner.add_csdvrp_instances()

    report = runner.run(
        time_limit=30.0,
        num_runs=10,
        verbose=True
    )

    runner.print_comparison_table()

    runner.save_results()
    runner.export_latex_table()

    print("\n[BENCHMARK] Experimento concluído com sucesso!")
    print("Resultados salvos para análise estatística e publicação acadêmica.")
    return runner


if __name__ == "__main__":
    runner = main()

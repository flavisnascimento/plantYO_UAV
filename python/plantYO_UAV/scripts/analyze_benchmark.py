#!/usr/bin/env python3
"""
Análise completa de resultados de benchmark com múltiplas runs.

Este script lê arquivos JSON do benchmark e fornece análises estatísticas detalhadas,
incluindo:
  - Estatísticas descritivas (média, desvio padrão, min, max)
  - Análise de consistência (variância entre runs)
  - Comparações pareadas entre solvers
  - Detecção de anomalias

Uso:
    python3 analyze_benchmark.py /path/to/benchmark.json
"""

import json
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path


class BenchmarkAnalyzer:
    """Analisa resultados de benchmark com suporte a múltiplas runs"""
    
    def __init__(self, json_file):
        """Carrega dados do JSON"""
        with open(json_file) as f:
            self.data = json.load(f)
        
        self.timestamp = self.data.get('timestamp', 'Unknown')
        self.solvers = self.data.get('solvers', [])
        self.instances = self.data.get('instances', [])
        self.aggregated = self.data.get('results', [])
        self.individual = self.data.get('individual_runs', [])
        
        # Organiza dados por (solver, instance)
        self._organize_data()
    
    def _organize_data(self):
        """Organiza dados individuais por (solver, instance)"""
        self.runs_by_pair = defaultdict(list)
        
        for result in self.individual:
            # Suporta tanto 'solver'/'instance' quanto 'solver_name'/'instance_name'
            solver = result.get('solver_name') or result.get('solver')
            instance = result.get('instance_name') or result.get('instance')
            key = (solver, instance)
            self.runs_by_pair[key].append(result)
    
    def print_summary(self):
        """Imprime resumo executivo"""
        print("\n" + "="*80)
        print("ANÁLISE DE BENCHMARK - MÚLTIPLAS RUNS")
        print("="*80)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Solvers: {len(self.solvers)}")
        print(f"Instâncias: {len(self.instances)}")
        print(f"Total de runs individuais: {len(self.individual)}")
        print(f"Total de pares (solver, instance): {len(self.runs_by_pair)}")
        
        if self.individual:
            num_runs = len(next(iter(self.runs_by_pair.values()))) if self.runs_by_pair else 0
            print(f"Execuções por par: {num_runs}")
    
    def print_statistics(self):
        """Imprime estatísticas detalhadas por solver e instância"""
        print("\n" + "="*80)
        print("ESTATÍSTICAS DETALHADAS POR INSTÂNCIA")
        print("="*80)
        
        # Por instância
        for instance in self.instances:
            print(f"\n{'─'*80}")
            print(f"INSTÂNCIA: {instance}")
            print(f"{'─'*80}")
            print(f"{'Solver':<25} {'Distância':<45} {'Tempo (s)':<12}")
            print(f"{'':25} {'Média ± Std | Min - Max':<45} {'Média ± Std':<12}")
            print("-"*80)
            
            for solver in self.solvers:
                key = (solver, instance)
                runs = self.runs_by_pair.get(key, [])
                
                if not runs:
                    print(f"{solver:<25} {'N/A':<45} {'N/A':<12}")
                    continue
                
                distances = [r['distance'] for r in runs]
                times = [r['time_s'] for r in runs]
                
                dist_mean = np.mean(distances)
                dist_std = np.std(distances)
                dist_min = np.min(distances)
                dist_max = np.max(distances)
                
                time_mean = np.mean(times)
                time_std = np.std(times)
                
                dist_str = f"{dist_mean:.1f} ± {dist_std:.1f} | {dist_min:.1f} - {dist_max:.1f}"
                time_str = f"{time_mean:.3f} ± {time_std:.3f}"
                
                print(f"{solver:<25} {dist_str:<45} {time_str:<12}")
    
    def print_consistency_analysis(self):
        """Analisa consistência (variância) dos solvers"""
        print("\n" + "="*80)
        print("ANÁLISE DE CONSISTÊNCIA")
        print("="*80)
        print("\nCoeficiente de Variação (CV = std/mean) - quanto menor, mais consistente")
        print("  CV < 0.05: Muito consistente ✅")
        print("  CV 0.05-0.10: Consistente")
        print("  CV 0.10-0.20: Moderadamente variável")
        print("  CV > 0.20: Altamente variável ❌")
        
        print(f"\n{'Solver':<25} {'Instância':<20} {'CV Distância':<15} {'CV Tempo':<15}")
        print("-"*75)
        
        cv_data = defaultdict(list)
        
        for solver in self.solvers:
            for instance in self.instances:
                key = (solver, instance)
                runs = self.runs_by_pair.get(key, [])
                
                if not runs or len(runs) < 2:
                    continue
                
                distances = [r['distance'] for r in runs]
                times = [r['time_s'] for r in runs]
                
                cv_dist = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
                cv_time = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
                
                cv_data[solver].append(cv_dist)
                
                # Flag de alerta
                dist_flag = "❌" if cv_dist > 0.20 else ("⚠️" if cv_dist > 0.10 else "✅")
                time_flag = "❌" if cv_time > 0.20 else ("⚠️" if cv_time > 0.10 else "✅")
                
                print(f"{solver:<25} {instance:<20} {cv_dist:.4f} {dist_flag:<6} {cv_time:.4f} {time_flag:<6}")
        
        # Resumo por solver
        print("\n" + "─"*75)
        print("RESUMO DE CONSISTÊNCIA POR SOLVER")
        print("-"*75)
        
        for solver in self.solvers:
            if cv_data[solver]:
                avg_cv = np.mean(cv_data[solver])
                flag = "✅ CONSISTENTE" if avg_cv < 0.10 else ("⚠️ MODERADO" if avg_cv < 0.20 else "❌ VARIÁVEL")
                print(f"{solver:<25} CV médio: {avg_cv:.4f} ({flag})")
    
    def print_degradation_analysis(self):
        """Analisa degradação em instâncias maiores"""
        print("\n" + "="*80)
        print("ANÁLISE DE DEGRADAÇÃO EM INSTÂNCIAS MAIORES")
        print("="*80)
        
        # Ordena instâncias por tamanho (assume padrão csdvrp_XXxXX)
        sorted_instances = sorted(self.instances, key=self._extract_size)
        
        print(f"\nOrder de instâncias (por tamanho): {sorted_instances}\n")
        
        for solver in self.solvers:
            print(f"\n{solver}:")
            print("─" * 70)
            print(f"{'Instância':<20} {'Distância Média':<20} {'Crescimento':<20}")
            print("-" * 70)
            
            prev_dist = None
            growth_rates = []
            
            for instance in sorted_instances:
                key = (solver, instance)
                runs = self.runs_by_pair.get(key, [])
                
                if not runs:
                    print(f"{instance:<20} {'N/A':<20}")
                    continue
                
                distances = [r['distance'] for r in runs]
                mean_dist = np.mean(distances)
                
                if prev_dist is not None:
                    growth = ((mean_dist - prev_dist) / prev_dist) * 100
                    growth_rates.append(growth)
                    growth_str = f"+{growth:.1f}%" if growth > 0 else f"{growth:.1f}%"
                    
                    # Alerta se degradação > 50%
                    flag = "🔴" if growth > 50 else ("🟡" if growth > 20 else "🟢")
                    print(f"{instance:<20} {mean_dist:>18.1f} {growth_str:>18} {flag}")
                else:
                    print(f"{instance:<20} {mean_dist:>18.1f} {'(baseline)':<18}")
                
                prev_dist = mean_dist
            
            if growth_rates:
                avg_growth = np.mean(growth_rates)
                print("-" * 70)
                print(f"{'Taxa média de crescimento':<20} {avg_growth:>18.1f}%")
    
    def print_aha_vs_hgs_comparison(self):
        """Comparação específica AHA vs HGS"""
        print("\n" + "="*80)
        print("COMPARAÇÃO ESPECÍFICA: AHA vs HGS")
        print("="*80)
        
        sorted_instances = sorted(self.instances, key=self._extract_size)
        
        print(f"\n{'Instância':<20} {'AHA':<20} {'HGS':<20} {'Gap AHA->HGS':<20}")
        print("-"*80)
        
        gaps = []
        for instance in sorted_instances:
            aha_runs = self.runs_by_pair.get(('AHA-CVRP', instance), [])
            hgs_runs = self.runs_by_pair.get(('HGS-CVRP', instance), [])
            
            if not aha_runs or not hgs_runs:
                print(f"{instance:<20} {'N/A':<20}")
                continue
            
            aha_mean = np.mean([r['distance'] for r in aha_runs])
            hgs_mean = np.mean([r['distance'] for r in hgs_runs])
            
            gap = ((aha_mean - hgs_mean) / hgs_mean) * 100
            gaps.append(gap)
            
            # Alerta visual
            flag = "🔴" if gap > 30 else ("🟡" if gap > 10 else "🟢")
            print(f"{instance:<20} {aha_mean:>18.1f} {hgs_mean:>18.1f} {gap:>18.1f}% {flag}")
        
        if gaps:
            print("-"*80)
            avg_gap = np.mean(gaps)
            print(f"{'Gap médio':<20} {avg_gap:>18.1f}%")
            
            # Conclusão
            print("\n📊 CONCLUSÃO:")
            if avg_gap > 30:
                print("   🔴 AHA está SIGNIFICATIVAMENTE PIOR que HGS em média")
            elif avg_gap > 10:
                print("   🟡 AHA está MODERADAMENTE PIOR que HGS")
            else:
                print("   🟢 AHA está COMPETITIVO com HGS")
    
    def _extract_size(self, instance_name):
        """Extrai tamanho da instância para ordenação"""
        try:
            parts = instance_name.split('_')[-1]  # 'csdvrp_25x25' -> '25x25'
            size = int(parts.split('x')[0])
            return size
        except:
            return 0
    
    def export_detailed_csv(self, output_file=None):
        """Exporta dados detalhados em CSV para análise em Excel/R"""
        if output_file is None:
            output_file = f"benchmark_detailed_{self.timestamp.replace(' ', '_').replace(':', '')}.csv"
        
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Solver', 'Instance', 'Run', 'Distance', 'Routes', 'Time(s)', 'Feasible'])
            
            for result in self.individual:
                writer.writerow([
                    result['solver_name'],
                    result['instance_name'],
                    result.get('run_number', ''),
                    result['distance'],
                    result['num_routes'],
                    result['time_s'],
                    result['feasible']
                ])
        
        print(f"\n📊 CSV exportado: {output_file}")
    
    def print_stability_by_instance(self):
        """Análise de estabilidade específica por instância - desvio padrão absoluto"""
        print("\n" + "="*80)
        print("ESTABILIDADE POR INSTÂNCIA - DESVIO PADRÃO ABSOLUTO")
        print("="*80)
        print("\nMostra o desvio padrão de cada solver em cada instância")
        print("Desvios altos indicam comportamento inconsistente/colapso\n")
        
        sorted_instances = sorted(self.instances, key=self._extract_size)
        
        for instance in sorted_instances:
            print(f"\n{instance} (tamanho: ~{self._extract_size(instance)}x{self._extract_size(instance)}m)")
            print("─" * 80)
            print(f"{'Solver':<25} {'Desvio (std)':<15} {'Coef. Var (CV)':<15} {'Status':<30}")
            print("-" * 85)
            
            for solver in self.solvers:
                key = (solver, instance)
                runs = self.runs_by_pair.get(key, [])
                
                if not runs or len(runs) < 2:
                    print(f"{solver:<25} {'N/A':<15}")
                    continue
                
                distances = [r['distance'] for r in runs]
                dist_std = np.std(distances)
                dist_mean = np.mean(distances)
                cv = dist_std / dist_mean if dist_mean > 0 else 0
                
                # Classificação de estabilidade
                if cv < 0.01:
                    status = "🟢 Ultra-Estável (CV < 0.01)"
                elif cv < 0.05:
                    status = "🟢 Muito Estável (CV < 0.05)"
                elif cv < 0.10:
                    status = "🟡 Estável (CV < 0.10)"
                elif cv < 0.20:
                    status = "🟠 Moderado (CV < 0.20)"
                else:
                    status = "🔴 Instável (CV ≥ 0.20)"
                
                print(f"{solver:<25} {dist_std:>13.2f} {cv:>14.4f} {status:<30}")
        
        # Análise de colapso do AHA: mostra crescimento de variabilidade
        print("\n" + "="*80)
        print("ANÁLISE DE COLAPSO DO AHA - VARIABILIDADE POR TAMANHO")
        print("="*80)
        
        aha_std_by_instance = {}
        for instance in sorted_instances:
            key = ('AHA-CVRP', instance)
            runs = self.runs_by_pair.get(key, [])
            if runs:
                distances = [r['distance'] for r in runs]
                aha_std_by_instance[instance] = {
                    'std': np.std(distances),
                    'mean': np.mean(distances),
                    'cv': np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
                }
        
        if aha_std_by_instance:
            print("\nProgresso da variabilidade do AHA conforme cresce a instância:")
            print("-" * 90)
            print(f"{'Instância':<18} {'Std Dev':<12} {'Coef. Var':<12} {'Visualização':<50}")
            print("-" * 90)
            
            max_std = max(d['std'] for d in aha_std_by_instance.values())
            
            for instance, data in aha_std_by_instance.items():
                std = data['std']
                cv = data['cv']
                bar_length = int((std / max_std) * 40) if max_std > 0 else 0
                bar = "█" * bar_length + "░" * (40 - bar_length)
                
                if cv < 0.01:
                    status = "✅"
                elif cv < 0.05:
                    status = "✅"
                elif cv < 0.10:
                    status = "🟡"
                elif cv < 0.20:
                    status = "🟠"
                else:
                    status = "🔴"
                
                print(f"{instance:<18} {std:>10.2f} {cv:>11.4f} {status} {bar}")
            
            # Conclusão sobre estabilidade do AHA
            std_values = list(aha_std_by_instance.values())
            if len(std_values) > 1:
                first_std = std_values[0]['std']
                last_std = std_values[-1]['std']
                growth_factor = last_std / first_std if first_std > 0 else 1
                
                print("\n" + "─" * 90)
                if growth_factor > 4:
                    print(f"🔴 CONCLUSÃO: AHA apresenta COLAPSO em instâncias grandes!")
                    print(f"   Variabilidade cresce {growth_factor:.1f}x de {sorted_instances[0]} para {sorted_instances[-1]}")
                elif growth_factor > 2:
                    print(f"🟠 CONCLUSÃO: AHA mostra DEGRADAÇÃO significativa em instâncias maiores")
                    print(f"   Variabilidade cresce {growth_factor:.1f}x")
                elif growth_factor > 1.2:
                    print(f"🟡 CONCLUSÃO: AHA mostra DEGRADAÇÃO gradual em instâncias maiores")
                    print(f"   Variabilidade cresce {growth_factor:.1f}x")
                else:
                    print(f"🟢 CONCLUSÃO: AHA mantém ESTABILIDADE em todas as instâncias")
                    print(f"   Variabilidade cresce apenas {growth_factor:.1f}x")
                print("─" * 90)



def main():
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_benchmark.py <arquivo_json>")
        print("\nExemplo:")
        print("  python3 analyze_benchmark.py /home/plantyo_benchmarks/benchmark_20260214.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"Erro: Arquivo não encontrado: {json_file}")
        sys.exit(1)
    
    analyzer = BenchmarkAnalyzer(json_file)
    
    analyzer.print_summary()
    analyzer.print_statistics()
    analyzer.print_consistency_analysis()
    analyzer.print_stability_by_instance()  # ← NOVO: Análise por instância
    analyzer.print_degradation_analysis()
    analyzer.print_aha_vs_hgs_comparison()
    
    print("\n" + "="*80)
    print("FIM DA ANÁLISE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

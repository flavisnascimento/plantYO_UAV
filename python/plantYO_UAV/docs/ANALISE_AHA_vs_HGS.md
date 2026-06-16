# Análise Comparativa: AHA vs HGS

## 📋 RESUMO EXECUTIVO

| Aspecto | AHA (aha_optimizer.py) | HGS (hgs_solver.py + benchmark) |
|---------|------------------------|--------------------------------|
| **Problema Resolvido** | TSP com restrições agronômicas | CVRP com capacidade e autonomia |
| **Entrada** | Lista de 13 pontos fixos | Grid de N waypoints (N pode ser 400+) |
| **Escala** | ~10-15 plantas | ~200-1600 pontos |
| **Restrições** | Espaçamento de plantas, solo | Capacidade veículo, autonomia |
| **Saída** | 1 rota (sequência de visita) | Múltiplas rotas |

---

## 🔴 PROBLEMA CRÍTICO: NÃO ESTÃO RESOLVENDO O MESMO PROBLEMA!

### AHA (aha_optimizer.py) - O que ele FAZ:

```python
# Linha 392-428: Lista FIXA de 13 plantas
self.original_points = [
    {"name":"Pequi",  "x":8.0,  "y":6.0,  "yaw":0.0},
    {"name":"Baru",   "x":12.0, "y":8.0,  "yaw":0.0},
    {"name":"Cagaita","x":5.0,  "y":2.0,  "yaw":0.0},
    # ... mais 10 plantas
]
```

**O AHA resolve:** "Qual a melhor ORDEM para visitar 13 plantas específicas?"
- É um **TSP** (Traveling Salesman Problem)
- Função fitness considera: distância + espaçamento + solo + prioridade
- NÃO considera: capacidade do veículo, autonomia, múltiplas rotas
- Gera: **1 única rota** com todas as 13 plantas

### HGS (hgs_planter_node.py + hgs_solver.py) - O que ele FAZ:

```python
# Linha 55-76 do hgs_planter.launch: Grid grande
<arg name="grid_size_x" default="100.0" />
<arg name="grid_size_y" default="100.0" />
<arg name="waypoint_spacing" default="5.0" />
# Gera ~400 waypoints!
```

**O HGS resolve:** "Como dividir 400 pontos em rotas factíveis?"
- É um **CVRP** (Capacitated VRP)
- Considera: capacidade (300 sementes), autonomia (2025m)
- Gera: **Múltiplas rotas** (ex: 27 rotas para 400 pontos)

---

## 📊 COMPARAÇÃO DIRETA

| Característica | AHA | HGS |
|----------------|-----|-----|
| **Tamanho do problema** | 13 pontos | 400+ pontos |
| **Tipo de problema** | TSP | CVRP |
| **Restrição de capacidade** | ❌ Não | ✅ Sim (300 sementes) |
| **Restrição de autonomia** | ❌ Não | ✅ Sim (2025m) |
| **Múltiplas rotas** | ❌ Não (1 rota) | ✅ Sim (~27 rotas) |
| **Retornos à base** | ❌ Não | ✅ Sim |
| **Biblioteca usada** | Implementação própria | hygese (Vidal 2022) |

---

## 🎯 PARA COMPARAR DE FORMA JUSTA:

### Opção 1: Escalar o AHA para CVRP (RECOMENDADO)
O `AHASolverBenchmark` no `solver_benchmark.py` já faz isso!

```python
# Linha 340-450 de solver_benchmark.py
class AHASolverBenchmark(BaseSolver):
    """AHA adaptado para CVRP"""
    
    def _split_into_routes(self, sequence, demands, capacity, autonomy, dm):
        """Split algorithm - divide sequência em rotas factíveis"""
        # ESTE já considera capacidade e autonomia!
```

**Este AHA:**
- Gera permutação de N waypoints
- Usa `_split_into_routes()` para dividir em rotas
- Respeita capacidade e autonomia
- Pode ser comparado diretamente com HGS

### Opção 2: Reduzir HGS para TSP
Usar HGS com instância de 13 pontos e capacidade infinita.
**Não recomendado** - perde a utilidade do HGS.

---

## 📁 ESTRUTURA DOS ARQUIVOS

```
aha_optimizer.py
├── AdvancedAHA_Optimizer     # AHA "puro" para 13 plantas (TSP)
│   ├── fitness_function()    # Considera espaçamento, solo, prioridade
│   ├── guided_foraging()     # Operador AHA
│   ├── territorial_foraging()
│   └── migration_foraging()
│
└── OptimizedPlanterNode      # Nó ROS que usa o AHA
    └── original_points = [...13 plantas...]  # FIXO!

hgs_solver.py  
├── HGSSolver                 # Wrapper do hygese
│   ├── solve()               # Resolve CVRP
│   ├── solve_with_autonomy() # Considera autonomia
│   └── split_routes_by_autonomy() # Pós-processamento
│
└── DroneConfig               # Configuração do drone
    ├── dispenser_capacity = 300
    └── autonomy_meters = 2250

solver_benchmark.py
├── AHASolverBenchmark        # AHA ESCALADO para CVRP! ⬅️ USE ESTE
│   ├── solve()               # Recebe matriz NxN
│   └── _split_into_routes()  # Divide em rotas
│
└── HGSSolverBenchmark        # HGS para benchmark
    └── solve()               # Usa hygese internamente
```

---

## ✅ RECOMENDAÇÃO FINAL

Para sua dissertação, use o **`solver_benchmark.py`** que já tem:

1. **AHASolverBenchmark** - AHA adaptado para CVRP (comparável)
2. **HGSSolverBenchmark** - HGS wrapper
3. **NearestNeighborSolver** - Baseline

Ambos recebem:
- `distance_matrix`: Matriz de distâncias NxN
- `demands`: Lista de demandas
- `capacity`: Capacidade do veículo
- `autonomy`: Autonomia em metros

E retornam:
- `routes`: Lista de rotas
- `total_distance`: Distância total
- `num_routes`: Número de rotas

**NÃO COMPARE:**
- `aha_optimizer.py` (TSP, 13 plantas) 
- com `hgs_planter_node.py` (CVRP, 400 pontos)

**COMPARE:**
- `AHASolverBenchmark` (CVRP, N pontos)
- com `HGSSolverBenchmark` (CVRP, N pontos)

---

## 🔧 CÓDIGO PARA BENCHMARK JUSTO

```python
from solver_benchmark import (
    BenchmarkInstance,
    AHASolverBenchmark,
    HGSSolverBenchmark,
    NearestNeighborSolver,
    BenchmarkRunner
)

# Cria instância de teste (100x100m = ~400 waypoints)
instance = BenchmarkInstance.from_grid(
    grid_size_x=100.0,
    grid_size_y=100.0,
    spacing=5.0,
    base_x=50.0,
    base_y=0.0,
    capacity=225,
    autonomy=2025.0
)

# Solvers
aha = AHASolverBenchmark(population_size=150, max_iterations=1000)
hgs = HGSSolverBenchmark()
nn = NearestNeighborSolver()

# Executa
aha_result = aha.solve(instance.distance_matrix, instance.demands, 
                       instance.capacity, instance.autonomy)
hgs_result = hgs.solve(instance.distance_matrix, instance.demands,
                       instance.capacity, instance.autonomy)

# Compara
print(f"AHA: {aha_result.total_distance:.1f}m em {aha_result.num_routes} rotas")
print(f"HGS: {hgs_result.total_distance:.1f}m em {hgs_result.num_routes} rotas")
```

# Configuração Experimental — Benchmark de Solvers C-SDVRP

## Detalhes Técnicos do Experimento

### Número de Execuções

Cada solver foi executado **10 vezes** por instância, proporcionando análise estatística robusta com múltiplas replicações. A implementação está em `solver_benchmark.py`, na função `run` com o parâmetro `num_runs=10`.

### Limite de Tempo

Cada solver recebe um limite de **30 segundos** por execução, aplicado uniformemente a todos os algoritmos em cada instância. Esse limite garante fairness na comparação entre métodos com diferentes complexidades computacionais. Implementado em `solver_benchmark.py`, na função `run` com o parâmetro `time_limit=30.0`.

### Instâncias de Teste

#### Tamanho das Instâncias

As instâncias foram geradas sinteticamente pelo módulo `GridGenerator`, que implementa a Transformação de Clientes Virtuais conforme Petris, 2024. Cada instância representa uma malha ortogonal uniforme com espaçamento de 2.5 m entre waypoints, escalada em cinco tamanhos de campo:

| Instância | Identificador | Escala |
|-----------|---------------|--------|
| 25×25 m   | `csdvrp_25x25`  | Pequena |
| 50×50 m   | `csdvrp_50x50`  | Média |
| 75×75 m   | `csdvrp_75x75`  | Grande |
| 100×100 m | `csdvrp_100x100` | Muito grande |
| 150×150 m | `csdvrp_150x150` | Extra grande — escalabilidade |

O total de clientes virtuais varia de aproximadamente 81 a 2601, dependendo do tamanho do campo.

#### Distribuição de Commodities

Três tipos de sementes são considerados — Erva, Arbusto e Árvore — cada uma com capacidade por compartimento $Q_k = 100$ unidades, totalizando $Q_{total} = 300$ unidades por viagem. Cada waypoint demanda 15 sementes distribuídas entre os três tipos.

#### Parâmetros de Reflorestamento

- **Autonomia do drone** — aproximadamente 2025 metros, derivada da Eq. do $L_{\max}$ com margem de segurança $\sigma = 0.9$
- **Capacidade total do drone** — 225 unidades, limitada por peso e volume

### Critério de Gap

O gap percentual compara a qualidade relativa de cada solver em relação à melhor solução encontrada pelo HGS dentro do limite de tempo:

$$\text{Gap} (\%) = \frac{C_{\text{solver}} - C_{\text{HGS}}}{C_{\text{HGS}}} \times 100$$

Implementação em `solver_benchmark.py`:

```python
@property
def gap_percent(self) -> Optional[float]:
    """Gap percentual para melhor solução conhecida"""
    if 'optimal' in self.metadata and self.metadata['optimal']:
        return ((self.total_distance - self.metadata['optimal']) / 
                self.metadata['optimal']) * 100
    return None
```

**Interpretação** — Gap = 0% indica solução ótima encontrada; Gap > 0% indica solução subótima com a diferença percentual em relação à ótima; Gap indefinido quando solução ótima não é conhecida.

### Plataforma Computacional

#### Hardware

Os experimentos foram conduzidos em uma workstation CPU-only com processador Intel Core i5-1135G7 a 2.40 GHz e 8 GB de RAM. GPU não foi utilizada.

**Configuração recomendada** — mínimo de 8 GB RAM com processador dual-core 2.4 GHz; recomendado 16 GB RAM com processador quad-core 3.0 GHz.

#### Ambiente de Software

- **Linguagem** — Python 3.8+
- **NumPy** — operações matriciais
- **SciPy** — cálculos científicos
- **HGS** — solver especializado para CVRP, conforme Vidal, 2022

### Solvers Avaliados

Cinco algoritmos de roteamento foram comparados:

- **Nearest Neighbor** — baseline de referência com complexidade O(n²). A partir do depósito, o UAV avança para o cliente mais próximo que satisfaça as restrições de payload e bateria.
- **D-AHA** — Discrete Artificial Hummingbird Algorithm, metaheurística bio-inspirada adaptada para o domínio discreto combinatório. Evolui uma população de soluções usando três operadores: Prefix Crossover, Swap Mutation e Segment Inversion.
- **HGS** — Hybrid Genetic Search, estado da arte conforme Vidal, 2022. Explora soluções viáveis e inviáveis usando fitness penalizado, servindo como referência de qualidade.
- **LKH-Split Optimal** — Lin-Kernighan seguido de partição ótima via programação dinâmica. Garante partição de custo mínimo para uma dada ordenação do tour.
- **LKH-Split Greedy** — Lin-Kernighan seguido de partição gulosa sequencial. Prioriza velocidade computacional sobre qualidade da partição.

### Métricas Capturadas

Para cada execução, o benchmark registra distância total percorrida em metros, número de rotas necessárias, tempo computacional em segundos, gap percentual em relação ao HGS, violações de capacidade e de autonomia por rota, e viabilidade da solução.

### Referências do Código

**Arquivo principal** — `solver_benchmark.py`
- `BenchmarkRunner` — orquestra execução completa
- `BenchmarkInstance` — representa instância de teste
- `SolverResult` — armazena resultado de execução
- `add_csdvrp_instances` — cria instâncias experimentais
- `run` — executa benchmark com parâmetros

**Arquivo de suporte** — `grid_generator.py`
- `GridGenerator` — cria instâncias sintéticas via Transformação de Clientes Virtuais, conforme Petris, 2024

---

**Última atualização** — Fevereiro 2026
**Status** — Configuração pronta para execução experimental

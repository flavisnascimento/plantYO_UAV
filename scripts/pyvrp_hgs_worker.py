#!/usr/bin/env python3
"""
Worker do HGS com capacidade por compartimento, via PyVRP.

PyVRP (Wouda, Lan e Kool, 2024) implementa o Hybrid Genetic Search e aceita uma
capacidade por dimensao de carga, uma para cada compartimento. Exige Python
>= 3.10, entao roda num ambiente separado do ROS (Python 3.8) e e chamado por
hgs_compartimentos.py via subprocess.

Entrada (stdin, JSON):
    dist          matriz inteira NxN (indice 0 = deposito), em centimetros
    loads         lista N de vetores de carga por compartimento (linha 0 = deposito)
    caps          capacidade de cada compartimento, na mesma ordem dos vetores
    max_distance  autonomia por rota, em centimetros
    time_limit    tempo de busca em segundos
    seed          semente do gerador aleatorio

Saida (stdout, JSON):
    routes    lista de rotas, cada uma com os indices 1..N-1 da matriz de entrada
    feasible  True se a melhor solucao respeita todas as restricoes
"""
import json
import sys

from pyvrp import Model
from pyvrp.stop import MaxRuntime


def main():
    d = json.load(sys.stdin)
    dist = d["dist"]
    loads = d["loads"]
    n = len(dist)

    m = Model()
    # As coordenadas nao sao usadas: as distancias vem da matriz de entrada.
    locs = [m.add_location(x=0, y=0) for _ in range(n)]
    m.add_depot(locs[0])
    m.add_vehicle_type(num_available=n - 1, capacity=d["caps"],
                       max_distance=d["max_distance"])
    for i in range(1, n):
        m.add_client(locs[i], delivery=loads[i])
    for i in range(n):
        row = dist[i]
        for j in range(n):
            if i != j:
                m.add_edge(locs[i], locs[j], distance=row[j])

    res = m.solve(stop=MaxRuntime(d["time_limit"]), display=False, seed=d.get("seed", 0))
    # idx de uma atividade de cliente e o indice do cliente (0..N-2) na ordem de
    # insercao; +1 devolve o indice da matriz de entrada.
    routes = [[a.idx + 1 for a in r if a.is_client()] for r in res.best.routes()]
    json.dump({"routes": routes, "feasible": bool(res.is_feasible())}, sys.stdout)


if __name__ == "__main__":
    main()

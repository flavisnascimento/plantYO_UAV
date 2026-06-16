#!/usr/bin/env python3
"""
Helper de modo de compartimento.
- 3comp: roteia TODOS os clientes virtuais juntos (drone leva as 3 guildas)
- 1comp: 3 campanhas (erva, arvore, arbusto), tanque unico de 300 por vez,
         distancias somadas. Pontos das outras guildas nao sao parada na campanha.
Mesmo grid, mesma agregacao Petris, mesmo solver. So muda o agrupamento.
"""
import numpy as np
from grid_generator import PlantType

def _calc_total(routes, dm):
    total = 0.0
    for route in routes:
        if route:
            total += dm[0, route[0]]
            for i in range(len(route) - 1):
                total += dm[route[i], route[i + 1]]
            total += dm[route[-1], 0]
    return total

def _build_subproblem(gen, dm, demands, guild):
    """Submatriz so com os clientes virtuais de uma guilda."""
    gc = [i for i, vc in enumerate(gen.virtual_clients) if vc.commodity == guild]
    if not gc:
        return None, None, None
    k = len(gc)
    sub = np.zeros((k + 1, k + 1))
    for a in range(1, k + 1):
        oa = gc[a - 1] + 1
        sub[0, a] = dm[0, oa]
        sub[a, 0] = dm[oa, 0]
        for b in range(1, k + 1):
            if a != b:
                sub[a, b] = dm[oa, gc[b - 1] + 1]
    sub_dem = [0] + [demands[i + 1] for i in gc]
    return sub, sub_dem, gc

def solve_with_mode(modo, route_fn, gen, dm, demands):
    """
    modo: '1comp' ou '3comp'
    route_fn(dm, demands) -> lista de rotas (indices 1-based no dm passado)
    retorna (rotas_originais, distancia_total, n_campanhas)
    """
    if modo == '3comp':
        routes = route_fn(dm, demands)
        return routes, _calc_total(routes, dm), 1

    # 1comp: campanha por guilda
    all_routes = []
    total = 0.0
    campanhas = 0
    for guild in (PlantType.ERVA, PlantType.ARVORE, PlantType.ARBUSTO):
        sub, sub_dem, gc = _build_subproblem(gen, dm, demands, guild)
        if sub is None:
            continue
        campanhas += 1
        sub_routes = route_fn(sub, sub_dem)
        for r in sub_routes:
            all_routes.append([gc[s - 1] + 1 for s in r])
        total += _calc_total(sub_routes, sub)
    return all_routes, total, campanhas

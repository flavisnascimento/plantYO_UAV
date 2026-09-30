#!/usr/bin/env python3
"""
Restricao de capacidade por compartimento (uma capacidade por guilda).

Cada cliente virtual pertence a uma unica guilda e cabe no compartimento dela,
mas uma rota pode juntar varios clientes da mesma guilda. Por isso conferir so
a capacidade total da rota NAO garante que cada compartimento seja respeitado.
Este modulo concentra a regra usada por todos os solvers e pela validacao:

    para toda rota r e toda guilda k:  soma das demandas de k em r <= Q_k

`commodities` e sempre uma lista alinhada com `demands` (indice 0 = deposito,
valor None) e `caps` um dicionario guilda -> capacidade do compartimento.
As guildas sao identificadas por string ('erva', 'arbusto', 'arvore').
"""

from typing import Dict, List, Optional, Sequence, Tuple


def chave(commodity):
    """Normaliza PlantType ou string para string ('erva', 'arbusto', 'arvore')."""
    if commodity is None:
        return None
    return getattr(commodity, "value", commodity)


def dados_compartimentos(generator) -> Tuple[List[Optional[str]], Dict[str, int]]:
    """Guilda de cada cliente virtual e capacidade de cada compartimento.

    Returns:
        commodities: lista alinhada com generator.get_demands() (indice 0 = None)
        caps: {'erva': Q_erva, 'arbusto': Q_arbusto, 'arvore': Q_arvore}
    """
    if not generator.virtual_clients:
        generator.generate()
    commodities = [None] + [chave(vc.commodity) for vc in generator.virtual_clients]
    cc = generator.config.commodity_capacity
    caps = {"erva": int(cc.erva), "arbusto": int(cc.arbusto), "arvore": int(cc.arvore)}
    return commodities, caps


def cargas(route: Sequence[int], demands: Sequence[int],
           commodities: Sequence) -> Dict[str, int]:
    """Sementes de cada guilda carregadas por uma rota."""
    c: Dict[str, int] = {}
    for cliente in route:
        k = chave(commodities[cliente])
        c[k] = c.get(k, 0) + demands[cliente]
    return c


def rota_ok(route: Sequence[int], demands: Sequence[int],
            commodities: Sequence, caps: Dict) -> bool:
    """True se nenhuma guilda da rota passa da capacidade do seu compartimento."""
    for k, carga in cargas(route, demands, commodities).items():
        if carga > caps.get(k, float("inf")):
            return False
    return True


def violacoes(routes: Sequence[Sequence[int]], demands: Sequence[int],
              commodities: Sequence, caps: Dict) -> List[Tuple[int, str, int, int]]:
    """Lista (indice_da_rota, guilda, carga, capacidade) de cada estouro."""
    out = []
    for i, route in enumerate(routes):
        for k, carga in cargas(route, demands, commodities).items():
            cap = caps.get(k, float("inf"))
            if carga > cap:
                out.append((i, k, carga, cap))
    return out

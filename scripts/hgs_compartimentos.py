#!/usr/bin/env python3
"""
HGS com capacidade por compartimento, chamado a partir do Python do ROS.

O hygese (HGS-CVRP) aceita so uma capacidade escalar, o que basta para o
dispensador de compartimento unico. Com varios compartimentos, cada guilda tem
o seu limite, e a rota precisa respeitar todos. Este modulo monta a instancia
com uma dimensao de carga por compartimento e a resolve com o HGS do PyVRP,
executado no ambiente indicado por PLANTYO_PYVRP_PYTHON (padrao
~/pyvrp_env/bin/python).
"""
import json
import os
import subprocess
from typing import Dict, List, Sequence, Tuple

PYVRP_PYTHON = os.environ.get("PLANTYO_PYVRP_PYTHON",
                              os.path.expanduser("~/pyvrp_env/bin/python"))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyvrp_hgs_worker.py")
GUILDAS = ("erva", "arbusto", "arvore")
ESCALA = 100  # metros -> centimetros, o PyVRP trabalha com inteiros


def multicompartimento(commodities, caps) -> bool:
    """True quando ha pelo menos dois compartimentos ativos."""
    return commodities is not None and bool(caps) and sum(1 for c in caps.values() if c > 0) >= 2


def resolver(distance_matrix, demands: Sequence[int], commodities: Sequence,
             caps: Dict[str, int], autonomy: float, time_limit: float,
             seed: int = 0) -> Tuple[List[List[int]], bool]:
    """Resolve com HGS respeitando cada compartimento. Devolve (rotas, viavel)."""
    if not os.path.exists(PYVRP_PYTHON):
        raise RuntimeError(
            f"Python do PyVRP nao encontrado em {PYVRP_PYTHON}. Crie o ambiente "
            f"(ver README) ou defina PLANTYO_PYVRP_PYTHON.")
    guildas = [k for k in GUILDAS if caps.get(k, 0) > 0]
    loads = [[0] * len(guildas)] + [
        [int(demands[i]) if commodities[i] == k else 0 for k in guildas]
        for i in range(1, len(demands))]
    payload = {
        "dist": [[int(round(float(x) * ESCALA)) for x in row] for row in distance_matrix],
        "loads": loads,
        "caps": [int(caps[k]) for k in guildas],
        "max_distance": int(autonomy * ESCALA),
        "time_limit": float(time_limit),
        "seed": int(seed),
    }
    proc = subprocess.run([PYVRP_PYTHON, WORKER], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=float(time_limit) + 300)
    if proc.returncode != 0:
        raise RuntimeError("HGS (PyVRP) falhou:\n" + proc.stderr[-3000:])
    out = json.loads(proc.stdout)
    return out["routes"], out["feasible"]

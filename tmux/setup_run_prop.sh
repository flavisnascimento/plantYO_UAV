#!/bin/bash
if [ $# -ne 3 ]; then
    echo "Usage: ./setup_run_prop.sh <solver> <modo> <grid_size>"
    exit 1
fi
SOLVER=$1; MODO=$2; GRID=$3
BASE=$(python3 -c "print($GRID / 2)")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MISSION_NAME="${SOLVER}_${MODO}_prop_grid${GRID}_${TIMESTAMP}"
CAP_ERVA=120; CAP_ARBUSTO=120; CAP_ARVORE=60
python3 - << PYEOF
import re
fp = "session.yml"; src = open(fp).read()
old = r'(\s*- waitForControl;\s*sleep\s+\d+;\s*)roslaunch plantyo_uav dispensor_planter\.launch [^\n]*'
new = r'\1roslaunch plantyo_uav dispensor_planter.launch modo:=${MODO} solver:=${SOLVER} grid_size_x:=${GRID}.0 grid_size_y:=${GRID}.0 base_x:=${BASE} mission_name:=${MISSION_NAME} capacity_erva:=${CAP_ERVA} capacity_arbusto:=${CAP_ARBUSTO} capacity_arvore:=${CAP_ARVORE}'
open(fp,"w").write(re.sub(old, new, src))
PYEOF
echo "== PROPORCIONAL 120/120/60 =="
echo "  ${SOLVER} | ${MODO} | ${GRID}x${GRID} | base ${BASE} | erva=${CAP_ERVA} arbusto=${CAP_ARBUSTO} arvore=${CAP_ARVORE}"
echo "  mission: ${MISSION_NAME}"
grep dispensor_planter session.yml

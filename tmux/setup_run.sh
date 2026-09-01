#!/bin/bash
if [ $# -ne 3 ]; then
    echo "Usage: ./setup_run.sh <solver> <modo> <grid_size>"
    exit 1
fi

SOLVER=$1
MODO=$2
GRID=$3
BASE=$(python3 -c "print($GRID / 2)")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MISSION_NAME="${SOLVER}_${MODO}_grid${GRID}_${TIMESTAMP}"

# Preserva a indentacao: substitui SO a parte do "roslaunch..." em diante
python3 << PYEOF
fp = "session.yml"
src = open(fp).read()
import re
# padrao: pega a linha que tem "roslaunch plantyo_uav dispensor_planter" e substitui o meio
old_pattern = r'(\s*- waitForControl;\s*sleep\s+\d+;\s*)roslaunch plantyo_uav dispensor_planter\.launch [^\n]*'
new_content = r'\1roslaunch plantyo_uav dispensor_planter.launch modo:=${MODO} solver:=${SOLVER} grid_size_x:=${GRID}.0 grid_size_y:=${GRID}.0 base_x:=${BASE} mission_name:=${MISSION_NAME}'
src_new = re.sub(old_pattern, new_content, src)
open(fp, "w").write(src_new)
PYEOF

echo "===== configurado ====="
echo "  Solver: ${SOLVER} | Modo: ${MODO} | Grid: ${GRID}x${GRID} | Base: (${BASE},0)"
echo "  Mission name: ${MISSION_NAME}"
echo ""
grep "dispensor_planter" session.yml

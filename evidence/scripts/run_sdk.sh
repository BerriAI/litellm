#!/usr/bin/env bash
# usage: run_sdk.sh <source checkout> <output dir>   (ANTHROPIC_API_KEY must be set for the live smoke)
SRC=$1; OUT=$2; KIT=$(cd "$(dirname "$0")" && pwd); PY=$HOME/repos/litellm/.venv/bin/python
export PYTHONPATH=$SRC:$KIT LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1
cd "$OUT"
{ echo "sha=$(git -C $SRC rev-parse HEAD)"; echo "tracked_changes=$(git -C $SRC status --porcelain | wc -l)"; $PY -c 'import litellm;print(litellm.__file__)'; } > env.log 2>&1
for s in ${SCRIPTS:-repro_issue49929 adjacent_matrix compat_seeded compat_finish_details bridge_site_probe live_anthropic_smoke}; do
  $PY $KIT/$s.py > $s.log 2>&1; echo "$s rc=$?"
done

#!/usr/bin/env bash
# typecheck job of .github/workflows/test-linting.yml. usage: run_typecheck.sh <source checkout> <base sha> <output dir>
SRC=$1; BASE=$2; OUT=$3; export PATH=$HOME/repos/litellm/.venv/bin:$PATH
cd "$SRC"; unset DATABASE_URL DIRECT_URL; export SQLX_OFFLINE=true
python litellm/scripts/type_check_gate.py --base "$BASE" > $OUT/type_check_gate.log 2>&1; echo "type_check_gate rc=$?" > $OUT/rc_typecheck.txt

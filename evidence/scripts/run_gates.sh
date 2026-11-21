#!/usr/bin/env bash
# Steps of .github/workflows/test-linting.yml (python + typecheck jobs) run locally in the locked env.
# usage: run_gates.sh <source checkout> <base sha> <output dir>
SRC=$1; BASE=$2; OUT=$3; export PATH=$HOME/repos/litellm/.venv/bin:$PATH
cd "$SRC"; unset DATABASE_URL DIRECT_URL; export SQLX_OFFLINE=true PYTHONDONTWRITEBYTECODE=1
step() { n=$1; shift; "$@" > $OUT/$n.log 2>&1; echo "$n rc=$?" >> $OUT/rc.txt; }
: > $OUT/rc.txt
step uv_lock_check uv lock --check
git diff --name-only --diff-filter=ACMR "$BASE" HEAD -- ':(glob)litellm/**/*.py' | grep -Ev '^litellm/(enterprise|ui|vscode-extension|rust|proxy-extras|packaging|cli|scripts|runtime)/' > $OUT/ruff_format_files.txt
step ruff_format xargs -a $OUT/ruff_format_files.txt ruff format --check --exclude '/enterprise/'
step mcp_operation_boundary python litellm/scripts/check_mcp_operation_boundary.py
step ruff_source bash -c 'cd litellm && ruff check .'
step ruff_tests ruff check --config .github/linters/ruff-tests.toml tests
step ruff_strict_gate python litellm/scripts/ruff_strict_gate.py --base "$BASE"
step type_discipline_gate python litellm/scripts/type_discipline_gate.py --base "$BASE"
step test_quality_gate python litellm/scripts/test_quality_gate.py --base "$BASE"
step openapi_docs_gate env CARGO_TARGET_DIR=$HOME/repos/litellm/litellm/rust/target python litellm/scripts/openapi_docs_gate.py --base "$BASE"
step circular_imports bash -c 'cd litellm && python ../tests/documentation_tests/test_circular_imports.py'
step migrations_no_data_rewrites python ./tests/code_coverage_tests/check_migrations_no_data_rewrites.py
step migrations_lock_timeout python ./tests/code_coverage_tests/check_migrations_lock_timeout.py
echo "light gates done" >> $OUT/rc.txt

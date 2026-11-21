#!/usr/bin/env bash
# usage: run_tests.sh <source checkout> <output dir>
SRC=$1; OUT=$2; PY=$HOME/repos/litellm/.venv/bin/python
cd "$SRC"; export PYTHONPATH=$SRC LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1 COVERAGE_FILE=$OUT/.coverage
T="tests/unit/caching/test_caching_handler.py tests/unit/litellm_core_utils/llm_response_utils/test_convert_dict_to_response.py tests/unit/litellm_core_utils/llm_response_utils/test_convert_to_streaming_response.py tests/unit/litellm_core_utils/test_streaming_handler.py"
env -u DATABASE_URL -u DIRECT_URL $PY -m coverage run --branch --include='litellm/caching/caching_handler.py,litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py,litellm/litellm_core_utils/streaming_handler.py' -m pytest $T -q -p no:cacheprovider -o addopts= > $OUT/mapped.log 2>&1; echo "mapped rc=$?" > $OUT/rc.txt
$PY -m coverage json -o $OUT/coverage.json > /dev/null 2>&1
$PY $(dirname "$0")/changed_coverage.py "$SRC" "${BASE:-6bcc7ff6b9ea7f21c6801cf999429a04fcc78664}" $OUT/coverage.json > $OUT/changed_coverage.txt 2>&1

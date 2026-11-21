#!/usr/bin/env bash
# One proxy leg: start the proxy from <source checkout> on <port>, wait for readiness, run the driver, stop the proxy.
# usage: run_proxy_leg.sh <source checkout> <port> <output dir>   (redis on 6390 and mock on 8791 must be up; ANTHROPIC_API_KEY set)
SRC=$1; PORT=$2; OUT=$3; KIT=$(cd "$(dirname "$0")" && pwd); G=$HOME/verif/issue49929/proxy-rig; PY=$HOME/repos/litellm/.venv/bin/python
redis-cli -p 6390 flushall > /dev/null
$G/start_proxy.sh "$SRC" $PORT "$OUT/proxy.log"
ready=0; for i in $(seq 1 100); do [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' localhost:$PORT/health/liveliness)" = 200 ] && ready=1 && break; sleep 3; done
echo "sha=$(git -C $SRC rev-parse HEAD) ready=$ready proxy_cwd=$(readlink /proc/$(cat $G/run/proxy_$PORT.pid)/cwd)" > "$OUT/leg.txt"
if [ $ready = 1 ]; then
  (cd "$OUT" && LIVE=1 LITELLM_MASTER_KEY=$(cat $G/.master_key) PYTHONPATH=$KIT timeout 600 $PY $KIT/proxy_driver.py http://127.0.0.1:$PORT > proxy_driver.log 2>&1; echo "driver rc=$?" >> leg.txt)
  $PY $KIT/summarize_proxy.py "$OUT/proxy_driver.log" > "$OUT/proxy_summary.txt" 2>&1
fi
P=$(cat $G/run/proxy_$PORT.pid); kill -TERM -- -$P 2>/dev/null; sleep 5; kill -KILL -- -$P 2>/dev/null
echo "port_listeners_after_stop=$(ss -ltn | grep -c ":$PORT ")" >> "$OUT/leg.txt"

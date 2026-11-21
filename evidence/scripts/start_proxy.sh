#!/usr/bin/env bash
# usage: start_proxy.sh <source checkout> <port> <log file>   (ANTHROPIC_API_KEY must be set)
SRC=$1; PORT=$2; LOG=$3; G=$(cd "$(dirname "$0")" && pwd)
cd "$SRC"
export LITELLM_MASTER_KEY=$(cat $G/.master_key) LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SRC:$SRC/litellm/proxy-extras:$SRC/litellm/enterprise
env -u DATABASE_URL -u DIRECT_URL setsid nohup $HOME/repos/litellm/.venv/bin/python litellm/proxy/proxy_cli.py \
  --config $G/config.yaml --port $PORT --num_workers 2 > "$LOG" 2>&1 &
echo $! > $G/run/proxy_$PORT.pid

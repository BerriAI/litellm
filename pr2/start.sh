#!/bin/bash
# usage: start.sh <worktree> <config> <db name> <log> [slack kind]
set -e
unset OP_SERVICE_ACCOUNT_TOKEN DIRECT_URL AWS_BEARER_TOKEN_BEDROCK
export DATABASE_URL=$(echo "$DATABASE_URL" | sed -E "s#/litellm(\?.*)?\$#/$3#")
export LITELLM_SALT_KEY=sk-pr2-disposable-salt LITELLM_MASTER_KEY=sk-pr2-disposable-master PROXY_BASE_URL=http://127.0.0.1:4396
cd "$1"; export PYTHONPATH="$1:$1/litellm/proxy-extras:$1/litellm/enterprise"
setsid nohup $HOME/repos/litellm/.venv/bin/python litellm/proxy/proxy_cli.py --config "$2" --port 4396 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver >> "$4" 2>&1 < /dev/null &
echo $! > $HOME/cachezero/pr2n/proxy.pid
for i in $(seq 120); do curl -sf -m 2 http://127.0.0.1:4396/health/liveliness >/dev/null && break; sleep 2; done
curl -sf -m 2 http://127.0.0.1:4396/health/liveliness >/dev/null && echo up || (echo FAILED; tail -5 "$4")

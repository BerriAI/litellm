#!/bin/bash
# usage (host): start_ctr.sh <worktree> <config> <db name> <log>   -- starts the proxy inside container cachezero-pr2-glibc239 (host network)
set -e
DBU=$(echo "$DATABASE_URL" | sed -E "s#/litellm(\?.*)?\$#/$3#")
docker exec -d -e DATABASE_URL="$DBU" -e ANTHROPIC_API_KEY -e LITELLM_SALT_KEY=sk-pr2-disposable-salt -e LITELLM_MASTER_KEY=sk-pr2-disposable-master \
  -e PROXY_BASE_URL=http://127.0.0.1:4396 -e PYTHONPATH="$1:$1/litellm/proxy-extras:$1/litellm/enterprise" -e PATH=$HOME/repos/litellm/.venv/bin:/home/ubuntu/.nvm/versions/node/v24.22.3/bin:/usr/bin:/bin -w "$1" \
  cachezero-pr2-glibc239 bash -c "exec python litellm/proxy/proxy_cli.py --config $2 --port 4396 --num_workers 2 --detailed_debug --reload --use_v2_migration_resolver >> $4 2>&1"
for i in $(seq 150); do curl -sf -m 2 http://127.0.0.1:4396/health/liveliness >/dev/null && break; sleep 2; done
curl -sf -m 2 http://127.0.0.1:4396/health/liveliness >/dev/null && echo up || (echo FAILED; tail -5 "$4")

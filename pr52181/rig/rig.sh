#!/bin/bash
# Live A/B rig for the BYOK shared auth code PR on the VM.
# usage: rig.sh <base_tree> <head_tree> [redis|noredis] [workers]
# Per side: own Postgres, own Redis (redis mode), a recorder in front of the shared BYOK upstream, and a proxy.
set -u
BASE_TREE="$1"; HEAD_TREE="$2"; MODE="${3:-redis}"; WORKERS="${4:-4}"
NS=prrisk49642
RIG=$HOME/rig49642
PG_A=15292; PG_B=15293; PX_A=16292; PX_B=16293; RD_A=16492; RD_B=16493; UP=16592; REC_A=16692; REC_B=16693
cd "$RIG" || exit 1

if ! docker ps --format '{{.Names}}' | grep -q "^${NS}-pg-a$"; then
  for side in a b; do
    port=$PG_A; [ $side = b ] && port=$PG_B
    docker run --rm -d --name ${NS}-pg-$side -p 127.0.0.1:$port:5432 \
      -e POSTGRES_DB=litellm -e POSTGRES_USER=llmproxy -e POSTGRES_PASSWORD=dbpassword9090 postgres:16 >/dev/null || exit 1
    rport=$RD_A; [ $side = b ] && rport=$RD_B
    docker run --rm -d --name ${NS}-redis-$side -p 127.0.0.1:$rport:6379 redis:7 >/dev/null || exit 1
  done
  for side in a b; do
    for _ in $(seq 1 60); do docker exec ${NS}-pg-$side pg_isready -U llmproxy -d litellm >/dev/null 2>&1 && break; sleep 1; done
  done
  sleep 3
fi

for side in a b; do
  rport=$RD_A; [ $side = b ] && rport=$RD_B
  {
    echo "model_list: []"
    echo "general_settings:"
    echo "  master_key: sk-rig-49642"
    if [ "$MODE" = redis ]; then
      echo "  coordination_redis:"
      echo "    host: 127.0.0.1"
      echo "    port: $rport"
      echo "router_settings:"
      echo "  redis_host: 127.0.0.1"
      echo "  redis_port: $rport"
    fi
  } > "$RIG/config-$side.yaml"
done

pgrep -f "rig49642/[u]pstream.py" >/dev/null || {
  setsid nohup "$HEAD_TREE/.venv/bin/python" "$RIG/upstream.py" $UP > "$RIG/upstream.out" 2>&1 < /dev/null &
  sleep 3
}
recorder () {  # recorder <port> <tag>
  pkill -f "rig49642/[r]ecorder.py $1 " 2>/dev/null
  setsid nohup python3 "$RIG/recorder.py" "$1" "http://127.0.0.1:$UP" "$RIG/wire-$2.jsonl" > "$RIG/recorder-$2.out" 2>&1 < /dev/null &
  for _ in $(seq 1 20); do grep -q "listening on" "$RIG/recorder-$2.out" && return 0; sleep 0.5; done
  cat "$RIG/recorder-$2.out"; return 1
}
recorder $REC_A base || exit 1
recorder $REC_B head || exit 1

launch () {  # launch <tree> <pgport> <pxport> <tag> <side>
  local tree=$1 tag=$4 side=$5
  local db="postgresql://llmproxy:dbpassword9090@127.0.0.1:$2/litellm"
  if [ -f "$RIG/proxy-$tag.pid" ]; then kill -- -"$(cat "$RIG/proxy-$tag.pid")" 2>/dev/null; sleep 3; fi
  for _ in $(seq 1 20); do ss -Htan "sport = :$3" | grep -q LISTEN || break; sleep 1; done
  cd "$tree" || return 1
  PATH="$tree/.venv/bin:$PATH" .venv/bin/prisma generate --schema litellm/proxy/schema.prisma > "$RIG/prisma-$tag.log" 2>&1 || { tail -5 "$RIG/prisma-$tag.log"; return 1; }
  DATABASE_URL="$db" PATH="$tree/.venv/bin:$PATH" .venv/bin/prisma db push --schema litellm/proxy/schema.prisma --skip-generate --accept-data-loss >> "$RIG/prisma-$tag.log" 2>&1 || { tail -5 "$RIG/prisma-$tag.log"; return 1; }
  PYTHONPATH="$tree" LITELLM_DISABLE_NO_REDIS_WARNING=true DATABASE_URL="$db" PATH="$tree/.venv/bin:$PATH" \
    setsid nohup .venv/bin/python -m litellm.proxy.proxy_cli --config "$RIG/config-$side.yaml" --host 127.0.0.1 --port "$3" --num_workers "$WORKERS" \
    > "$RIG/proxy-$tag.log" 2>&1 < /dev/null &
  echo $! > "$RIG/proxy-$tag.pid"
  cd "$RIG" || return 1
}
launch "$BASE_TREE" $PG_A $PX_A base a || exit 1
launch "$HEAD_TREE" $PG_B $PX_B head b || exit 1

for port in $PX_A $PX_B; do
  ok=0
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/health/liveliness" >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  [ $ok = 1 ] || { echo "proxy on $port never came up"; exit 1; }
done
sleep 8
for tag in base head; do
  echo "$tag workers: $(grep -c 'Started server process' "$RIG/proxy-$tag.log") mode=$MODE"
done
echo RIG_READY

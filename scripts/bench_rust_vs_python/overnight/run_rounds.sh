#!/usr/bin/env bash
# Resumable overnight rounds. Each round runs one load pair, one sequential pair and, every PROF_EVERY rounds,
# one profiled run. Finished runs are skipped, and the script exits once fewer than ROUND_S seconds of BUDGET_S remain,
# so it can be relaunched in chunks that each stay under the 2 h background-task limit.
set -u
H=${BENCH_HARNESS_PYTHON:-/home/user/bench/harness/bin/python}
B=$(cd "$(dirname "$0")/.." && pwd)
WT=${WORKTREES:-/home/user/bench/wt}
R=${RESULTS:-/home/user/bench/results}
TAG=${TAG:?set TAG, e.g. n15}
A=${VARIANT_A:?set VARIANT_A}
A2=${VARIANT_B:?set VARIANT_B, the A/A twin of VARIANT_A}
LOAD_PLANS=${LOAD_PLANS:?set LOAD_PLANS}
SEQ_PLANS=${SEQ_PLANS:?set SEQ_PLANS}
ROUNDS=${ROUNDS:-24}
PROF_EVERY=${PROF_EVERY:-3}
PROF_SECONDS=${PROF_SECONDS:-180}
PROF_RATE=${PROF_RATE:-50}
BUDGET_S=${BUDGET_S:-6960}
ROUND_S=${ROUND_S:-1680}
LOG=$R/$TAG.log
START=$(date +%s)
export BENCH_HARNESS_PYTHON=$H
cd "$B"

clean_ports() {
  pkill -f "$WT/.*multiprocessing-fork" 2>/dev/null
  pkill -f 'proxy_cli.py' 2>/dev/null
  pkill -f 'moto_server' 2>/dev/null
  pkill -f 'server.py anthropic' 2>/dev/null
  pkill -f 'locust' 2>/dev/null
  sleep 2
}

done_run() { [ -f "$R/$TAG-$1/$2-r$3/meta.json" ]; }

one() {
  local kind=$1 round=$2 seed=$3; shift 3
  if done_run "$kind" "$A" "$round" && { [ "$kind" = prof ] || done_run "$kind" "$A2" "$round"; }; then return; fi
  local tmp=$R/$TAG-tmp
  rm -rf "$tmp"
  "$H" -u run_bench.py --worktrees "$WT" --out "$tmp" "$@" --reps 1 --seed "$seed" >> "$LOG" 2>&1
  clean_ports
  mkdir -p "$R/$TAG-$kind"
  for d in "$tmp"/*-r1; do
    [ -d "$d" ] || continue
    local name dest
    name=$(basename "$d" -r1)
    dest=$R/$TAG-$kind/$name-r$round
    rm -rf "$dest"
    mv "$d" "$dest"
    if [ -f "$dest/error.txt" ]; then
      echo "$(date -u +%FT%TZ) $kind $name r$round FAILED $(cat "$dest/error.txt")"
    else
      echo "$(date -u +%FT%TZ) $kind $name r$round $("$H" -c "import json,sys;m=json.load(open(sys.argv[1]));print({k:m[k] for k in ('ok','records','prometheus_requests_delta','s3_objects')})" "$dest/meta.json")"
    fi
  done
}

clean_ports
rm -rf "$R/$TAG-tmp"
echo "chunk start $(date -u +%FT%TZ) uptime: $(uptime -p)"
for i in $(seq 1 "$ROUNDS"); do
  [ -f "$R/$TAG.STOP" ] && { echo "stopped at round $i"; exit 0; }
  prof_round=$(( i % PROF_EVERY == 0 ? i / PROF_EVERY : 0 ))
  if done_run seq "$A2" "$i" && { [ "$prof_round" = 0 ] || done_run prof "$A" "$prof_round"; }; then continue; fi
  if [ $(( $(date +%s) - START + ROUND_S )) -gt "$BUDGET_S" ]; then echo "CHUNK END before round $i $(date -u +%FT%TZ)"; exit 0; fi
  one load "$i" $((1000 + i)) --plans "$LOAD_PLANS" --variants "$A,$A2" --order rounds
  one seq "$i" $((2000 + i)) --plans "$SEQ_PLANS" --variants "$A,$A2" --order rounds --users 1 \
    --first-byte-s 0 --first-byte-jitter-s 0 --token-interval-s 0
  if [ "$prof_round" != 0 ]; then
    one prof "$prof_round" $((3000 + i)) --plans "$LOAD_PLANS" --variants "$A" \
      --profile-seconds "$PROF_SECONDS" --profile-rate "$PROF_RATE"
  fi
done
echo "ALL DONE $(date -u +%FT%TZ)"

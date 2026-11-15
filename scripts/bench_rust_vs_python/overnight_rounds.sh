#!/usr/bin/env bash
set -u
H=/home/user/bench/harness/bin/python
B=/home/user/litellm/scripts/bench_rust_vs_python
R=/home/user/bench/results
ROUNDS=${ROUNDS:-24}
BUDGET_S=${BUDGET_S:-6600}
ROUND_S=1800
START=$(date +%s)
export BENCH_HARNESS_PYTHON=$H
cd $B

clean_ports() {
  pkill -f 'bench/wt/.*multiprocessing-fork' 2>/dev/null
  pkill -f 'proxy_cli.py' 2>/dev/null
  pkill -f 'moto_server' 2>/dev/null
  pkill -f 'server.py anthropic' 2>/dev/null
  pkill -f 'locust' 2>/dev/null
  sleep 2
}

one() {
  local kind=$1 round=$2 seed=$3; shift 3
  if [ -f $R/n15-$kind/N-r$round/meta.json ] && { [ $kind = prof ] || [ -f $R/n15-$kind/Nb-r$round/meta.json ]; }; then return; fi
  local tmp=$R/n15-tmp
  rm -rf $tmp
  $H -u run_bench.py --worktrees /home/user/bench/wt --out $tmp "$@" --reps 1 --seed $seed >> /home/user/bench/overnight15.log 2>&1
  clean_ports
  mkdir -p $R/n15-$kind
  for d in $tmp/*-r1; do
    [ -d "$d" ] || continue
    local name; name=$(basename "$d" -r1)
    rm -rf $R/n15-$kind/$name-r$round
    mv "$d" $R/n15-$kind/$name-r$round
    echo "$(date -u +%FT%TZ) $kind $name r$round $( [ -f $R/n15-$kind/$name-r$round/error.txt ] && cat $R/n15-$kind/$name-r$round/error.txt || $H -c "import json,sys;m=json.load(open(sys.argv[1]));print({k:m[k] for k in ('ok','records','prometheus_requests_delta','s3_objects')})" $R/n15-$kind/$name-r$round/meta.json)"
  done
}

clean_ports
rm -rf $R/n15-tmp
echo "chunk start $(date -u +%FT%TZ) uptime: $(uptime -p)"
for i in $(seq 1 $ROUNDS); do
  [ -f /home/user/bench/STOP ] && { echo "stopped at round $i"; exit 0; }
  if [ -f $R/n15-seq/Nb-r$i/meta.json ] && { [ $((i % 3)) != 0 ] || [ -f $R/n15-prof/N-r$((i / 3))/meta.json ]; }; then continue; fi
  if [ $(( $(date +%s) - START + ROUND_S )) -gt $BUDGET_S ]; then echo "CHUNK END before round $i $(date -u +%FT%TZ)"; exit 0; fi
  one load $i $((1000 + i)) --plans /home/user/bench/plans/n15 --variants N,Nb --order rounds
  one seq $i $((2000 + i)) --plans /home/user/bench/plans/seq --variants N,Nb --order rounds --users 1 \
    --first-byte-s 0 --first-byte-jitter-s 0 --token-interval-s 0
  if [ $((i % 3)) = 0 ]; then
    one prof $((i / 3)) $((3000 + i)) --plans /home/user/bench/plans/n15 --variants N --profile-seconds 180 --profile-rate 50
  fi
done
echo "ALL DONE $(date -u +%FT%TZ)"

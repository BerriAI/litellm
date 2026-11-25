#!/bin/bash
# One side of the live proof: A/B with no Mixpanel, sad/burst/pause, then the Mixpanel leg. Runs from the tree.
set -u
L=$1; LP=$2; AP=$3
H=$HOME/lit10367/live
LOG=$H/run-$L.log
echo "tree $PWD start $(date -u +%H:%M:%S)" > "$LOG"
test -f litellm/rust_bridge/_native.abi3.so || cp ~/lit10367-native.abi3.so litellm/rust_bridge/_native.abi3.so
.venv/bin/python -c 'import litellm, litellm.rust_bridge._native, opentelemetry.proto' >> "$LOG" 2>&1 || { echo "IMPORT FAIL" >> "$LOG"; exit 1; }
echo "prisma skipped: base side generated the shared venv client from the same schema" >> "$LOG"
timeout 1500 .venv/bin/python "$H/ab.py" "$L" "$AP" > "$H/ab-$L.out" 2>&1; echo "ab exit $? $(date -u +%H:%M:%S)" >> "$LOG"
timeout 900 .venv/bin/python "$H/ab2.py" "$L" "$AP" > "$H/ab2-$L.out" 2>&1; echo "ab2 exit $? $(date -u +%H:%M:%S)" >> "$LOG"
timeout 600 .venv/bin/python "$H/bodykeys.py" "$L" "$AP" > "$H/bk-$L.out" 2>&1; echo "bodykeys exit $? $(date -u +%H:%M:%S)" >> "$LOG"
timeout 1800 .venv/bin/python "$H/leg_v3.py" "live-$L" "$LP" > "$H/leg-$L.out" 2>&1; echo "leg exit $? $(date -u +%H:%M:%S)" >> "$LOG"
echo DONE >> "$LOG"

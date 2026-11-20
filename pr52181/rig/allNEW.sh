#!/bin/bash
# Every live leg at the given head tag in one runner, plus the gauntlet lead cells. Ends with ALLNEW_READY and a Redis rig left up for the browser leg.
# usage: allNEW.sh <tag>
cd ~/rig49642 || exit 1
TAG="$1"
BASE=~/w/pr49642-base-14aee0cb; HEAD=~/w/pr49642-byok-460903be; PY=$HEAD/.venv/bin/python
echo "head tree $TAG: $(grep -c _consume_auth_code $HEAD/litellm/proxy/mcp_server/byok_oauth_endpoints.py) consume lines, $(grep -c increment_if_present $HEAD/litellm/proxy/mcp_server/byok_oauth_endpoints.py) Lua release lines, base: $(grep -c _release_auth_code_slot $BASE/litellm/proxy/mcp_server/byok_oauth_endpoints.py)"
./teardown.sh; rm -f rows-*-$TAG.jsonl state.json wire-*.jsonl codes-restart.json codes-kill.json
echo "== rig redis 4"; timeout 900 ./rig.sh $BASE $HEAD redis 4 || { echo RIG_FAILED; echo ALLNEW_READY; exit 1; }
for phase in setup main chaos window_rollover wait_expiry; do
  echo "== $phase"; timeout 1500 $PY driver.py $phase rows-$phase-$TAG.jsonl 2>&1
done
echo "base deaths $(grep -c "Child process .* died" proxy-base.log) 500s $(grep -c " 500 " proxy-base.log)"
echo "head deaths $(grep -c "Child process .* died" proxy-head.log) 500s $(grep -c " 500 " proxy-head.log)"
for t in base head; do $PY -c "
import json
rows=[json.loads(l) for l in open('wire-$t.jsonl')]
keys=sorted({k.lower() for r in rows for k in (r.get('headers') or {})})
print('wire $t', len(rows), keys)
"; done
cp wire-base.jsonl wire-base-$TAG.jsonl; cp wire-head.jsonl wire-head-$TAG.jsonl
echo "== restart leg"
timeout 300 $PY restart_leg.py issue codes-restart.json
timeout 900 ./rig.sh $BASE $HEAD redis 4 || { echo RIG2_FAILED; }
timeout 300 $PY restart_leg.py redeem codes-restart.json "both proxies restarted"
timeout 300 $PY restart_leg.py issue codes-kill.json
for tag in base head; do
  w=$(ps --ppid "$(cat proxy-$tag.pid)" -o pid=,args= | grep multiprocessing | head -1 | awk '{print $1}')
  echo "$tag killing worker $w"; [ -n "$w" ] && kill -9 "$w"
done
sleep 5
timeout 300 $PY restart_leg.py redeem codes-kill.json "one worker killed"
echo "== rig noredis 1"; ./teardown.sh; timeout 900 ./rig.sh $BASE $HEAD noredis 1 && RIG_MODE="no Redis, 1 worker" timeout 600 $PY driver.py noredis rows-noredis-$TAG.jsonl 2>&1
echo "== rig noredis 4"; ./teardown.sh; timeout 900 ./rig.sh $BASE $HEAD noredis 4 && RIG_FLOWS=40 RIG_MODE="no Redis, 4 workers" timeout 600 $PY driver.py noredis rows-noredis-$TAG.jsonl 2>&1
echo "== rig redis 1 for the gauntlet leads"; ./teardown.sh; timeout 900 ./rig.sh $BASE $HEAD redis 1 && timeout 1500 $PY leads.py rows-leads-$TAG.jsonl 2>&1
for tag in base head; do
  echo "leads $tag: slot-not-released $(grep -c 'slot not released' proxy-$tag.log), store-unavailable $(grep -c 'store unavailable' proxy-$tag.log), deaths $(grep -c 'Child process .* died' proxy-$tag.log), 500s $(grep -c ' 500 ' proxy-$tag.log)"
done
echo "== rig redis 4 for UI"; ./teardown.sh; rm -f state.json; timeout 900 ./rig.sh $BASE $HEAD redis 4 && timeout 600 $PY driver.py setup rows-ui-setup-$TAG.jsonl
echo ALLNEW_READY

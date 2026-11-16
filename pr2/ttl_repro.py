"""Bugbot 4581492446 repro refresh at PR2 5796f53 (same script run earlier at 8e01f88): whether a late hit or late proof of STREAK_SCRIPT extends the proof key's expiry (KEYS[2])
but not the retained misses key (KEYS[3]). Runs the shipped script on a real Redis (port 6392, task-owned), with
ttl=8000 ms instead of 300000 ms so the boundary is reachable. No provider calls, no source edits.
Timeline (ms after proof start T): proof started 0; hit H started 2000 ended 2100; misses M1 started 2500, M2 started
2600. Callback arrival order: P, M1, M2, H (H is late), all after T+3000. M3 starts and arrives after T+8000."""
import json, sys, time, uuid
import redis
sys.path.insert(0, "/home/ubuntu/cachezero/wt/pr2")
from litellm.integrations.prompt_cache_alerts.detector import MISSES_TO_ALERT, STREAK_SCRIPT, _keys

r = redis.Redis(port=6392, decode_responses=True)
run = r.register_script(STREAK_SCRIPT)
TTL = 8000
ms = lambda: int(time.time() * 1000)

def case(name, refresh):
    keys = _keys(f"bugbot-repro-{uuid.uuid4().hex}", "cfg", "prefix")
    _, state, misses, _ = keys
    T = ms()
    call = lambda ev, s, e, cid: run(keys=keys, args=[ev, s, e, ms(), TTL, cid, 2048, MISSES_TO_ALERT])
    out = {"case": name, "ttl_ms": TTL, "steps": []}
    def step(label, reply=None):
        out["steps"].append({"step": label, "at_ms": ms() - T, "reply": reply, "state_pttl": r.pttl(state),
                             "misses_pttl": r.pttl(misses), "misses_zcard": r.zcard(misses),
                             "proof_until_ms": (int(r.hget(state, "until")) - T) if r.exists(state) else None})
    step("P proof", call("proof", T, T + 10, "p"))
    while ms() < T + 3000: time.sleep(0.01)
    step("M1 miss started 2500", call("miss", T + 2500, T + 2550, "m1"))
    step("M2 miss started 2600", call("miss", T + 2600, T + 2650, "m2"))
    step("late %s started 2000" % refresh, call(refresh, T + 2000, T + 2100, "h"))
    if True:
        while ms() < T + 8100: time.sleep(0.01)
        step("waited past old boundary T+8000; misses key exists=%d" % r.exists(misses))
    s = ms()
    step("M3 miss started now", call("miss", s, s + 10, "m3"))
    out["alert_triggered"] = bool(out["steps"][-1]["reply"])
    r.delete(*keys)
    return out

res = [case('late hit, M3 after the old boundary', 'hit'), case('late proof, M3 after the old boundary', 'proof')]
import subprocess
json.dump({'sha': subprocess.check_output(['git','-C','/home/ubuntu/cachezero/wt/pr2','rev-parse','HEAD'],text=True).strip(), 'cases': res}, open('repro_5796f53.json', 'w'), indent=1)
for c in res:
    print(c['case'], '-> alert_triggered', c['alert_triggered'])
    for s in c['steps']: print('  ', json.dumps(s))

"""Codes issued before a proxy restart or a worker kill, redeemed after it.

usage: python restart_leg.py issue <codes.json> | redeem <codes.json> <label>
"""
import json
import sys

sys_argv = sys.argv[:]
sys.argv = ["driver.py", "none", "/dev/null"]
exec(compile(open("driver.py").read().rsplit("\n{\"setup\"", 1)[0], "driver.py", "exec"))
mode, path = sys_argv[1], sys_argv[2]
st = load_state()
if mode == "issue":
    out = {}
    for side in ("base", "head"):
        out[side] = []
        for i in range(10):
            u = st[side]["users"][["u1", "u2", "u3", "u4"][i % 4]]
            r, code, verifier = authorize(side, u["cookie"], st[side]["server_id"], f"byok-restart-{side}-{i}")
            out[side].append([r.status_code, code, verifier])
    json.dump(out, open(path, "w"))
    print("issued", {s: [c[0] for c in v] for s, v in out.items()})
else:
    label = sys_argv[3]
    codes = json.load(open(path))
    for side in ("base", "head"):
        res = {}
        for status, code, verifier in codes[side]:
            if code is None:
                res[f"authorize {status}"] = res.get(f"authorize {status}", 0) + 1
                continue
            t = exchange(side, code, verifier)
            k = f"{t.status_code} {err(t)}"
            res[k] = res.get(k, 0) + 1
        ok = res.get("200 access_token", 0)
        emit(side, "restart", f"10 codes issued, then {label}, then redeemed", "head 10/10", f"{ok}/10 token 200: {res}", True)

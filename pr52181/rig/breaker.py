"""How long BYOK connects keep failing after a Redis outage ends, on the 4-worker Redis rig.

usage: python breaker.py <out.jsonl>
Redis is paused under a 20-flow burst so worker circuit breakers open, then unpaused. Every 5s after that,
one full connect per burst user is driven until every connect in a probe returns 200, up to 90s.
"""
import base64
import hashlib
import json
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import httpx

SIDES = {"base": "http://127.0.0.1:16292", "head": "http://127.0.0.1:16293"}
REDIS = {"base": "prrisk49642-redis-a", "head": "prrisk49642-redis-b"}
PG = {"base": "prrisk49642-pg-a", "head": "prrisk49642-pg-b"}
REC = {"base": 16692, "head": 16693}
MASTER = "sk-rig-49642"
CALLBACK = "http://127.0.0.1:47777/callback"
OUT = sys.argv[1]
RUN = secrets.token_hex(3)


def emit(side, scenario, cell, expected, observed, ok):
    row = {"ts": round(time.time(), 2), "side": side, "scenario": scenario, "cell": cell,
           "expected": expected, "observed": observed, "pass": bool(ok)}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(("PASS " if ok else "FAIL ") + f"{side:4} {scenario}/{cell}: {observed}", flush=True)


def pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def post(url, **kw):
    with httpx.Client(timeout=60) as client:
        return client.post(url, **kw)


def admin(side, path, body):
    r = post(SIDES[side] + path, json=body, headers={"Authorization": f"Bearer {MASTER}"})
    assert r.status_code in (200, 201), (side, path, r.status_code, r.text[:300])
    return r.json()


def onboarding_key(ui_token):
    payload = ui_token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["key"]


def fresh_user(side, name):
    email = f"{name}-{side}-{RUN}@rig49642.test"
    pw = f"Pw-{name}-{secrets.token_hex(6)}!9"
    user = admin(side, "/user/new", {"user_email": email, "user_role": "internal_user"})
    invite = admin(side, "/invitation/new", {"user_id": user["user_id"]})
    with httpx.Client(timeout=60) as client:
        got = client.get(SIDES[side] + "/onboarding/get_token", params={"invite_link": invite["id"]})
        assert got.status_code == 200, (side, got.status_code, got.text[:200])
        claimed = client.post(SIDES[side] + "/onboarding/claim_token",
                              json={"invitation_link": invite["id"], "user_id": user["user_id"], "password": pw},
                              headers={"Authorization": "Bearer " + onboarding_key(got.json()["token"])})
        assert claimed.status_code == 200, (side, claimed.status_code, claimed.text[:200])
    cookie = post(SIDES[side] + "/login", data={"username": email, "password": pw}, follow_redirects=False).cookies.get("token")
    assert cookie, (side, name)
    return cookie, user["user_id"]


def authorize(side, cookie, sid, api_key):
    verifier, challenge = pkce()
    r = post(SIDES[side] + "/v1/mcp/oauth/authorize", cookies={"token": cookie}, follow_redirects=False, data={
        "client_id": "rig-client", "redirect_uri": CALLBACK, "code_challenge": challenge,
        "code_challenge_method": "S256", "state": secrets.token_hex(4), "server_id": sid, "api_key": api_key})
    code = parse_qs(urlparse(r.headers["location"]).query)["code"][0] if r.status_code == 302 else None
    return r, code, verifier


def exchange(side, code, verifier):
    return post(SIDES[side] + "/v1/mcp/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK,
        "code_verifier": verifier, "client_id": "rig-client"})


def err(r):
    try:
        body = r.json()
    except ValueError:
        return r.text[:80]
    return body.get("error") or body.get("detail") or ("access_token" if "access_token" in body else str(body)[:80])


def redis_cli(side, *args):
    out = subprocess.run(["docker", "exec", REDIS[side], "redis-cli", *args], capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def acl(side, *rules):
    got = redis_cli(side, "ACL", "SETUSER", "default", *rules)
    assert got == "OK", (side, rules, got)


def credential_rows(side, user_id, sid):
    out = subprocess.run(["docker", "exec", PG[side], "psql", "-U", "llmproxy", "-d", "litellm", "-tAc",
                          f"select count(*) from \"LiteLLM_MCPUserCredentials\" where user_id='{user_id}' and server_id='{sid}'"],
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def start_of_fresh_window(need_s):
    left = 300 - time.time() % 300
    if left < need_s:
        time.sleep(left + 1)



def flow(side, cookie, sid, key):
    r, code, verifier = authorize(side, cookie, sid, key)
    if code is None:
        return f"{r.status_code} at submit"
    t = exchange(side, code, verifier)
    return f"{t.status_code}" if t.status_code == 200 else f"{t.status_code} {err(t)}"


def breaker(side, sid):
    users = [fresh_user(side, f"brk{i}")[0] for i in range(4)]
    subprocess.run(["docker", "pause", REDIS[side]], check=True, timeout=30)
    try:
        with ThreadPoolExecutor(10) as pool:
            burst = list(pool.map(lambda i: flow(side, users[i % 4], sid, f"byok-brk-{side}"), range(20)))
            time.sleep(1)
    finally:
        subprocess.run(["docker", "unpause", REDIS[side]], check=True, timeout=30)
    t0 = time.time()
    probes = []
    while time.time() - t0 < 90:
        time.sleep(5)
        with ThreadPoolExecutor(4) as pool:
            got = list(pool.map(lambda u: flow(side, u, sid, f"byok-brk-{side}"), users))
        probes.append((round(time.time() - t0, 1), got))
        print(side, probes[-1], flush=True)
        if all(g == "200" for g in got):
            break
    first_clean = next((t for t, got in probes if all(g == "200" for g in got)), None)
    tally = {}
    for g in burst:
        tally[g] = tally.get(g, 0) + 1
    emit(side, "redis_outage", "connects every 5s after a 7s Redis pause under a 20-flow burst, until all 4 return 200",
         "head 4 x 200 within the 60s breaker cool-down plus one probe",
         f"burst {tally}; first all-200 probe at {first_clean}s; probes "
         + "; ".join(f"{t}s {sorted(set(got))}" for t, got in probes),
         first_clean is not None if side == "head" else True)


def main():
    sids = {side: admin(side, "/v1/mcp/server", {
        "server_name": f"byokbrk{RUN}", "alias": f"byokbrk{RUN}", "transport": "http",
        "url": f"http://127.0.0.1:{REC[side]}/mcp", "auth_type": "api_key", "is_byok": True})["server_id"]
        for side in SIDES}
    for side in ("head", "base"):
        breaker(side, sids[side])


main()

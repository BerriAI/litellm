"""Validate the gauntlet's three undeclared #52181 leads on a live A/B rig with real Redis.

usage: python leads.py <out.jsonl>
Each fault is a Redis ACL change on the proxy's own Redis user, applied between requests: a real Redis
refusing a command, not a patched client. A row's pass means "no damage"; a head FAIL confirms the lead.
"""
import base64
import hashlib
import json
import secrets
import subprocess
import sys
import time
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


def l5(side, sid):
    cookie, uid = fresh_user(side, "l5")
    r, code, verifier = authorize(side, cookie, sid, f"byok-l5-{side}")
    assert code, (side, r.status_code, r.text[:200])
    acl(side, "-expire")
    try:
        first = exchange(side, code, verifier)
    finally:
        acl(side, "+@all")
    claim = redis_cli(side, "GET", "mcp_byok_auth_code_claim:" + code)
    claim_ttl = redis_cli(side, "TTL", "mcp_byok_auth_code_claim:" + code)
    retry = exchange(side, code, verifier)
    rows = credential_rows(side, uid, sid)
    connected = first.status_code == 200 or retry.status_code == 200
    emit(side, "L5", "token request while Redis refuses EXPIRE, retried once EXPIRE works",
         "the connect completes on the first try or the retry",
         f"first {first.status_code} {err(first)}, claim key {claim or 'absent'} (ttl {claim_ttl}), "
         f"retry {retry.status_code} {err(retry)}, credential rows {rows}",
         connected and rows == "1")
    # control: the same user connects normally once Redis is healthy
    r2, code2, v2 = authorize(side, cookie, sid, f"byok-l5-ctrl-{side}")
    t2 = exchange(side, code2, v2) if code2 else None
    emit(side, "L5", "control: next connect with Redis healthy", "302 then 200",
         f"{r2.status_code} then {t2.status_code if t2 else None}", t2 is not None and t2.status_code == 200)


def l6(side, sid):
    cookie, uid = fresh_user(side, "l6")
    start_of_fresh_window(90)
    window = int(time.time() // 300)
    acl(side, "-set", "-eval", "-evalsha")
    try:
        r, _, _ = authorize(side, cookie, sid, f"byok-l6-{side}")
    finally:
        acl(side, "+@all")
    counter = redis_cli(side, "GET", f"mcp_byok_auth_code_window:{uid}:{window}")
    statuses = [authorize(side, cookie, sid, f"byok-l6-{side}")[0].status_code for _ in range(21)]
    same_window = int(time.time() // 300) == window
    ok_count = statuses.count(302)
    refusal = statuses[ok_count] if ok_count < len(statuses) else "none"
    emit(side, "L6", "authorize while Redis refuses the code write (SET) and Lua scripts, then 21 unredeemed authorizes",
         "head 20 x 302 then 429 (the declared cap); base 21 x 302",
         f"faulted authorize {r.status_code} {err(r) if r.status_code != 302 else ''}, window counter after it "
         f"{counter or 'absent'}, then {ok_count} x 302 then {refusal}, same window {same_window}",
         ok_count == (20 if side == "head" else 21))


def finished_connects(side, sid, name, scripting):
    cookie, uid = fresh_user(side, name)
    start_of_fresh_window(150)
    if not scripting:
        acl(side, "-@scripting")
    try:
        res = []
        for i in range(25):
            r, code, verifier = authorize(side, cookie, sid, f"byok-{name}-{side}-{i}")
            t = exchange(side, code, verifier) if code else None
            res.append((r.status_code, t.status_code if t else None))
    finally:
        acl(side, "+@all")
    tokens = sum(1 for _, t in res if t == 200)
    refused = [(i + 1, a) for i, (a, _) in enumerate(res) if a != 302]
    first_refused = f"first refused authorize #{refused[0][0]} with {refused[0][1]}" if refused else "no refusals"
    label = "Redis allows Lua scripting" if scripting else "Redis refuses Lua scripting (ACL -@scripting)"
    emit(side, "Lua", f"25 finished connects in one window, {label}", "25/25 tokens on both sides",
         f"{tokens}/25 tokens, {first_refused}, credential rows {credential_rows(side, uid, sid)}", tokens == 25)


def main():
    for side in SIDES:
        print(side, "admin login", post(SIDES[side] + "/login", data={"username": "admin", "password": MASTER},
                                        follow_redirects=False).status_code, flush=True)
    sids = {side: admin(side, "/v1/mcp/server", {
        "server_name": f"byokleads{RUN}", "alias": f"byokleads{RUN}", "transport": "http",
        "url": f"http://127.0.0.1:{REC[side]}/mcp", "auth_type": "api_key", "is_byok": True})["server_id"]
        for side in SIDES}
    for side in SIDES:
        l5(side, sids[side])
    for side in SIDES:
        l6(side, sids[side])
    for side in SIDES:
        finished_connects(side, sids[side], "luaon", scripting=True)
        finished_connects(side, sids[side], "luaoff", scripting=False)


main()

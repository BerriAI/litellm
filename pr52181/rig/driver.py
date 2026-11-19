"""A/B driver for the BYOK shared auth code PR. Runs on the VM against both proxies.

usage: python driver.py <phase> <out.jsonl>
phases: setup, main, chaos, wait_expiry, noredis
Every HTTP request opens a fresh connection so the kernel spreads them over the workers.
"""
import asyncio
import base64
import hashlib
import json
import os
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
STATE = os.path.expanduser("~/rig49642/state.json")
PHASE, OUT = sys.argv[1], sys.argv[2]
RUN = secrets.token_hex(3)


def emit(side, scenario, cell, expected, observed, ok, **extra):
    row = {"ts": round(time.time(), 2), "side": side, "scenario": scenario, "cell": cell,
           "expected": expected, "observed": observed, "pass": bool(ok), **extra}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(("PASS " if ok else "FAIL ") + f"{side:4} {scenario}/{cell}: {observed}", flush=True)


def fp(key):
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


class TimedOut:
    status_code = 0
    text = "client timeout after 60s"
    headers = {}
    cookies = {}
    elapsed_s = 60.0

    def json(self):
        return {"error": "client_timeout"}


def post(url, **kw):
    start = time.time()
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(url, **kw)
    except httpx.TimeoutException:
        return TimedOut()
    r.elapsed_s = round(time.time() - start, 2)
    return r


def admin(side, path, body):
    r = post(SIDES[side] + path, json=body, headers={"Authorization": f"Bearer {MASTER}"})
    assert r.status_code in (200, 201), (side, path, r.status_code, r.text[:300])
    return r.json()


def onboarding_key(ui_token):
    payload = ui_token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["key"]


def make_user(side, email, pw):
    user = admin(side, "/user/new", {"user_email": email, "user_role": "internal_user"})
    for _ in range(20):
        made = post(SIDES[side] + "/invitation/new", json={"user_id": user["user_id"]},
                    headers={"Authorization": f"Bearer {MASTER}"})
        if made.status_code == 200:
            break
        time.sleep(1)
    assert made.status_code == 200, (side, made.status_code, made.text[:200])
    invite = made.json()
    with httpx.Client(timeout=60) as client:
        got = client.get(SIDES[side] + "/onboarding/get_token", params={"invite_link": invite["id"]})
        assert got.status_code == 200, (side, got.status_code, got.text[:200])
        claimed = client.post(SIDES[side] + "/onboarding/claim_token",
                              json={"invitation_link": invite["id"], "user_id": user["user_id"], "password": pw},
                              headers={"Authorization": "Bearer " + onboarding_key(got.json()["token"])})
        assert claimed.status_code == 200, (side, claimed.status_code, claimed.text[:200])
    return user


def load_state():
    with open(STATE) as fh:
        return json.load(fh)


def authorize(side, cookie, server_id, api_key, challenge=None, method="S256", extra=None):
    verifier, ch = pkce()
    data = {"client_id": "rig-client", "redirect_uri": CALLBACK, "code_challenge": challenge or ch,
            "code_challenge_method": method, "state": secrets.token_hex(4), "server_id": server_id,
            "api_key": api_key, **(extra or {})}
    cookies = {"token": cookie} if cookie else None
    r = post(SIDES[side] + "/v1/mcp/oauth/authorize", data=data, cookies=cookies, follow_redirects=False)
    code = None
    if r.status_code == 302:
        code = parse_qs(urlparse(r.headers["location"]).query).get("code", [None])[0]
    return r, code, verifier


def exchange(side, code, verifier, client_id="rig-client"):
    return post(SIDES[side] + "/v1/mcp/oauth/token", data={
        "grant_type": "authorization_code", "code": code, "redirect_uri": CALLBACK,
        "code_verifier": verifier, "client_id": client_id})


def err(r):
    try:
        body = r.json()
    except ValueError:
        return r.text[:80]
    return body.get("error") or body.get("detail") or ("access_token" if "access_token" in body else str(body)[:80])


def tool_call(side, vkey, server_id):
    r = post(SIDES[side] + "/mcp-rest/tools/call", headers={"x-litellm-api-key": vkey},
             json={"server_id": server_id, "name": "byokecho-whoami", "arguments": {}})
    text = r.text
    got = text.split("fp=")[1][:12] if "fp=" in text else None
    return r.status_code, got, text[:200]


def redis_cli(side, *args):
    out = subprocess.run(["docker", "exec", REDIS[side], "redis-cli", *args], capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def psql(side, sql):
    out = subprocess.run(["docker", "exec", PG[side], "psql", "-U", "llmproxy", "-d", "litellm", "-tAc", sql],
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


# ---------------------------------------------------------------- setup
def setup():
    state = {}
    for side in SIDES:
        admin_login = post(SIDES[side] + "/login", data={"username": "admin", "password": MASTER}, follow_redirects=False)
        print(side, "admin login", admin_login.status_code, flush=True)
        created = post(SIDES[side] + "/v1/mcp/server", headers={"Authorization": f"Bearer {MASTER}"}, json={
            "server_name": "byokecho", "alias": "byokecho", "transport": "http",
            "url": f"http://127.0.0.1:{REC[side]}/mcp", "auth_type": "api_key", "is_byok": True,
            "description": "BYOK rig upstream"})
        if created.status_code == 400 and "already exists" in created.text:
            sid = created.text.split("server_id=")[1].split(")")[0]
        else:
            assert created.status_code in (200, 201), created.text[:300]
            sid = created.json()["server_id"]
        users = {}
        for name in ["u1", "u2", "u3", "u4", "cap", "other", "tool_a", "tool_b", "race", "pkce", "redis1",
                     "redis2", "redis3", "expiry", "ui", "size", "bad", "finish", "roll"]:
            email = f"{name}-{side}-{RUN}@rig49642.test"
            pw = f"Pw-{name}-{secrets.token_hex(6)}!9"
            user = make_user(side, email, pw)
            key = admin(side, "/key/generate", {"user_id": user["user_id"], "object_permission": {"mcp_servers": [sid]}})
            login = post(SIDES[side] + "/login", data={"username": email, "password": pw}, follow_redirects=False)
            cookie = login.cookies.get("token")
            assert cookie, (side, name, login.status_code, login.text[:200])
            users[name] = {"user_id": user["user_id"], "email": email, "password": pw, "key": key["key"], "cookie": cookie}
        state[side] = {"server_id": sid, "users": users}
        print(side, "server", sid, "users", len(users), flush=True)
    with open(STATE, "w") as fh:
        json.dump(state, fh)
    os.chmod(STATE, 0o600)


# ---------------------------------------------------------------- main scenarios
def full_flow(side, user, sid, api_key):
    r, code, verifier = authorize(side, user["cookie"], sid, api_key)
    if code is None:
        return r, None, None, None
    t = exchange(side, code, verifier)
    return r, t, code, verifier


def main():
    st = load_state()
    for side in SIDES:
        sid = st[side]["server_id"]
        U = st[side]["users"]

        # S1 cross-worker: 40 complete flows over four users, 10 each
        results = []
        first_ok = None
        for name in ["u1", "u2", "u3", "u4"]:
            for _ in range(10):
                r, t, code, verifier = full_flow(side, U[name], sid, f"byok-{name}-{side}-key")
                results.append((r.status_code, t.status_code if t else None, err(t) if t else err(r)))
                if t is not None and t.status_code == 200 and first_ok is None:
                    first_ok = (code, verifier)
        ok = sum(1 for _, ts, _ in results if ts == 200)
        bad = {}
        for _, ts, e in results:
            if ts != 200:
                bad[f"{ts} {e}"] = bad.get(f"{ts} {e}", 0) + 1
        emit(side, "cross_worker", "40 authorize+token flows, 4 workers", "40/40 token 200",
             f"{ok}/40 token 200; failures {bad}", ok == 40, authorize_statuses=sorted({a for a, _, _ in results}))

        # S2 replay of a redeemed code
        if first_ok:
            t2 = exchange(side, *first_ok)
            emit(side, "replay", "redeem the first successful code again", "400 invalid_grant",
                 f"{t2.status_code} {err(t2)}", t2.status_code == 400 and err(t2) == "invalid_grant")

        # S3 wrong verifier, then the right one
        r, code, verifier = authorize(side, U["pkce"]["cookie"], sid, f"byok-pkce-{side}-key")
        wrong = exchange(side, code, pkce()[0])
        emit(side, "pkce", "wrong code_verifier", "400 invalid_grant", f"{wrong.status_code} {err(wrong)}",
             wrong.status_code == 400 and err(wrong) == "invalid_grant")
        tries = [exchange(side, code, verifier) for _ in range(1)]
        emit(side, "pkce", "right verifier after a wrong one, one try", "200 on head",
             f"{tries[0].status_code} {err(tries[0])}", tries[0].status_code == 200 or side == "base")

        # S4 eight concurrent redemptions of one code
        r, code, verifier = authorize(side, U["race"]["cookie"], sid, f"byok-race-{side}-key")
        with ThreadPoolExecutor(8) as pool:
            racers = list(pool.map(lambda _: exchange(side, code, verifier), range(8)))
        wins = sum(1 for x in racers if x.status_code == 200)
        emit(side, "race", "8 concurrent redemptions of one code", "exactly 1 x 200",
             f"{wins} x 200, others {sorted({f'{x.status_code} {err(x)}' for x in racers if x.status_code != 200})}",
             wins == 1)

        # S5 per-user cap: 21 authorizes without redeeming, then another user
        statuses = []
        for _ in range(21):
            r, code, _ = authorize(side, U["cap"]["cookie"], sid, f"byok-cap-{side}-key")
            statuses.append(r.status_code)
        last = r
        other, _, _ = authorize(side, U["other"]["cookie"], sid, f"byok-other-{side}-key")
        emit(side, "per_user_cap", "21 unredeemed authorizes by one user", "head 20 x 302 then 429",
             f"{statuses.count(302)} x 302, then {statuses[-1]} {err(last) if statuses[-1] != 302 else ''}".strip(),
             statuses == [302] * 20 + [429] if side == "head" else statuses == [302] * 21)
        emit(side, "per_user_cap", "a second user right after", "302", f"{other.status_code}", other.status_code == 302)

        # S5b finished connects give their slot back: 25 full connects by one user, then 21 unredeemed
        finished = [full_flow(side, U["finish"], sid, f"byok-finish-{side}-key") for _ in range(25)]
        submits = [f[0].status_code for f in finished]
        tokens = sum(1 for f in finished if f[1] is not None and f[1].status_code == 200)
        emit(side, "per_user_cap", "25 finished connects by one user", "no 429, head 25/25 token 200",
             f"{submits.count(302)} x 302, {submits.count(429)} x 429, {tokens}/25 token 200",
             submits == [302] * 25 and (tokens == 25 or side == "base"))
        issued21 = [authorize(side, U["finish"]["cookie"], sid, f"byok-finish-{side}-key") for _ in range(21)]
        pend = [x[0].status_code for x in issued21]
        emit(side, "per_user_cap", "then 21 unredeemed authorizes by the same user", "head 20 x 302 then 429",
             f"{pend.count(302)} x 302, then {pend[-1]}",
             pend == [302] * 20 + [429] if side == "head" else pend == [302] * 21)
        refused10 = [authorize(side, U["finish"]["cookie"], sid, f"byok-finish-{side}-key")[0].status_code for _ in range(10)]
        redeemed20 = [exchange(side, x[1], x[2]).status_code for x in issued21[:20]]
        again = [authorize(side, U["finish"]["cookie"], sid, f"byok-finish-{side}-key")[0].status_code for _ in range(21)]
        emit(side, "per_user_cap", "10 more refused, redeem the 20, then 21 more", "head 10 x 429, 20 x 200, then 20 x 302 and 429",
             f"{refused10.count(429)} x 429, redeemed {redeemed20.count(200)} x 200, then {again.count(302)} x 302, last {again[-1]}",
             (refused10 == [429] * 10 and redeemed20 == [200] * 20 and again == [302] * 20 + [429]) if side == "head"
             else (refused10 == [302] * 10 and again == [302] * 21))

        # S6 tool calls carry each user's own key, via REST and the MCP endpoint
        for name in ["tool_a", "tool_b"]:
            key_value = f"byok-{name}-{side}-{secrets.token_hex(4)}"
            done = None
            for attempt in range(1, 21):
                r, t, _, _ = full_flow(side, U[name], sid, key_value)
                if t is not None and t.status_code == 200:
                    done = attempt
                    break
            status, got, text = tool_call(side, U[name]["key"], sid)
            emit(side, "tool_call", f"{name} REST tools/call after connect", f"fp {fp(key_value)}",
                 f"connected on try {done}; {status} fp {got}", status == 200 and got == fp(key_value),
                 attempts=done)
            st[side]["users"][name]["byok_key"] = key_value
        mcp_rows = asyncio.run(mcp_client_calls(side, st))
        for name, got, want in mcp_rows:
            emit(side, "tool_call", f"{name} MCP client tools/call", f"fp {want}", f"fp {got}", got == want)

        # S7 sizes and malformed requests
        r, _, _ = authorize(side, U["size"]["cookie"], sid, "k" * 8192)
        emit(side, "input_size", "api_key of 8192 chars", "302", f"{r.status_code}", r.status_code == 302)
        r, _, _ = authorize(side, U["size"]["cookie"], sid, "k" * 8193)
        emit(side, "input_size", "api_key of 8193 chars", "head 400, base 302", f"{r.status_code} {err(r) if r.status_code != 302 else ''}".strip(),
             r.status_code == (400 if side == "head" else 302))
        for label, extra in [("code_challenge of 129 chars", {"code_challenge": "c" * 129}),
                             ("client_id of 8193 chars", {"client_id": "c" * 8193}),
                             ("server_id of 8193 chars", {"server_id": "s" * 8193}),
                             ("redirect_uri over 8192 chars", {"redirect_uri": CALLBACK + "?pad=" + "p" * 8192})]:
            r, _, _ = authorize(side, U["size"]["cookie"], sid, "k", extra=extra)
            emit(side, "input_size", label, "head 400, base 302", f"{r.status_code} {err(r) if r.status_code != 302 else ''}".strip(),
                 r.status_code == (400 if side == "head" else 302))
        plain = [authorize(side, U["bad"]["cookie"], sid, "k", method="plain")[0].status_code for _ in range(25)]
        after, _, _ = authorize(side, U["bad"]["cookie"], sid, "k")
        emit(side, "malformed", "25 x code_challenge_method=plain then a valid one", "25 x 400 then 302",
             f"{plain.count(400)} x 400 then {after.status_code}", plain == [400] * 25 and after.status_code == 302)
        anon = [authorize(side, None, sid, "k")[0].status_code for _ in range(3)]
        emit(side, "malformed", "no session cookie", "401", f"{anon}", anon == [401] * 3)

        # S8 what the shared store and the DB hold
        if side == "head":
            r, code, _ = authorize(side, U["expiry"]["cookie"], sid, f"byok-expiry-{side}-plaintext-marker")
            keys = redis_cli(side, "--scan", "--pattern", "*mcp_byok_auth_code:*").split()
            sealed = redis_cli(side, "GET", f"mcp_byok_auth_code:{code}")
            ttl = redis_cli(side, "TTL", f"mcp_byok_auth_code:{code}")
            wkey = (redis_cli(side, "--scan", "--pattern", f"*mcp_byok_auth_code_window:{U['expiry']['user_id']}:*").split() or ["none"])[0]
            window = redis_cli(side, "GET", wkey)
            wttl = redis_cli(side, "TTL", wkey)
            emit(side, "store", "pending code record in Redis", "encrypted, TTL<=300",
                 f"{len(keys)} pending keys; record has plaintext={('plaintext-marker' in sealed)}; ttl {ttl}; user window {wkey.split(':')[-1]} = {window} ttl {wttl}",
                 "plaintext-marker" not in sealed and 0 < int(ttl) <= 300 and 0 < int(wttl) <= 300)
            st[side]["expiry_code"] = code
        for name in ["tool_a"]:
            row = psql(side, f"SELECT credential_b64 FROM \"LiteLLM_MCPUserCredentials\" WHERE user_id='{U[name]['user_id']}'")
            emit(side, "store", f"{name} credential row in Postgres", "row present, no plaintext",
                 f"rows {len(row.splitlines())}, plaintext={st[side]['users'][name]['byok_key'] in row}",
                 bool(row) and st[side]["users"][name]["byok_key"] not in row)
    with open(STATE, "w") as fh:
        json.dump(st, fh)


async def mcp_client_calls(side, st):
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    rows = []
    for name in ["tool_a", "tool_b"]:
        user = st[side]["users"][name]
        headers = {"x-litellm-api-key": user["key"]}
        async with create_mcp_http_client(headers=headers) as http_client:
            async with streamable_http_client(SIDES[side] + "/mcp/", http_client=http_client) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("byokecho-whoami", {})
                    text = " ".join(getattr(c, "text", "") for c in result.content)
                    got = text.split("fp=")[1][:12] if "fp=" in text else text[:60]
                    rows.append((name, got, fp(user["byok_key"])))
    return rows


# ---------------------------------------------------------------- chaos
def chaos():
    st = load_state()
    for side in SIDES:
        sid = st[side]["server_id"]
        U = st[side]["users"]
        # C1 Redis paused at authorize
        subprocess.run(["docker", "pause", REDIS[side]], check=True)
        try:
            r, code, verifier = authorize(side, U["redis1"]["cookie"], sid, f"byok-redis1-{side}-key")
            emit(side, "redis_outage", "authorize while Redis is paused", "head 503, base 302",
                 f"{r.status_code} {err(r) if r.status_code != 302 else ''} in {r.elapsed_s}s".strip(),
                 r.status_code == (503 if side == "head" else 302))
            with httpx.Client(timeout=30) as client:
                live = client.get(SIDES[side] + "/health/liveliness")
            emit(side, "redis_outage", "liveliness while Redis is paused", "200", f"{live.status_code}", live.status_code == 200)
        finally:
            subprocess.run(["docker", "unpause", REDIS[side]], check=True)
        time.sleep(3)
        # C2 code issued, Redis paused before the token request, then retried after
        r, code, verifier = authorize(side, U["redis2"]["cookie"], sid, f"byok-redis2-{side}-key")
        subprocess.run(["docker", "pause", REDIS[side]], check=True)
        try:
            during = exchange(side, code, verifier)
        finally:
            subprocess.run(["docker", "unpause", REDIS[side]], check=True)
        time.sleep(3)
        after = exchange(side, code, verifier)
        emit(side, "redis_outage", "token request while Redis is paused", "head 503 temporarily_unavailable",
             f"{during.status_code} {err(during)} in {during.elapsed_s}s",
             (during.status_code == 503 and err(during) == "temporarily_unavailable") if side == "head" else True)
        emit(side, "redis_outage", "same code once Redis is back", "head 200",
             f"{after.status_code} {err(after)}", after.status_code == 200 if side == "head" else True)
        # C3 Redis paused for 7s under a 20-flow burst (10 at a time), then resumed
        burst_users = [U[n] for n in ("u1", "u2", "u3", "u4")]

        def one(i):
            user = burst_users[i % 4]
            r, code, verifier = authorize(side, user["cookie"], sid, f"byok-burst-{side}")
            t = exchange(side, code, verifier) if code else None
            return r.status_code, (t.status_code if t else None), (err(t) if t else err(r)), code, verifier

        subprocess.run(["docker", "pause", REDIS[side]], check=True)
        with ThreadPoolExecutor(10) as pool:
            futs = [pool.submit(one, i) for i in range(20)]
            time.sleep(7)
            subprocess.run(["docker", "unpause", REDIS[side]], check=True)
            res = [f.result() for f in futs]
        time.sleep(3)
        tally = {}
        for a, t, e, _, _ in res:
            tally[f"{a}/{t} {e}"] = tally.get(f"{a}/{t} {e}", 0) + 1
        emit(side, "redis_outage", "20 flows, Redis paused 7s under the burst", "no 500, refusals are 503",
             f"{tally}", all(a in (302, 503) and t in (None, 200, 400, 503) for a, t, _, _, _ in res))
        retry = [exchange(side, code, v) for a, t, _, code, v in res if a == 302 and t == 503]
        rt = {}
        for x in retry:
            rt[f"{x.status_code} {err(x)}"] = rt.get(f"{x.status_code} {err(x)}", 0) + 1
        emit(side, "redis_outage", "retry every code whose token request got 503", "200, or invalid_grant if the claim landed",
             f"{len(retry)} retried: {rt}", all(x.status_code in (200, 400) for x in retry))
        recovered = [full_flow(side, U["other"], sid, f"byok-recover-{side}")[1] for _ in range(4)]
        codes = [x.status_code if x is not None else None for x in recovered]
        emit(side, "redis_outage", "4 flows after Redis is back", "head 4 x 200", f"{codes}",
             codes == [200] * 4 if side == "head" else True)


# ---------------------------------------------------------------- expiry and window reset
def wait_expiry():
    st = load_state()
    issued = {}
    for side in SIDES:
        sid = st[side]["server_id"]
        r, code, verifier = authorize(side, st[side]["users"]["expiry"]["cookie"], sid, f"byok-exp-{side}")
        issued[side] = (code, verifier)
    time.sleep(306)
    for side in SIDES:
        sid = st[side]["server_id"]
        t = exchange(side, *issued[side])
        emit(side, "expiry", "redeem a code 306s after issue", "400 invalid_grant", f"{t.status_code} {err(t)}",
             t.status_code == 400 and err(t) == "invalid_grant")
        r, _, _ = authorize(side, st[side]["users"]["cap"]["cookie"], sid, f"byok-cap-{side}-key")
        emit(side, "per_user_cap", "capped user after the 5 minute window", "302", f"{r.status_code}", r.status_code == 302)


# ---------------------------------------------------------------- store check only
def store_check():
    st = load_state()
    side = "head"
    U = st[side]["users"]
    sid = st[side]["server_id"]
    r, code, _ = authorize(side, U["expiry"]["cookie"], sid, f"byok-expiry-{side}-plaintext-marker")
    keys = redis_cli(side, "--scan", "--pattern", "*mcp_byok_auth_code:*").split()
    sealed = redis_cli(side, "GET", f"mcp_byok_auth_code:{code}")
    ttl = redis_cli(side, "TTL", f"mcp_byok_auth_code:{code}")
    wkey = (redis_cli(side, "--scan", "--pattern", f"*mcp_byok_auth_code_window:{U['expiry']['user_id']}:*").split() or ["none"])[0]
    window = redis_cli(side, "GET", wkey)
    wttl = redis_cli(side, "TTL", wkey)
    emit(side, "store", "pending code record in Redis", "encrypted, TTL<=300",
         f"{len(keys)} pending keys; record has plaintext={('plaintext-marker' in sealed)}; ttl {ttl}; user window {wkey.split(':')[-1]} = {window} ttl {wttl}",
         "plaintext-marker" not in sealed and 0 < int(ttl) <= 300 and 0 < int(wttl) <= 300)


# ---------------------------------------------------------------- window rollover
def window_rollover():
    st = load_state()
    boundary = (int(time.time()) // 300 + 1) * 300
    if boundary - time.time() < 8:
        boundary += 300
    time.sleep(max(0, boundary - 5 - time.time()))
    old = {}
    for side in SIDES:
        r, code, verifier = authorize(side, st[side]["users"]["roll"]["cookie"], st[side]["server_id"], f"byok-roll-{side}")
        old[side] = (r.status_code, code, verifier, round(time.time() - boundary, 2))
    time.sleep(max(0, boundary + 2 - time.time()))
    for side in SIDES:
        sid = st[side]["server_id"]
        cookie = st[side]["users"]["roll"]["cookie"]
        current = [authorize(side, cookie, sid, f"byok-roll-{side}")[0].status_code for _ in range(20)]
        t = exchange(side, old[side][1], old[side][2])
        last = authorize(side, cookie, sid, f"byok-roll-{side}")[0]
        emit(side, "per_user_cap", "old code issued 5s before a window boundary, redeemed after 20 new ones",
             "head old 200, then 429",
             f"old issued {old[side][0]} at {old[side][3]}s, new {current.count(302)} x 302, old redeem {t.status_code}, then {last.status_code}",
             t.status_code == 200 and current == [302] * 20 and last.status_code == (429 if side == "head" else 302))


# ---------------------------------------------------------------- no Redis
def noredis():
    for side in SIDES:
        # the admin's first dashboard login creates the admin user row that invitations reference
        print(side, "admin login", post(SIDES[side] + "/login", data={"username": "admin", "password": MASTER},
                                        follow_redirects=False).status_code, flush=True)
        sid = admin(side, "/v1/mcp/server", {
            "server_name": f"byokecho{RUN}", "alias": f"byokecho{RUN}", "transport": "http",
            "url": f"http://127.0.0.1:{REC[side]}/mcp", "auth_type": "api_key", "is_byok": True})["server_id"]
        email = f"nr-{secrets.token_hex(3)}-{side}@rig49642.test"
        pw = "Pw-" + secrets.token_hex(6) + "!9"
        make_user(side, email, pw)
        cookie = post(SIDES[side] + "/login", data={"username": email, "password": pw}, follow_redirects=False).cookies.get("token")
        oks = 0
        flows = int(os.environ.get("RIG_FLOWS", "10"))
        for _ in range(flows):
            r, code, verifier = authorize(side, cookie, sid, f"byok-nr-{side}")
            t = exchange(side, code, verifier) if code else None
            oks += int(t is not None and t.status_code == 200)
        mode = os.environ.get("RIG_MODE", "noredis")
        emit(side, "no_redis", f"{flows} flows, {mode}", "see mode", f"{oks}/{flows} token 200", True, mode=mode)


{"setup": setup, "main": main, "chaos": chaos, "wait_expiry": wait_expiry, "noredis": noredis,
 "window_rollover": window_rollover, "store_check": store_check}[PHASE]()

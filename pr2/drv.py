"""PR2 E2E driver. usage: RUN=<id> drv.py <db> <out.jsonl> <step> [step ...]. Requests go through the proxy on :4396.
A step is <name> or <name>:<deployment>. "real" modes reach Anthropic; write/hit/miss/nousage/fail are labelled relay fixtures."""
import json, os, subprocess, sys, time, uuid, urllib.request
DB, OUT, STEPS = sys.argv[1], sys.argv[2], sys.argv[3:]
P, KEY, RUN = "http://127.0.0.1:4396", os.environ["LITELLM_MASTER_KEY"], os.environ["RUN"]
DBURL = os.environ["DATABASE_URL"].rsplit("/", 1)[0] + "/" + DB
def prefix(dep): return f"[{RUN}-{dep}] " + "Reference notes for the assistant. " * 420
def req(dep, mode, surface="chat", stream=False, ttl=None, two=False):
    cc = {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}
    user = f"Reply ok. {uuid.uuid4().hex} MODE:{mode}"
    ucontent = [{"type": "text", "text": user, "cache_control": {"type": "ephemeral"}}] if two else user
    if surface == "messages":
        path, body = "/v1/messages", {"model": dep, "max_tokens": 8, "system": [{"type": "text", "text": prefix(dep), "cache_control": cc}], "messages": [{"role": "user", "content": ucontent}]}
    elif surface == "responses":
        path, body = "/v1/responses", {"model": dep, "max_output_tokens": 16, "input": [{"role": "system", "content": [{"type": "input_text", "text": prefix(dep), "cache_control": cc}]}, {"role": "user", "content": user}]}
    else:
        path, body = "/v1/chat/completions", {"model": dep, "max_tokens": 8, "messages": [{"role": "system", "content": [{"type": "text", "text": prefix(dep), "cache_control": cc}]}, {"role": "user", "content": ucontent}]}
    if stream: body["stream"] = True
    r = urllib.request.Request(P + path, json.dumps(body).encode(), {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=180) as f:
            raw = f.read().decode(); out = {"mode": mode, "status": f.status, "ms": int((time.time() - t0) * 1000)}
            if stream: return {**out, "stream": True}
            j = json.loads(raw); u = j.get("usage", {}); d = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
            return {**out, "read": d.get("cached_tokens", u.get("cache_read_input_tokens")), "write": u.get("cache_creation_input_tokens", d.get("cache_creation_tokens")), "model": j.get("model"), "keys": sorted(j)}
    except Exception as e:
        return {"mode": mode, "status": getattr(e, "code", repr(e)[:80])}
def sql(q): return subprocess.run(["psql", DBURL, "-Atc", q], capture_output=True, text=True).stdout.strip()
def rows(dep): return sql(f"select id||'|'||status||'|'||(evidence->>'consecutive_misses')||'|'||(evidence->>'proven_write_tokens')||'|'||to_char(updated_at,'HH24:MI:SS.MS') from \"LiteLLM_Alert\" where type='prompt_cache' and resource_id='{dep}' order by id").split("\n")
def api(method, path, body=None):
    r = urllib.request.Request(P + path, json.dumps(body).encode() if body else None, {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as f: return f.status, json.load(f)
    except Exception as e: return getattr(e, "code", repr(e)[:80]), None
def run(dep, modes, **kw):
    out = []
    for m in modes:
        k = dict(kw)
        if m == "stream": k["stream"] = True; m = "miss"
        r = req(dep, m, **k); out.append(r); time.sleep(0.5)
    return out
def emit(step, dep, reqs, **extra):
    time.sleep(3); rs = [x for x in rows(dep) if x]
    rec = {"step": step, "deployment": dep, "requests": [f"{r['mode']}{'(stream)' if r.get('stream') else ''}:{r['status']}:r{r.get('read')}/w{r.get('write')}" for r in reqs], "rows": rs, **extra}
    open(OUT, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec)[:900], flush=True)
SEQ = {  # step -> (modes, request kwargs)
 "real_proof": (["real", "real"], {}), "misses3": (["miss"] * 3, {}), "misses6": (["miss"] * 6, {}), "miss1": (["miss"], {}), "miss2": (["miss"] * 2, {}),
 "fx_proof": (["write"], {}), "fx_proof_2miss": (["write", "miss", "miss"], {}), "fx_incident": (["write", "hit", "miss", "miss", "miss"], {}), "fx_hit": (["hit"], {}),
 "never_proven": (["miss"] * 5, {}), "break_hit": (["write", "miss", "miss", "hit", "miss", "miss"], {}), "break_nousage": (["write", "miss", "miss", "nousage", "miss", "miss"], {}),
 "break_fail": (["write", "miss", "miss", "fail", "miss", "miss"], {}), "break_stream": (["write", "miss", "miss", "stream", "miss", "miss"], {}),
 "surface_messages": (["write", "miss", "miss", "miss", "miss"], {"surface": "messages"}), "surface_responses": (["write", "miss", "miss", "miss", "miss"], {"surface": "responses"}),
 "all_stream": (["stream"] * 5, {}), "two_breakpoints": (["write", "miss", "miss", "miss", "miss"], {"two": True}), "ttl_1h": (["write", "miss", "miss", "miss", "miss"], {"ttl": "1h"}),
 "provider_openai": (["write", "miss", "miss", "miss", "miss"], {}),
}
for s in STEPS:
    name, _, dep = s.partition(":")
    if name == "dismiss":
        rid = rows(dep)[0].split("|")[0]; st, _ = api("PATCH", f"/alerts/{rid}", {"status": "dismissed"}); emit(name, dep, [], patch_status=st)
    elif name == "show":
        st, j = api("GET", f"/alerts?type=prompt_cache"); mine = [a for a in (j or {}).get("alerts", (j or {}).get("data", [])) if a.get("resource_id") == dep]
        emit(name, dep, [], list_status=st, api_alerts=[{k: a.get(k) for k in ("id", "type", "resource_type", "status", "evidence")} for a in mine])
    else:
        modes, kw = SEQ[name]; emit(name, dep, run(dep, modes, **kw))

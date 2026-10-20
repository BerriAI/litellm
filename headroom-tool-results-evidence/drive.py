import json, os, random, sys, time, hashlib, urllib.request, urllib.error
W = os.environ["W"]; KEY = os.environ["HR_MASTER_KEY"]; OUT = f"{W}/res"; os.makedirs(OUT, exist_ok=True)
DEFAULT_PX = {"base": 4021, "tipoff": 4031, "tipon": 4032}
ALL_PX = {**DEFAULT_PX, "cand": 4032, "final": 4034}
PX = (
    {p: ALL_PX[p] for p in os.environ["PATHS"].split(",")}
    if "PATHS" in os.environ
    else DEFAULT_PX
)
CASE_FILTER = set(os.environ["CASE_FILTER"].split(",")) if "CASE_FILTER" in os.environ else None
NO_DIRECT = os.environ.get("NO_DIRECT") == "1"
MT_SOURCE = "tipon" if "tipon" in PX else next(iter(PX))
DIRECT_PREFIX = f"{next(iter(PX))}." if "PATHS" in os.environ and len(PX) == 1 else ""
CM, RM, SYS = "claude-haiku-5-5", "gpt-6.1-sol", "You are a file assistant."
def post(url, body, hdr={}):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **hdr})
    try:
        x = urllib.request.urlopen(r, timeout=300); return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:3000]
def save(n, o): json.dump(o, open(f"{OUT}/{n}.json", "w"), indent=1)
def lst(root, seed, canary=None):
    rnd = random.Random(seed)
    rows = [{"path": f"{root}/pkg{i//40}/mod_{i}.{['py','ts','md','json','yaml','lock'][i%6]}", "type": "file", "size": rnd.randint(100, 90000),
             "mtime": f"2026-10-{1+i%18:02d}T12:{i%60:02d}:00Z", "owner": "dev", "mode": "0644"} for i in range(400)]
    if canary: rows[263] = {**rows[263], "owner": canary, "size": 77777}
    return json.dumps(rows)
T = {"type": "function", "function": {"name": "list_dir", "description": "List files as JSON", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
RT = {"type": "function", **T["function"]}
QK = "Which file in /srv/app is owned by zeta-canary-7731? Reply with its exact path and size in bytes. Use the tool."
LK = lst("/srv/app", 11, "zeta-canary-7731")
INSTR = "TASK: Find the single ERROR line in the log below and reply with exactly 'ANSWER: <request id>' and nothing else."
TAIL = "FINAL CONSTRAINT: write the request id in UPPERCASE."
LOG = "\n".join("2026-10-19T12:06:57Z ERROR svc-payments upstream timeout id=req00417 latency_ms=30000" if i == 417 else
    f"2026-10-19T12:{i//60%60:02d}:{i%60:02d}Z INFO svc-{i%7} request ok id=req{i:05d} latency_ms={40+i%90}" for i in range(600))
call = lambda i, a: {"role": "assistant", "content": None, "tool_calls": [{"id": i, "type": "function", "function": {"name": "list_dir", "arguments": a}}]}
CASES = {
 "K1_chat": ("chat", {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": QK},
     call("call_k1_0001", '{"path": "/srv/app"}'), {"role": "tool", "tool_call_id": "call_k1_0001", "content": LK}]}),
 "K1_resp": ("resp", {"model": RM, "store": False, "tools": [RT], "instructions": SYS, "input": [{"role": "user", "content": QK},
     {"type": "function_call", "call_id": "call_k1_0001", "name": "list_dir", "arguments": '{"path": "/srv/app"}'},
     {"type": "function_call_output", "call_id": "call_k1_0001", "output": LK}]}),
 "L1_chat": ("chat", {"model": CM, "max_tokens": 100, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": INSTR + "\n\n" + LOG + "\n\n" + TAIL}]}),
 "M1_chat": ("chat", {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": QK},
     call("call_m1_0001", '{"path": "/srv/app"}'), {"role": "tool", "tool_call_id": "call_m1_0001", "content": [
       {"type": "text", "text": "Listing of /srv/app as JSON follows."}, {"type": "text", "text": LK, "cache_control": {"type": "ephemeral"}}]}]}),
}
def to_chat(kind, body):
    if kind == "chat": return [m for m in body["messages"]]
    out = [{"role": "system", "content": body["instructions"]}]
    for it in body["input"]:
        if it.get("type") == "function_call": out.append(call(it["call_id"], it["arguments"]))
        elif it.get("type") == "function_call_output": out.append({"role": "tool", "tool_call_id": it["call_id"], "content": it["output"]})
        else: out.append(it)
    return out
def setcase(c): open(f"{W}/case", "w").write(c)
def via(path, kind, name, body):
    setcase(f"{path}.{name}"); ep = "chat/completions" if kind == "chat" else "responses"
    st, o = post(f"http://127.0.0.1:{PX[path]}/v1/{ep}", body, {"REDACTED": f"REDACTED {KEY}"}); time.sleep(2)
    save(f"{path}.{name}.client_resp", {"status": st, "body": o}); return st, o
def direct(name, msgs, model, cfg=None):
    b = {"messages": msgs, "model": model, **({"config": cfg} if cfg else {})}
    st, o = post("http://127.0.0.1:8787/v1/compress", b); save(f"direct.{DIRECT_PREFIX}{name}", {"req": b, "status": st, "resp": o}); time.sleep(1); return o
step = sys.argv[1]
if step == "cases":
    for n, (k, b) in CASES.items():
        if CASE_FILTER is not None and n not in CASE_FILTER: continue
        for p in PX: print(n, p, via(p, k, n, b)[0], flush=True)
        if CASE_FILTER is None: direct(f"{n}.full", to_chat(k, b), b["model"])
if step == "mt":
    Qs = ["List /srv/app and tell me the largest .py file. Use the tool.", "Now list /srv/lib and tell me the largest .ts file. Use the tool.",
          "Now list /srv/web and tell me the largest .md file. Use the tool."]
    roots = ["/srv/app", "/srv/lib", "/srv/web"]
    for p in PX:
        hist = []
        for t in range(3):
            if CASE_FILTER is not None and f"MT{t + 1}" not in CASE_FILTER: continue
            hist = hist + [{"role": "user", "content": Qs[t]}, {"type": "function_call", "call_id": f"call_mt_{t}", "name": "list_dir", "arguments": json.dumps({"path": roots[t]})},
                           {"type": "function_call_output", "call_id": f"call_mt_{t}", "output": lst(roots[t], 20 + t)}]
            body = {"model": RM, "store": False, "tools": [RT], "instructions": SYS, "input": hist}
            save(f"{p}.MT{t+1}.client_req", body); st, o = via(p, "resp", f"MT{t+1}", body)
            txt = "".join(c.get("text", "") for i in (o.get("output", []) if isinstance(o, dict) else []) if i.get("type") == "message" for c in i["content"])
            print(p, t + 1, st, (o.get("usage") if isinstance(o, dict) else o), flush=True)
            hist = hist + [{"role": "assistant", "content": txt or "OK."}]
    if CASE_FILTER is None and not NO_DIRECT:
        for t in range(3):
            raw = json.load(open(f"{OUT}/{MT_SOURCE}.MT{t+1}.client_req.json"))
            msgs = to_chat("resp", raw)
            direct(f"MT{t+1}.full", msgs, RM)
            prev = json.load(open(f"{OUT}/{MT_SOURCE}.MT{t}.client_req.json")) if t else None
            direct(f"MT{t+1}.frozen", msgs, RM, {"frozen_message_count": len(to_chat('resp', prev)) if prev else 0})
            direct(f"MT{t+1}.session", msgs, RM, {"session_id": "rc-mt-session-1"})

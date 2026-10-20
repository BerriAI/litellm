import json, os, sys, time, urllib.request, urllib.error
W = os.environ["W"]; KEY = os.environ["HR_MASTER_KEY"]; OUT = f"{W}/res"; os.makedirs(OUT, exist_ok=True)
ALL = {"base": 4021, "cand": 4032, "basefo": 4022, "candfo": 4033, "final": 4034, "finalfo": 4035}
PX = {k: ALL[k] for k in os.environ.get("PATHS", "base,cand").split(",")}
CASE_FILTER = set(os.environ["CASE_FILTER"].split(",")) if "CASE_FILTER" in os.environ else None
CM, RM, SYS = "claude-haiku-5-5", "gpt-6.1-sol", "You are a file assistant."
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.py")).read())
def post(url, body, hdr={}):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **hdr})
    try:
        x = urllib.request.urlopen(r, timeout=300); raw = x.read().decode()
        return x.status, (json.loads(raw) if raw[:1] == "{" else raw[:20000])
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:3000]
def save(n, o): json.dump(o, open(f"{OUT}/{n}.json", "w"), indent=1)
def setcase(c): open(f"{W}/case", "w").write(c)
def fault(f): open(f"{W}/fault", "w").write(f)
EP = {"chat": "chat/completions", "resp": "responses", "msgs": "messages"}
def via(path, kind, name, body, hdr={}):
    setcase(f"{path}.{name}")
    st, o = post(f"http://127.0.0.1:{PX[path]}/v1/{EP[kind]}", body, {"Authorization": f"Bearer {KEY}", **hdr}); time.sleep(2)
    save(f"{path}.{name}.client_req", body); save(f"{path}.{name}.client_resp", {"status": st, "body": o}); return st, o
call = lambda i, a: {"role": "assistant", "content": None, "tool_calls": [{"id": i, "type": "function", "function": {"name": "list_dir", "arguments": a}}]}
OLDQ = "Here is yesterday's service log for reference; just acknowledge it.\n\n" + LOG
A = {"name": "list_dir", "description": "List files as JSON", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
def amsgs(tool_content):
    return [{"role": "user", "content": QK},
            {"role": "assistant", "content": [{"type": "text", "text": "Listing it."}, {"type": "tool_use", "id": "toolu_n1_0001", "name": "list_dir", "input": {"path": "/srv/app"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_n1_0001", "content": tool_content, "cache_control": {"type": "ephemeral"}}]}]
RINP = lambda out: [{"role": "user", "content": QK}, {"type": "function_call", "call_id": "call_r_0001", "name": "list_dir", "arguments": '{"path": "/srv/app"}'},
                    {"type": "function_call_output", "call_id": "call_r_0001", "output": out}]
K1 = {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": QK},
      call("call_k1_0001", '{"path": "/srv/app"}'), {"role": "tool", "tool_call_id": "call_k1_0001", "content": LK}]}
CASES = {
 "N1_msgs": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs(LK)}),
 "N2_msgs_multipart": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs(
     [{"type": "text", "text": "Listing of /srv/app as JSON follows."}, {"type": "text", "text": LK}])}),
 "R2_resp_block": ("resp", {"model": RM, "store": False, "tools": [RT], "instructions": SYS, "input": RINP([{"type": "input_text", "text": LK}])}),
 "R3_resp_multipart": ("resp", {"model": RM, "store": False, "tools": [RT], "instructions": SYS, "input": RINP(
     [{"type": "input_text", "text": "Listing of /srv/app as JSON follows."}, {"type": "input_text", "text": LK}])}),
 "O1_chat_olduser": ("chat", {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS},
     {"role": "user", "content": OLDQ}, {"role": "assistant", "content": "Acknowledged."}, {"role": "user", "content": QK},
     call("call_o1_0001", '{"path": "/srv/app"}'), {"role": "tool", "tool_call_id": "call_o1_0001", "content": LK}]}),
 "C1_chat_cacheprefix": ("chat", {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS},
     {"role": "user", "content": "List /srv/lib. Use the tool."}, call("call_c1_0000", '{"path": "/srv/lib"}'),
     {"role": "tool", "tool_call_id": "call_c1_0000", "content": lst("/srv/lib", 21)},
     {"role": "user", "content": [{"type": "text", "text": QK, "cache_control": {"type": "ephemeral"}}]},
     call("call_c1_0001", '{"path": "/srv/app"}'), {"role": "tool", "tool_call_id": "call_c1_0001", "content": LK}]}),
}
step = os.environ.get("STEP", "cases")
if step == "cases":
    fault("")
    for n, (k, b) in CASES.items():
        if CASE_FILTER is not None and n not in CASE_FILTER: continue
        for p in PX: print(n, p, via(p, k, n, b)[0], flush=True)
if step == "bypass":
    fault("")
    if CASE_FILTER is None or "B1_bypass" in CASE_FILTER:
        for p in PX: print("B1", p, via(p, "chat", "B1_bypass", K1, {"x-headroom-bypass": "true"})[0], flush=True)
if step == "stream":
    fault("")
    for p in PX:
        if CASE_FILTER is None or "S1_stream" in CASE_FILTER:
            print("S1", p, via(p, "chat", "S1_stream", {**K1, "stream": True, "stream_options": {"include_usage": True}})[0], flush=True)
        if CASE_FILTER is None or "S2_stream_resp" in CASE_FILTER:
            print("S2", p, via(p, "resp", "S2_stream_resp", {**CASES["R2_resp_block"][1], "stream": True})[0], flush=True)
if step == "faults":
    for f in ["close", "http500", "nonjson", "nomessages", "drop_row"]:
        fault(f)
        if CASE_FILTER is None or f"F_{f}" in CASE_FILTER:
            for p in PX: print("F", f, p, via(p, "chat", f"F_{f}", K1)[0], flush=True)
    fault("")

def amsgs2(block):
    return [{"role": "user", "content": QK},
            {"role": "assistant", "content": [{"type": "text", "text": "Listing it."}, {"type": "tool_use", "id": "toolu_n3_0001", "name": "list_dir", "input": {"path": "/srv/app"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_n3_0001", **block}]}]
NATIVE = {
 "N3_msgs_iserror_true": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs2(
     {"is_error": True, "content": "list_dir partially failed: permission denied on /srv/app/pkg9; partial listing follows.\n" + LK})}),
 "N4_msgs_iserror_false_block": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs2(
     {"is_error": False, "content": [{"type": "text", "text": LK, "cache_control": {"type": "ephemeral"}}]})}),
}
if step == "native":
    fault("")
    for n, (k, b) in NATIVE.items():
        if CASE_FILTER is not None and n not in CASE_FILTER: continue
        for p in PX: print(n, p, via(p, k, n, b)[0], flush=True)

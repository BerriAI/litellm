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
SR = {"type": "search_result", "source": "https://docs.example.internal/srv-app-owners", "title": "srv app ownership index",
      "content": [{"type": "text", "text": "Ownership index for /srv/app. Owners are listed in the directory listing output."}],
      "citations": {"enabled": True}}
LK2 = lst("/srv/lib", 21)
def amsgs3(results, uses):
    return [{"role": "user", "content": QK},
            {"role": "assistant", "content": [{"type": "text", "text": "Listing it."}, *uses]},
            {"role": "user", "content": results}]
U5 = {"type": "tool_use", "id": "toolu_n5_0001", "name": "list_dir", "input": {"path": "/srv/app"}}
U6 = {"type": "tool_use", "id": "toolu_n6_0002", "name": "list_dir", "input": {"path": "/srv/lib"}}
SRL = {**SR, "title": "srv app directory listing", "content": [{"type": "text", "text": LK}]}
MIXED = {"type": "tool_result", "tool_use_id": "toolu_n5_0001", "content": [SRL, SR]}
SAFE = {"type": "tool_result", "tool_use_id": "toolu_n6_0002", "content": LK2}
CACHED_CALL = {"role": "assistant", "content": [{"type": "text", "text": "Listing it.", "cache_control": {"type": "ephemeral"}}],
               "tool_calls": [{"id": "call_c2_0001", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "/srv/app"}'}}]}
CASES = {
 "N5_msgs_searchresult_alone": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs3([MIXED], [U5])}),
 "N6_msgs_searchresult_plus_safe": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs3([MIXED, SAFE], [U5, U6])}),
 "C2_chat_cached_call_fresh_result": ("chat", {"model": CM, "max_tokens": 300, "tools": [T], "messages": [{"role": "system", "content": SYS},
     {"role": "user", "content": QK}, CACHED_CALL, {"role": "tool", "tool_call_id": "call_c2_0001", "content": LK}]}),
"C3_msgs_cached_tool_use_fresh_result": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": [
     {"role": "user", "content": QK},
     {"role": "assistant", "content": [{"type": "text", "text": "Listing it."}, {**U5, "id": "toolu_c3_0001", "cache_control": {"type": "ephemeral"}}]},
     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_c3_0001", "content": LK}]}]}),
 "N7_msgs_paired_ids": ("msgs", {"model": CM, "max_tokens": 300, "system": SYS, "tools": [A], "messages": amsgs3(
     [{"type": "tool_result", "tool_use_id": "toolu.n7:0001", "content": LK}],
     [{"type": "tool_use", "id": "toolu.n7:0001", "name": "list_dir", "input": {"path": "/srv/app"}}])}),
 "N7c_chat_paired_ids": ("chat", {**K1, "messages": [
     *K1["messages"][:2],
     call("call.n7:0001", '{"path": "/srv/app"}'),
     {"role": "tool", "tool_call_id": "call.n7:0001", "content": LK},
 ]}),
}
fault("")
for n, (k, b) in CASES.items():
    if CASE_FILTER is not None and n not in CASE_FILTER: continue
    for p in PX: print(n, p, via(p, k, n, b)[0], flush=True)

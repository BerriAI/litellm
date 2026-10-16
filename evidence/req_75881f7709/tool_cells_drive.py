"""tool_policy and mcp_end_user_permission live cells: real guardrail classes, real Postgres, real Bedrock."""
import json, os, sys, time, httpx
RIG = "/private/tmp/claude-501/-Users-yucheng/7e7ca4ab-ffdd-4322-9071-fa1632b3ff42/scratchpad/rig"
PORTS = {"base": 4782, "head": 4783, "merge": 4784}
MASTER = {"Authorization": "Bearer <master key>"}
END_USER = {"x-litellm-end-user-id": "eu-github-only"}
def spec(name): return {"toolSpec": {"name": name, "description": f"{name} tool", "inputSchema": {"json": {"type": "object", "properties": {"q": {"type": "string"}}}}}}
def fn(name): return {"type": "function", "function": {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
def conv(text, *names): return {"messages": [{"role": "user", "content": [{"text": text}]}], "toolConfig": {"tools": [spec(n) for n in names]}, "inferenceConfig": {"maxTokens": 40}}
def chat(text, *names): return {"model": "claude-haiku", "messages": [{"role": "user", "content": text}], "tools": [fn(n) for n in names], "max_tokens": 40}
MCP = ("github-create_issue", "slack-post_message", "get_weather")
def conv_hist(guard=False):
    b = {"messages": [{"role": "user", "content": [{"text": "Post hi to the general channel."}]},
                      {"role": "assistant", "content": [{"text": "Posting."}, {"toolUse": {"toolUseId": "t1", "name": "slack-post_message", "input": {"q": "hi"}}}]},
                      {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "posted ok"}]}}, {"text": "Did it work? Answer in five words."}]}],
         "toolConfig": {"tools": [spec("slack-post_message")]}, "inferenceConfig": {"maxTokens": 40}}
    if guard: b["guardrailConfig"] = {"guardrailIdentifier": "164nbov72q77", "guardrailVersion": "DRAFT"}
    return b
CHAT_HIST = {"model": "claude-haiku", "max_tokens": 40, "tools": [fn("slack-post_message")], "messages": [
    {"role": "user", "content": "Post hi to the general channel."},
    {"role": "assistant", "content": "Posting.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "slack-post_message", "arguments": "{\"q\": \"hi\"}"}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "posted ok"},
    {"role": "user", "content": "Did it work? Answer in five words."}]}
CELLS = [
    ("TP1_converse_blocked_tool", "converse", "sk-lpr-toolpolicy-key", conv("Refund order 12. Answer in five words.", "refund_order", "ship_info"), {}),
    ("TP2_converse_allowed_tools", "converse", "sk-lpr-toolpolicy-key", conv("Describe the ship Aurora in five words.", "ship_info"), {}),
    ("TP3_chat_blocked_tool", "chat", "sk-lpr-toolpolicy-key", chat("Refund order 12. Answer in five words.", "refund_order", "ship_info"), {}),
    ("M1_converse_mcp_filtered", "converse", "sk-lpr-mcpeu-key", conv("Say hi in two words.", *MCP), END_USER),
    ("M2_converse_mcp_all_removed", "converse", "sk-lpr-mcpeu-key", conv("Say hi in two words.", "slack-post_message"), END_USER),
    ("M3_chat_mcp_filtered", "chat", "sk-lpr-mcpeu-key", chat("Say hi in two words.", *MCP), END_USER),
    ("M5_converse_mcp_all_removed_history", "converse", "sk-lpr-mcpeu-key", conv_hist(), END_USER),
    ("M6_converse_mcp_all_removed_history_guarded", "converse", "sk-lpr-mcpeu-key", conv_hist(True), END_USER),
    ("M7_chat_mcp_all_removed_history", "chat", "sk-lpr-mcpeu-key", CHAT_HIST, END_USER),
    ("M4_converse_mcp_no_end_user", "converse", "sk-lpr-mcpeu-key", conv("Say hi in two words.", *MCP), {}),
]
trees = sys.argv[1].split(","); label = sys.argv[2]
for t in trees:
    base = f"http://127.0.0.1:{PORTS[t]}"
    for key, g in (("sk-lpr-toolpolicy-key", "tool-policy"), ("sk-lpr-mcpeu-key", "mcp-end-user")):
        r = httpx.post(f"{base}/key/generate", headers=MASTER, json={"key": key, "key_alias": key, "guardrails": [g], "models": ["claude-haiku"]}, timeout=60)
        if r.status_code != 200 and "already exists" not in r.text and "Unique constraint" not in r.text: print(t, "keygen", r.status_code, r.text[:200])
    r = httpx.post(f"{base}/v1/tool/policy", headers=MASTER, json={"tool_name": "refund_order", "input_policy": "blocked"}, timeout=60); print(t, "policy", r.status_code, r.text[:120])
    r = httpx.post(f"{base}/customer/new", headers=MASTER, json={"user_id": "eu-github-only", "object_permission": {"mcp_servers": ["github"]}}, timeout=60)
    if r.status_code != 200 and "already exists" not in r.text: print(t, "customer", r.status_code, r.text[:200])
time.sleep(25)
def lines(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def tool_names(body):
    try: b = json.loads(body) if isinstance(body, str) else body
    except Exception: return None
    return [x.get("toolSpec", {}).get("name") for x in (b.get("toolConfig") or {}).get("tools", [])]
out = {}
for name, kind, key, body, hdrs in CELLS:
    for t in trees:
        wp = f"{RIG}/out/wire-bedrock-{t}.jsonl"; n0 = len(lines(wp))
        path = "/bedrock/model/claude-haiku/converse" if kind == "converse" else "/v1/chat/completions"
        r = httpx.post(f"http://127.0.0.1:{PORTS[t]}{path}", headers={"Authorization": f"Bearer {key}", **hdrs}, json=body, timeout=120)
        time.sleep(1)
        sent = lines(wp)[n0:]
        names = [tool_names(w.get("body")) for w in sent]
        def shape(body):
            try: b = json.loads(body) if isinstance(body, str) else body
            except Exception: return None
            return {"toolConfig": "toolConfig" in b, "messages": b.get("messages")}
        shapes = [shape(w.get("body")) for w in sent]
        out.setdefault(name, {})[t] = {"status": r.status_code, "call_id": r.headers.get("x-litellm-call-id"), "body": r.text[:400], "bedrock_calls": len(sent), "bedrock_tools": names, "bedrock_shape": shapes}
        print(f"{name:30s} {t:5s} st={r.status_code} bedrock={len(sent)} tools={names} toolConfig={[x and x['toolConfig'] for x in shapes]} {r.text[:110] if r.status_code != 200 else ''}", flush=True)
json.dump(out, open(f"{RIG}/tp/results-{label}.json", "w"), indent=1)

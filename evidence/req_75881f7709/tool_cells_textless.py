"""tp/textless.py <trees> <label>: Bugbot 4291211874 check. Image-only user turn with tools: does tool_policy / mcp_end_user_permission act, on Converse and on chat completions."""
import json, os, sys, time, httpx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fixtures import REQ
RIG = "/private/tmp/claude-501/-Users-yucheng/7e7ca4ab-ffdd-4322-9071-fa1632b3ff42/scratchpad/rig"
PORTS = {"base": 4782, "head": 4783, "merge": 4784}
END_USER = {"x-litellm-end-user-id": "eu-github-only"}
def find(o):
    if isinstance(o, dict):
        if "image" in o: return o["image"]
        for v in o.values():
            if (r := find(v)): return r
    if isinstance(o, list):
        for v in o:
            if (r := find(v)): return r
IMG = find(REQ["toolresult_text_image_card"])
def spec(n): return {"toolSpec": {"name": n, "description": f"{n} tool", "inputSchema": {"json": {"type": "object", "properties": {"q": {"type": "string"}}}}}}
def fn(n): return {"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
def conv(*names, text=None):
    content = [{"image": IMG}] + ([{"text": text}] if text else [])
    return {"messages": [{"role": "user", "content": content}], "toolConfig": {"tools": [spec(n) for n in names]}, "inferenceConfig": {"maxTokens": 30}}
def chat(*names, text=None):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{IMG['source']['bytes']}"}}] + ([{"type": "text", "text": text}] if text else [])
    return {"model": "claude-haiku", "messages": [{"role": "user", "content": content}], "tools": [fn(n) for n in names], "max_tokens": 30}
MCP = ("github-create_issue", "slack-post_message", "get_weather")
CELLS = [
    ("X1_converse_image_only_blocked_tool", "converse", "sk-lpr-toolpolicy-key", conv("refund_order", "ship_info"), {}),
    ("X2_chat_image_only_blocked_tool", "chat", "sk-lpr-toolpolicy-key", chat("refund_order", "ship_info"), {}),
    ("X3_converse_image_only_mcp", "converse", "sk-lpr-mcpeu-key", conv(*MCP), END_USER),
    ("X4_chat_image_only_mcp", "chat", "sk-lpr-mcpeu-key", chat(*MCP), END_USER),
    ("X5_converse_image_text_blocked_tool", "converse", "sk-lpr-toolpolicy-key", conv("refund_order", "ship_info", text="What is this? Five words."), {}),
    ("X6_chat_image_text_blocked_tool", "chat", "sk-lpr-toolpolicy-key", chat("refund_order", "ship_info", text="What is this? Five words."), {}),
]
def lines(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def names(w):
    b = w.get("body")
    try: b = json.loads(b) if isinstance(b, str) else b
    except Exception: return None
    if "toolConfig" in b: return [x.get("toolSpec", {}).get("name") for x in b["toolConfig"].get("tools", [])]
    return [t.get("name") for t in b.get("tools", [])] if "tools" in b else []
trees = sys.argv[1].split(","); out = {}
for name, kind, key, body, hdrs in CELLS:
    for t in trees:
        wp = f"{RIG}/out/wire-bedrock-{t}.jsonl"; n0 = len(lines(wp))
        path = "/bedrock/model/claude-haiku/converse" if kind == "converse" else "/v1/chat/completions"
        r = httpx.post(f"http://127.0.0.1:{PORTS[t]}{path}", headers={"Authorization": f"Bearer {key}", **hdrs}, json=body, timeout=120)
        time.sleep(1); sent = lines(wp)[n0:]
        out[f"{t}:{name}"] = {"status": r.status_code, "bedrock": len(sent), "tools": [names(w) for w in sent], "reply": r.text[:200]}
        print(f"{name:40s} {t:5s} st={r.status_code} bedrock={len(sent)} tools={[names(w) for w in sent]} {r.text[:110] if r.status_code != 200 else ''}", flush=True)
json.dump(out, open(f"{RIG}/tp/textless-{sys.argv[2]}.json", "w"), indent=1)

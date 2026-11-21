# Drives a live LiteLLM proxy for issue 49929: cache miss then hit on /v1/chat/completions (plain and stream)
# and /v1/responses, against the offline mock (tool_use + end_turn) and real Anthropic.
# Seeded cases rewrite the stored Redis cache entry between the miss and the hit.
# usage: proxy_driver.py <proxy url>   env: LITELLM_MASTER_KEY
import json, os, sys, time, uuid
import httpx, redis
from readiness_poll import poll

PROXY = sys.argv[1]
H = {"Authorization": "Bearer " + os.environ["LITELLM_MASTER_KEY"]}
R = redis.Redis(host="127.0.0.1", port=6390)
MISSING = "<missing>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
C = httpx.Client(timeout=120)

def chat(model, prompt, stream, tools=None):
    body = {"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}], "stream": stream}
    if tools:
        body["tools"] = tools
    if not stream:
        r = C.post(PROXY + "/v1/chat/completions", headers=H, json=body)
        if r.status_code != 200:
            return {"http": r.status_code, "error": r.text[:300]}
        ch = r.json()["choices"][0]
        return {"finish_reason": ch["finish_reason"], "tool_calls": bool(ch["message"].get("tool_calls")), "id": r.json()["id"],
                "cache_key_header": bool(r.headers.get("x-litellm-cache-key"))}
    reasons, tool, rid = [], False, None
    with C.stream("POST", PROXY + "/v1/chat/completions", headers=H, json=body) as r:
        if r.status_code != 200:
            return {"http": r.status_code, "error": r.read().decode()[:300]}
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            rid = rid or json.loads(line[6:]).get("id")
            for ch in json.loads(line[6:]).get("choices", []):
                tool = tool or bool(ch.get("delta", {}).get("tool_calls"))
                if ch.get("finish_reason"):
                    reasons.append(ch["finish_reason"])
    return {"finish_reasons_seen": reasons, "tool_calls": tool, "id": rid}

def responses(model, prompt):
    r = C.post(PROXY + "/v1/responses", headers=H, json={"model": model, "input": prompt, "max_output_tokens": 200})
    if r.status_code != 200:
        return {"http": r.status_code, "error": r.text[:300]}
    d = r.json()
    return {"status": d.get("status"), "output_types": [o.get("type") for o in d.get("output", [])],
            "incomplete_details": d.get("incomplete_details")}

def mock_hits(nonce):
    return C.get("http://127.0.0.1:8791/hits").json().get(nonce, 0)

def stored(keys):
    out = []
    for k in keys:
        try:
            v = json.loads(R.get(k))
            if not isinstance(v, dict):
                continue
            p = json.loads(v["response"]) if isinstance(v.get("response"), str) else v.get("response", v)
            out.append(p["choices"][0].get("finish_reason", MISSING) if "choices" in p else "<not chat-shaped>")
        except Exception as e:
            out.append("<unreadable %s>" % type(e).__name__)
    return out

def seed(keys, finish_reason, finish_details):
    for k in keys:
        v = json.loads(R.get(k))
        if not isinstance(v, dict) or not isinstance(v.get("response"), str):
            continue
        p = json.loads(v["response"])
        if "choices" not in p:
            continue
        ch = p["choices"][0]
        if finish_reason == MISSING:
            ch.pop("finish_reason", None)
        else:
            ch["finish_reason"] = finish_reason
        if finish_details is not None:
            ch["finish_details"] = finish_details
        v["response"] = json.dumps(p)
        R.set(k, json.dumps(v), keepttl=True)

def response_keys(before):
    out = []
    for k in R.keys("*"):
        if k in before or R.type(k) != b"string":
            continue
        try:
            v = json.loads(R.get(k))
        except Exception:
            continue
        if isinstance(v, dict) and "response" in v:
            out.append(k)
    return out

def assert_hit(name, miss, hit, nonce):
    if nonce:
        assert mock_hits(nonce) == 1, "%s: upstream was reached %d times" % (name, mock_hits(nonce))
    for field in ("id", "response_id"):
        if field in miss:
            assert miss[field] == hit[field], "%s: second call returned a different response id" % name
    if "cache_key_header" in hit:
        assert hit["cache_key_header"], "%s: x-litellm-cache-key header missing on the second call" % name

def case(name, call, nonce=None, seeding=None):
    before = set(R.keys("*"))
    miss = call()
    keys = poll(lambda: response_keys(before), "response cache entry in Redis for " + name)
    natural = stored(keys)
    if seeding:
        seed(keys, *seeding)
    hit = call()
    assert_hit(name, miss, hit, nonce)
    row = {"case": name, "miss": miss, "stored_finish_reason": natural, "hit": hit}
    if seeding:
        row["seeded"] = {"finish_reason": seeding[0], "finish_details": seeding[1]}
    if nonce:
        row["mock_requests"] = mock_hits(nonce)
    print(json.dumps(row), flush=True)

def mock_prompt(shape):
    nonce = uuid.uuid4().hex[:8]
    return "%s|%s|weather?" % (shape, nonce), nonce

for shape in ("tool_use+end_turn", "tool_use+tool_use", "text+end_turn", "text+max_tokens"):
    for stream in (False, True):
        p, n = mock_prompt(shape)
        case("mock chat %s %s" % (shape, "stream" if stream else "plain"), lambda: chat("mock-anthropic", p, stream), n)
for stream in ((False, True) if os.environ.get("SEEDED") == "1" else ()):
    for seeding in (("stop", None), (MISSING, None), (None, None), ("tool_calls", None), ("length", None),
                    (MISSING, "stop"), (None, "stop"), (MISSING, "length"), (None, "length")):
        p, n = mock_prompt("tool_use+end_turn")
        case("mock chat seeded tool_use+end_turn %s" % ("stream" if stream else "plain"),
             lambda: chat("mock-anthropic", p, stream), n, seeding)
for shape in ("tool_use+end_turn", "text+end_turn"):
    p, n = mock_prompt(shape)
    case("mock responses %s" % shape, lambda: responses("mock-anthropic", p), n)
if os.environ.get("LIVE") == "1":
    tag = uuid.uuid4().hex[:8]
    for stream in (False, True):
        mode = "stream" if stream else "plain"
        case("real anthropic chat tool_auto " + mode, lambda: chat("anthropic-sonnet-5-5",
             "[%s] Use the get_weather tool to look up the weather in Paris." % tag, stream, TOOLS))
        case("real anthropic chat text " + mode, lambda: chat("anthropic-sonnet-5-5", "[%s] Reply with the single word: ok" % tag, stream))
    case("real anthropic responses text", lambda: responses("anthropic-sonnet-5-5", "[%s] Reply with the single word: ok" % tag))

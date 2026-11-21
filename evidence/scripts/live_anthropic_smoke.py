# Disposable fixture for issue 49929: real Anthropic compatibility smoke through the SDK cache (key read from env).
import json, uuid
import litellm
from readiness import cache_entries_ready
from litellm.caching.caching import Cache

litellm.suppress_debug_info = True
litellm.cache = Cache(type="local")
MODEL = "anthropic/claude-sonnet-5-5"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def stored(before):
    out = []
    for k, v in litellm.cache.cache.cache_dict.items():
        if k not in before:
            out.append(json.loads(v["response"])["choices"][0].get("finish_reason", "<missing>"))
    return out

def plain(kw):
    r = litellm.completion(**kw)
    return {"finish_reason": r.choices[0].finish_reason, "tool_calls": bool(r.choices[0].message.tool_calls),
            "id": r.id, "cache_hit": r._hidden_params.get("cache_hit")}

def stream(kw):
    chunks = list(litellm.completion(**kw, stream=True))
    last = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    b = litellm.stream_chunk_builder(chunks)
    return {"last_chunk": last[-1] if last else None, "builder": b.choices[0].finish_reason,
            "builder_tool_calls": bool(b.choices[0].message.tool_calls), "id": chunks[0].id}

def case(name, fn, **extra):
    kw = dict(model=MODEL, max_tokens=200, **extra)
    before = set(litellm.cache.cache.cache_dict)
    miss = fn(kw)
    cache_entries_ready(None, before)
    st = stored(before)
    hit = fn(kw)
    assert miss["id"] == hit["id"], "second call was not served from the cache (response id changed)"
    if "cache_hit" in hit:
        assert hit["cache_hit"] is True, "cache_hit marker missing on the second call"
    print(json.dumps({"case": name, "miss": miss, "stored_finish_reason": st, "hit": hit,
                      "same_id_on_hit": miss["id"] == hit["id"]}), flush=True)

n = uuid.uuid4().hex[:8]
tool_msg = [{"role": "user", "content": "[%s] What is the weather in Paris? Use the tool." % n}]
text_msg = [{"role": "user", "content": "[%s] Reply with the single word: ok" % n}]
print(json.dumps({"litellm_file": litellm.__file__, "model": MODEL}))
for label, fn in (("plain", plain), ("stream", stream)):
    case("tool_auto/" + label, fn, messages=tool_msg, tools=TOOLS)
    case("text/" + label, fn, messages=text_msg)

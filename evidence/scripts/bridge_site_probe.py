# Disposable fixture for issue 49929: which cache converter sites a Responses-bridge call reaches, and what the
# bridge site returns when the Responses cache key holds a chat-shaped dict (seeded).
import json, sys, time, uuid, asyncio
import litellm, litellm.utils
from readiness import cache_entries_ready
from adjacent_matrix import HITS, LOOP, BASE

CALLS = []
_orig = litellm.utils.convert_to_model_response_object

def _probe(*a, **kw):
    f = sys._getframe(1)
    ro = kw.get("response_object")
    if f.f_code.co_filename.endswith("caching_handler.py") and isinstance(ro, dict) and "choices" in ro:
        stored = ro["choices"][0].get("finish_reason", "<missing>")
        out = _orig(*a, **kw)
        CALLS.append({"caller": "caching_handler.py:%s:%d" % (f.f_code.co_name, f.f_lineno),
                      "call_type": f.f_locals.get("call_type"), "stored": stored,
                      "returned": out.choices[0].finish_reason})
        return out
    return _orig(*a, **kw)

litellm.utils.convert_to_model_response_object = _probe

def chat_keys(before):
    store = litellm.cache.cache.cache_dict
    chat, resp = [], []
    for k in store:
        if k in before:
            continue
        p = json.loads(store[k]["response"])
        (chat if "choices" in p else resp).append(k)
    return chat, resp

def summary(r):
    if hasattr(r, "choices"):
        return {"type": type(r).__name__, "finish_reason": r.choices[0].finish_reason}
    return {"type": type(r).__name__, "status": r.status, "output": [getattr(o, "type", None) for o in r.output]}

def run(is_async, variant, seeded_value="stop"):
    nonce = uuid.uuid4().hex[:8]
    kw = dict(model="anthropic/claude-sonnet-4-5", input="tool_use+end_turn|%s|weather?" % nonce,
              max_output_tokens=100, api_base=BASE, api_key="x")
    call = (lambda: LOOP.run_until_complete(litellm.aresponses(**kw))) if is_async else (lambda: litellm.responses(**kw))
    before = set(litellm.cache.cache.cache_dict)
    miss = summary(call())
    cache_entries_ready(LOOP, before, min_entries=2)
    hits_after_miss = HITS.get(nonce, 0)
    chat, resp = chat_keys(before)
    store = litellm.cache.cache.cache_dict
    stored_chat = [json.loads(store[k]["response"])["choices"][0].get("finish_reason") for k in chat]
    if variant == "chat_shaped_under_responses_key":
        payload = json.loads(store[chat[0]]["response"])
        if seeded_value == "<missing>":
            payload["choices"][0].pop("finish_reason", None)
        else:
            payload["choices"][0]["finish_reason"] = seeded_value
        for k in resp:
            store[k]["response"] = json.dumps(payload)
    elif variant == "responses_entry_evicted":
        for k in resp:
            del store[k]
    del CALLS[:]
    try:
        hit = summary(call())
    except Exception as e:
        hit = {"error": repr(e)[:400]}
    assert hits_after_miss == 1 and HITS.get(nonce, 0) == 1, "second call reached the provider"
    print(json.dumps({"async": is_async, "variant": variant, "seeded": seeded_value if variant.startswith("chat") else None,
                      "new_chat_shaped_entries": len(chat), "new_responses_shaped_entries": len(resp),
                      "stored_chat_finish_reason": stored_chat, "miss": miss, "hit": hit,
                      "cache_converter_calls_on_second_call": list(CALLS), "provider_hits_total": HITS.get(nonce, 0)}), flush=True)

print(json.dumps({"litellm_file": litellm.__file__}))
for is_async in (False, True):
    run(is_async, "natural")
    run(is_async, "responses_entry_evicted")
    for v in ("stop", "<missing>", None, "tool_calls", "length"):
        run(is_async, "chat_shaped_under_responses_key", v)

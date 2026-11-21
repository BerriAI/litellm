# Disposable fixture for issue 49929: seeded stored finish_reason values and the chat-shaped Responses bridge cache site.
import json, time, uuid, asyncio
import litellm
from adjacent_matrix import MODES, HITS, LOOP, BASE, args, assert_hit
from readiness import cache_entries_ready

MISSING = "<missing>"

def new_entries(before):
    store = litellm.cache.cache.cache_dict
    return [k for k in store if k not in before]

def read_fr(key):
    v = litellm.cache.cache.cache_dict[key]
    payload = json.loads(v["response"]) if isinstance(v.get("response"), str) else v["response"]
    return payload["choices"][0].get("finish_reason", MISSING) if "choices" in payload else "<not chat-shaped>"

def seed(key, value):
    v = litellm.cache.cache.cache_dict[key]
    payload = json.loads(v["response"])
    if value == MISSING:
        payload["choices"][0].pop("finish_reason", None)
    else:
        payload["choices"][0]["finish_reason"] = value
    v["response"] = json.dumps(payload)

def seeded(shape, mode, value):
    nonce = uuid.uuid4().hex[:8]
    a = args(shape, nonce)
    before = set(litellm.cache.cache.cache_dict)
    miss = MODES[mode](a)
    keys = cache_entries_ready(LOOP, before)
    natural = [read_fr(k) for k in keys]
    for k in keys:
        seed(k, value)
    hits_after_miss = HITS.get(nonce, 0)
    hit = MODES[mode](a)
    assert_hit(miss, hit, hits_after_miss, HITS.get(nonce, 0))
    print(json.dumps({"case": "seeded", "shape": shape, "mode": mode, "miss": miss, "naturally_stored": natural,
                      "seeded_finish_reason": value, "hit": hit, "provider_hits_total": HITS.get(nonce, 0)}), flush=True)

def responses_summary(r):
    out = [{"type": getattr(o, "type", None), "status": getattr(o, "status", None)} for o in r.output]
    return {"status": r.status, "output": out, "incomplete_details": str(getattr(r, "incomplete_details", None)),
            "cache_hit": (getattr(r, "_hidden_params", None) or {}).get("cache_hit")}

def responses_case(shape, is_async, value=None):
    nonce = uuid.uuid4().hex[:8]
    kw = dict(model="anthropic/claude-sonnet-4-5", input="%s|%s|weather?" % (shape, nonce), max_output_tokens=100,
              api_base=BASE, api_key="x")
    call = (lambda: LOOP.run_until_complete(litellm.aresponses(**kw))) if is_async else (lambda: litellm.responses(**kw))
    before = set(litellm.cache.cache.cache_dict)
    miss = responses_summary(call())
    keys = cache_entries_ready(LOOP, before, min_entries=2)
    natural = [read_fr(k) for k in keys]
    if value is not None:
        for k in keys:
            seed(k, value)
    hits_after_miss = HITS.get(nonce, 0)
    hit = responses_summary(call())
    assert hits_after_miss == 1 and HITS.get(nonce, 0) == 1 and hit["cache_hit"] is True, "responses call was not a cache hit"
    print(json.dumps({"case": "responses_bridge", "shape": shape, "async": is_async, "miss": miss,
                      "naturally_stored": natural, "seeded_finish_reason": value, "hit": hit,
                      "provider_hits_after_miss": hits_after_miss, "provider_hits_total": HITS.get(nonce, 0)}), flush=True)

print(json.dumps({"litellm_file": litellm.__file__}))
for mode in MODES:
    for value in (MISSING, None, "stop", "tool_calls", "length"):
        seeded("tool_use+end_turn", mode, value)
    for value in (MISSING, None, "stop", "length"):
        seeded("text+end_turn", mode, value)
for is_async in (False, True):
    for shape in ("tool_use+end_turn", "tool_use+tool_use", "text+end_turn"):
        try:
            responses_case(shape, is_async)
        except Exception as e:
            print(json.dumps({"case": "responses_bridge", "shape": shape, "async": is_async, "error": repr(e)[:500]}), flush=True)

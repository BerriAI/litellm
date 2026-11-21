# Disposable fixture for issue 49929: cached entries whose finish_reason is missing or null but carry finish_details.
import asyncio, json, time, uuid
import litellm
from adjacent_matrix import MODES, HITS, LOOP, args, assert_hit
from readiness import cache_entries_ready

MISSING = "<missing>"


def seed(key, finish_reason, finish_details):
    v = litellm.cache.cache.cache_dict[key]
    payload = json.loads(v["response"])
    choice = payload["choices"][0]
    if finish_reason == MISSING:
        choice.pop("finish_reason", None)
    else:
        choice["finish_reason"] = finish_reason
    choice["finish_details"] = finish_details
    v["response"] = json.dumps(payload)

def case(shape, mode, finish_reason, finish_details):
    nonce = uuid.uuid4().hex[:8]
    a = args(shape, nonce)
    before = set(litellm.cache.cache.cache_dict)
    miss = MODES[mode](a)
    for k in cache_entries_ready(LOOP, before):
        seed(k, finish_reason, finish_details)
    hit = MODES[mode](a)
    assert_hit(miss, hit, 1, HITS.get(nonce, 0))
    print(json.dumps({"case": "finish_details", "shape": shape, "mode": mode, "seeded_finish_reason": finish_reason,
                      "seeded_finish_details": finish_details, "hit": hit, "provider_hits_total": HITS.get(nonce, 0)}), flush=True)

for mode in MODES:
    for shape in ("tool_use+end_turn", "text+end_turn"):
        for fr in (MISSING, None, "stop", "length"):
            for fd in ("stop", "length"):
                case(shape, mode, fr, fd)

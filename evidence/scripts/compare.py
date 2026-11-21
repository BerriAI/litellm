# Prints every case line whose outcome differs between two result directories.
import json, re, sys
a, b = sys.argv[1:3]
SKIP = {"miss", "naturally_stored", "provider_hits_after_miss", "new_chat_shaped_entries", "new_responses_shaped_entries", "stored_chat_finish_reason"}
def norm(l):
    l = re.sub(r"chatcmpl-[0-9a-f-]+", "ID", l)
    l = re.sub(r"caching_handler.py:(\w+):\d+", r"caching_handler.py:\1", l)
    return l
for f in ("adjacent_matrix", "compat_seeded", "compat_finish_details", "bridge_site_probe", "live_anthropic_smoke"):
    la, lb = [[norm(x) for x in open("%s/%s.log" % (d, f)).read().splitlines()[1:]] for d in (a, b)]
    diffs = [(x, y) for x, y in zip(la, lb) if x != y]
    print("== %s: %d lines, %d differ%s" % (f, len(la), len(diffs), "" if len(la) == len(lb) else " (LENGTH MISMATCH %d vs %d)" % (len(la), len(lb))))
    for x, y in diffs:
        dx, dy = json.loads(x), json.loads(y)
        key = {k: v for k, v in dx.items() if k not in SKIP and k not in ("hit", "cache_converter_calls_on_second_call")}
        print("  ", json.dumps(key), "\n      before:", json.dumps(dx.get("hit")), json.dumps(dx.get("miss")) if dx.get("miss") != dy.get("miss") else "", "\n      after: ", json.dumps(dy.get("hit")), json.dumps(dy.get("miss")) if dx.get("miss") != dy.get("miss") else "")

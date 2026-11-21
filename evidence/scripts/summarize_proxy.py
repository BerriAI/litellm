# One line per proxy_driver case: seeded values, stored value, miss, hit, mock request count.
import json, sys
def short(x):
    if "error" in x: return "HTTP %s" % x.get("http")
    return x.get("finish_reason") or ",".join(x.get("finish_reasons_seen", [])) or "%s %s" % (x.get("status"), x.get("output_types"))
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except Exception: print(l.rstrip()[:200]); continue
    s = d.get("seeded")
    print("%-46s seeded=%-34s stored=%-16s miss=%-34s hit=%-34s mock=%s" % (d["case"], "%s/%s" % (s["finish_reason"], s["finish_details"]) if s else "-", ",".join(map(str, d["stored_finish_reason"])), short(d["miss"]), short(d["hit"]), d.get("mock_requests", "-")))

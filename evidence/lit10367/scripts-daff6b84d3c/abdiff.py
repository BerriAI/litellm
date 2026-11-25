import json, re, sys
from collections import Counter
from pathlib import Path
home = Path.home() / "lit10367/live"
a, b = (json.loads((home / f"ab-{name}/result.json").read_text()) for name in sys.argv[1:3])
index = lambda doc: {(row["config"], row["cell"]): row for row in doc["rows"]}
A, B = index(a), index(b)
print("cells", len(A), len(B), "only in one:", sorted(set(A) ^ set(B))[:6])
DB = ("postgres", "redis", "batch_write")
is_db = lambda span: span[0].startswith(DB)
kinds = Counter()
for key in sorted(set(A) & set(B)):
    left, right = A[key], B[key]
    for field in ("status", "response_headers", "spend"):
        if left[field] != right[field]:
            kinds[field] += 1
            print("DIFF", key, field, str(left[field])[:300], "|", str(right[field])[:300])
    l = Counter(json.dumps(s) for s in left["spans"] if not is_db(s))
    r = Counter(json.dumps(s) for s in right["spans"] if not is_db(s))
    if l != r:
        kinds["spans"] += 1
        print("DIFF", key, "spans only left:", [x[:400] for x in (l - r)][:3], "only right:", [x[:400] for x in (r - l)][:3])
db_keys = lambda doc: {(row["config"], s[0], tuple(s[3]), s[4], s[5]) for row in doc["rows"] for s in row["spans"] if is_db(s)}
print("db span shapes only left:", sorted(map(str, db_keys(a) - db_keys(b)))[:5])
print("db span shapes only right:", sorted(map(str, db_keys(b) - db_keys(a)))[:5])
print("diff kinds", dict(kinds))
def scrub(value):
    text = json.dumps(value)
    text = re.sub(r"t-[0-9a-f]{6}", "t-x", text)
    return re.sub(r"Must be one of.*", "Must be one of ...", text)
for key in sorted(set(a["extras"]) | set(b["extras"])):
    if scrub(a["extras"].get(key)) != scrub(b["extras"].get(key)):
        print("EXTRA DIFF", key, scrub(a["extras"].get(key))[:500], "|", scrub(b["extras"].get(key))[:500])
print("extras same:", sum(scrub(a["extras"].get(k)) == scrub(b["extras"].get(k)) for k in a["extras"]), "of", len(a["extras"]))

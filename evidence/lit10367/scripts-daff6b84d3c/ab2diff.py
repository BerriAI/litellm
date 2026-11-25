import json, sys
from pathlib import Path
home = Path.home() / "lit10367/live"
a, b = (json.loads((home / f"ab-{n}/sad.json").read_text()) for n in sys.argv[1:3])
same = 0
for key in sorted(set(a) | set(b)):
    if a.get(key) == b.get(key):
        same += 1
    else:
        print("DIFF", key, str(a.get(key))[:340], "|", str(b.get(key))[:340])
print("same", same, "of", len(set(a) | set(b)))
print({k: a[k] for k in a if k.startswith(("burst", "postgres", "after", "log"))})

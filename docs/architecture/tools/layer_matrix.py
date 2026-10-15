"""Count module-level imports between the layers in explanation/02.

Usage: python3 layer_matrix.py IMPORTS_JSON
IMPORTS_JSON comes from invariants/tools/layer_scan_imports.py
"""
import collections
import json
import sys

LAYERS = (
    ("types", ("litellm.types", "litellm.models", "litellm.constants", "litellm._logging", "litellm._uuid")),
    ("core_utils", ("litellm.litellm_core_utils",)),
    ("llms", ("litellm.llms",)),
    ("integrations", ("litellm.integrations",)),
    ("utils_main", ("litellm.utils", "litellm.main")),
    ("router", ("litellm.router", "litellm.router_strategy", "litellm.router_utils")),
    ("proxy", ("litellm.proxy",)),
)
ORDER = {name: i for i, (name, _) in enumerate(LAYERS)}
ORDER["integrations"] = ORDER["llms"]


def under(mod, prefix):
    return mod == prefix or mod.startswith(prefix + ".")


def layer(mod):
    return next((name for name, prefixes in LAYERS if any(under(mod, p) for p in prefixes)), None)


rows = json.load(open(sys.argv[1]))
counts = collections.Counter(
    (layer(r[0]), layer(r[1])) for r in rows if r[2] == "module" and layer(r[0]) and layer(r[1])
)
for (src, dst), n in sorted(counts.items()):
    if src != dst:
        flag = "UP" if ORDER[dst] > ORDER[src] else ""
        print(f"{src:>12} -> {dst:<12} {n:>5} {flag}")

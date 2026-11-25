"""Request bodies carrying per-request callback keys, on a proxy with OTel V2 and no Mixpanel. Status and error text per key."""
import json, sys, uuid
from pathlib import Path
label, port = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], f"bk-{label}", port]
sys.path.insert(0, str(Path.home() / "lit10367/live"))
import ab
from ab import CHAT, MASTER, api, start_postgres, start_proxy, stop, sh, PG
CELLS = {
    "mixpanel_project_token": {"mixpanel_project_token": "caller-token"},
    "mixpanel_region": {"mixpanel_region": "eu"},
    "mixpanel_region-int": {"mixpanel_region": 5},
    "mixpanel_project_token-list": {"mixpanel_project_token": ["a"]},
    "mixpanel_in_metadata": {"metadata": {"mixpanel_project_token": "caller-token"}},
    "newrelic_region": {"newrelic_region": "eu"},
    "newrelic_in_metadata": {"metadata": {"newrelic_region": "eu"}},
    "langfuse_in_metadata": {"metadata": {"langfuse_public_key": "pk-x"}},
    "langfuse_public_key": {"langfuse_public_key": "pk-x"},
    "no-callback-key": {},
}
start_postgres()
proxy = None
try:
    proxy = start_proxy("plain", "", {})
    key = api("POST", "/key/generate", {"key_alias": f"bk-{uuid.uuid4().hex[:6]}"}).json()["key"]
    out = {}
    for name, extra in CELLS.items():
        reply = api("POST", "/v1/chat/completions", {**CHAT, **extra}, key)
        out[name] = [reply.status_code, "" if reply.status_code == 200 else reply.text[:160]]
    print(json.dumps(out, indent=0), flush=True)
finally:
    if proxy is not None:
        stop(proxy)
    sh(f"docker rm -fv {PG}")

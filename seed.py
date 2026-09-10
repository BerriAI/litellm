import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
settings = dict(line.split("=", 1) for line in (root / ".env").read_text().splitlines())
base = "http://127.0.0.1:47135"

def request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Authorization": "Bearer " + settings["LITELLM_MASTER_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.status, json.load(response)

if __name__ == "__main__":
    servers = {}
    for case, fields in [
        ("auth", {"url": "http://upstream:8080/mcp", "auth_type": "none"}),
        ("url", {"url": "http://upstream:8080/wrong", "auth_type": "basic", "credentials": {"auth_value": "preview:correct"}}),
        ("headers", {"url": "http://upstream:8080/mcp", "auth_type": "none", "static_headers": {"X-Preview-Key": "wrong"}}),
    ]:
        payload = {"server_name": "preview_" + case, "alias": "preview_" + case, "transport": "http", "allow_all_keys": True, **fields}
        status, result = request("POST", "/v1/mcp/server", payload)
        servers[case] = result["server_id"]
        print(case, status, result["server_id"])
    (root / "servers.json").write_text(json.dumps(servers, indent=2))

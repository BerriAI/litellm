"""mitmdump addon: records OTLP exports to Mixpanel with a header allowlist, and forwards them."""
import base64, json, os, time
from mitmproxy import http
LOG = os.environ["MITM_LOG"]
ALLOWED_HEADERS = ("content-type", "content-encoding", "user-agent")
def write(flow, status, body):
    if "mixpanel.com" not in flow.request.pretty_host:
        return
    record = {
        "ts": time.time(),
        "host": flow.request.pretty_host,
        "path": flow.request.path,
        "headers": {name: flow.request.headers.get(name) for name in ALLOWED_HEADERS if name in flow.request.headers},
        "has_basic_auth": flow.request.headers.get("authorization", "").lower().startswith("basic "),
        "body_b64": base64.b64encode(flow.request.raw_content or b"").decode(),
        "status": status,
        "resp": (body or b"")[:300].decode("utf-8", "replace"),
    }
    with open(LOG, "a") as out:
        out.write(json.dumps(record) + "\n")
def response(flow: http.HTTPFlow) -> None:
    write(flow, flow.response.status_code, flow.response.raw_content)
def error(flow: http.HTTPFlow) -> None:
    write(flow, 0, str(flow.error).encode())

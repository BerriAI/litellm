"""mitmdump addon: records OTLP exports to Mixpanel (header allowlist) and forwards them, unless a mode file says
the destination is down (answer 503 without forwarding) or slow (wait 4 s, then forward)."""
import asyncio, base64, json, os, time
from mitmproxy import http
LOG = os.environ["MITM_LOG"]
MODE_FILE = os.environ["MITM_MODE_FILE"]
ALLOWED_HEADERS = ("content-type", "content-encoding", "user-agent")
def mode():
    try:
        with open(MODE_FILE) as handle:
            return handle.read().strip()
    except OSError:
        return "up"
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
        "mode": flow.metadata.get("mode", "up"),
        "resp": (body or b"")[:300].decode("utf-8", "replace"),
    }
    with open(LOG, "a") as out:
        out.write(json.dumps(record) + "\n")
async def request(flow: http.HTTPFlow) -> None:
    if "mixpanel.com" not in flow.request.pretty_host:
        return
    current = mode()
    flow.metadata["mode"] = current
    if current == "down":
        flow.response = http.Response.make(503, b"destination down", {"content-type": "text/plain"})
    elif current == "slow":
        await asyncio.sleep(4)
def response(flow: http.HTTPFlow) -> None:
    write(flow, flow.response.status_code, flow.response.raw_content)
def error(flow: http.HTTPFlow) -> None:
    write(flow, 0, str(flow.error).encode())

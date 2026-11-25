"""Background response read by another key: which caller id does Mixpanel's copy of the creator's bill carry?
Real OpenAI Responses (background mode) through the OpenAI pass-through, real Mixpanel ingest behind a recorder."""
import base64, gzip, json, os, signal, socket, subprocess, sys, time, uuid
from pathlib import Path
import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

LABEL, PORT = sys.argv[1], int(sys.argv[2])
TOOLS = Path.home() / "lit10367/live"
OUT = TOOLS / LABEL
OUT.mkdir(parents=True, exist_ok=True)
MASTER = "sk-lit10367-bg-master"
PG = f"lit10367-bg-pg-{LABEL}"
PG_PORT, MITM_PORT, PROXY_PORT = PORT, PORT + 1, PORT + 2
TOKEN = os.environ["MIXPANEL_TEST_TOKEN"]
DB = f"postgresql://postgres:postgres@127.0.0.1:{PG_PORT}/litellm"
BASE = f"http://127.0.0.1:{PROXY_PORT}"
MODE = Path(f"/dev/shm/lit10367-mp-mode-{LABEL}")
CONFIG = """model_list:
  - model_name: gpt-4o-mini
    litellm_params: {model: openai/gpt-4o-mini, api_key: os.environ/OPENAI_API_KEY}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
litellm_settings:
  callbacks: ["mixpanel"]
"""
def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)
def listening(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0
def api(method, path, body=None, key=MASTER, headers=None):
    return httpx.request(method, f"{BASE}{path}", headers={"Authorization": f"Bearer {key}", **(headers or {})}, json=body, timeout=120, trust_env=False)
def main():
    sh(f"docker rm -fv {PG}")
    assert sh(f"docker run -d --name {PG} -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=litellm -p 127.0.0.1:{PG_PORT}:5432 postgres:16").returncode == 0
    for _ in range(90):
        if sh(f"docker exec {PG} pg_isready -U postgres -d litellm").returncode == 0:
            break
        time.sleep(1)
    time.sleep(2)
    assert sh(f"DATABASE_URL={DB} .venv/bin/prisma db push --schema litellm/proxy/schema.prisma --skip-generate --accept-data-loss").returncode == 0
    wire = OUT / "wire.jsonl"
    wire.write_text("")
    MODE.write_text("up")
    mitm_env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_LICENSE", "MIXPANEL_TEST_TOKEN")}
    mitm_env.update({"MITM_LOG": str(wire), "MITM_MODE_FILE": str(MODE)})
    mitm = subprocess.Popen([str(TOOLS / "mitm-venv/bin/mitmdump"), "-q", "-p", str(MITM_PORT), "--listen-host", "127.0.0.1", "--set", f"confdir={TOOLS / 'mitm-conf'}", "-s", str(TOOLS / "addon_chaos.py")], env=mitm_env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while not listening(MITM_PORT):
        time.sleep(0.5)
    config = OUT / "bg.yaml"
    config.write_text(CONFIG)
    env = {k: v for k, v in os.environ.items() if k not in ("HTTPS_PROXY", "HTTP_PROXY", "MIXPANEL_TEST_TOKEN")}
    env.update({"DATABASE_URL": DB, "LITELLM_MASTER_KEY": MASTER, "LITELLM_SALT_KEY": "lit10367-bg-salt-0123456789abcdef", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_OTEL_V2": "true", "OTEL_BSP_SCHEDULE_DELAY": "500", "LITELLM_DISABLE_NO_REDIS_WARNING": "true", "MIXPANEL_PROJECT_TOKEN": TOKEN, "HTTPS_PROXY": f"http://127.0.0.1:{MITM_PORT}", "REQUESTS_CA_BUNDLE": str(TOOLS / "mitm-conf/mitmproxy-ca-cert.pem"), "NO_PROXY": "127.0.0.1,localhost,api.openai.com,license.litellm.ai,raw.githubusercontent.com", "PYTHONPATH": os.getcwd()})
    log = OUT / "bg-proxy.log"
    proxy = subprocess.Popen([".venv/bin/python", "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(PROXY_PORT), "--num_workers", "1"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(300):
            try:
                if httpx.get(f"{BASE}/health/liveliness", timeout=2, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        key_a = api("POST", "/key/generate", {"key_alias": "creator-a"}).json()["key"]
        key_b = api("POST", "/key/generate", {"key_alias": "reader-b"}).json()["key"]
        cells = []
        variants = (
            ("same-key/customer-header", key_a, lambda who: {"x-litellm-customer-id": who}),
            ("same-key/baggage", key_a, lambda who: {"baggage": f"user.id={who}"}),
            ("same-key/no-id", key_a, lambda who: {}),
            ("master-key/customer-header", MASTER, lambda who: {"x-litellm-customer-id": who}),
        )
        for variant, reader_key, reader_headers in variants:
            alice, bob = f"alice-{uuid.uuid4().hex[:6]}", f"bob-{uuid.uuid4().hex[:6]}"
            trace_a, trace_b = uuid.uuid4().hex, uuid.uuid4().hex
            created = api("POST", "/openai_passthrough/v1/responses", {"model": "gpt-4o-mini", "input": "Reply with ok.", "max_output_tokens": 16, "background": True, "user": alice}, key_a, {"traceparent": f"00-{trace_a}-{uuid.uuid4().hex[:16]}-01"})
            response_id = created.json().get("id")
            print(variant, "create", created.status_code, created.json().get("status"), flush=True)
            time.sleep(2.5)
            read = api("GET", f"/openai_passthrough/v1/responses/{response_id}", None, reader_key, {"traceparent": f"00-{trace_b}-{uuid.uuid4().hex[:16]}-01", **reader_headers(bob)})
            print(variant, "read", read.status_code, read.json().get("status"), flush=True)
            cells.append({"variant": variant, "alice": alice, "bob": bob, "trace_a": trace_a, "trace_b": trace_b})
            time.sleep(8)
        time.sleep(10)
    finally:
        os.killpg(proxy.pid, signal.SIGKILL)
        os.killpg(mitm.pid, signal.SIGKILL)
        sh(f"docker rm -fv {PG}")
        MODE.unlink(missing_ok=True)
    def value(any_value):
        kind = any_value.WhichOneof("value")
        return getattr(any_value, kind) if kind in ("string_value", "int_value", "bool_value", "double_value") else None
    spans = []
    for line in open(wire):
        record = json.loads(line)
        raw = base64.b64decode(record["body_b64"])
        if record["headers"].get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        request = ExportTraceServiceRequest()
        request.ParseFromString(raw)
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    attributes = {item.key: value(item.value) for item in span.attributes}
                    spans.append({"trace": span.trace_id.hex(), "name": span.name, "attrs": attributes})
    for cell in cells:
        print("==", cell["variant"], "alice", cell["alice"], "bob", cell["bob"], flush=True)
        for side in ("trace_a", "trace_b"):
            for span in [s for s in spans if s["trace"] == cell[side]]:
                attrs = span["attrs"]
                keep = {k: v for k, v in attrs.items() if any(t in k for t in ("user", "key_hash", "api_key", "call_type", "end_user", "customer", "operation", "url", "route"))}
                print(side, span["name"][:60], json.dumps({k: v for k, v in keep.items() if k in ("user.id", "litellm.call_type", "litellm.metadata.user_api_key_alias", "litellm.metadata.user_api_key_end_user_id", "gen_ai.operation.name")}, default=str), flush=True)
    for span in spans:
        if span["trace"] in {c[k] for c in cells for k in ("trace_a", "trace_b")}:
            continue
        attrs = span["attrs"]
        if "pass_through" in str(attrs.get("litellm.call_type")) or attrs.get("gen_ai.operation.name") or attrs.get("user.id"):
            print("other-trace", span["trace"][:8], span["name"], attrs.get("litellm.call_type"), attrs.get("litellm.metadata.user_api_key_alias"), "user.id=", attrs.get("user.id"), flush=True)
    print("log tracebacks", log.read_text(errors="replace").count("Traceback"), flush=True)
main()

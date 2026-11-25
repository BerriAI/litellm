"""One side of the A/B for proxies with no Mixpanel at all. Real OpenAI and Anthropic, a local OTLP collector."""
import gzip, json, os, re, signal, socket, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

LABEL, PORT = sys.argv[1], int(sys.argv[2])
OUT = Path.home() / "lit10367/live" / f"ab-{LABEL}"
OUT.mkdir(parents=True, exist_ok=True)
MASTER = "sk-lit10367-ab-master"
PG = f"lit10367-ab-pg-{LABEL}"
PG_PORT, SINK_PORT, PROXY_PORT = PORT, PORT + 1, PORT + 2
DB = f"postgresql://postgres:postgres@127.0.0.1:{PG_PORT}/litellm"
SPANS = []
class Sink(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        if self.headers.get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        request = ExportTraceServiceRequest()
        try:
            request.ParseFromString(raw)
        except Exception:
            request = ExportTraceServiceRequest()
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    attributes = {item.key: item.value for item in span.attributes}
                    SPANS.append({"trace": span.trace_id.hex(), "name": span.name, "kind": span.kind, "status": span.status.code, "keys": sorted(attributes), "user_id": attributes["user.id"].string_value if "user.id" in attributes else None, "path": self.path})
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()
    def log_message(self, *args):
        pass
BASE = """model_list:
  - model_name: gpt-4o-mini
    litellm_params: {model: openai/gpt-4o-mini, api_key: os.environ/OPENAI_API_KEY}
  - model_name: claude-haiku
    litellm_params: {model: anthropic/claude-haiku-5-5, api_key: os.environ/ANTHROPIC_API_KEY}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  user_api_key_cache_ttl: 2
"""
def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)
def listening(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0
def start_postgres():
    sh(f"docker rm -fv {PG}")
    started = sh(f"docker run -d --name {PG} -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=litellm -p 127.0.0.1:{PG_PORT}:5432 postgres:16 -c max_connections=300")
    assert started.returncode == 0, started.stderr
    for _ in range(90):
        if sh(f"docker exec {PG} pg_isready -U postgres -d litellm").returncode == 0:
            break
        time.sleep(1)
    time.sleep(2)
    pushed = sh(f"DATABASE_URL={DB} .venv/bin/prisma db push --schema litellm/proxy/schema.prisma --skip-generate --accept-data-loss")
    assert pushed.returncode == 0, pushed.stdout + pushed.stderr
def start_proxy(name, settings, extra):
    config = OUT / f"{name}.yaml"
    config.write_text(BASE + settings)
    drop = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "MIXPANEL_PROJECT_TOKEN", "OTEL_EXPORTER", "OTEL_ENDPOINT", "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    environment = {key: value for key, value in os.environ.items() if key not in drop}
    environment.update({"DATABASE_URL": DB, "LITELLM_MASTER_KEY": MASTER, "LITELLM_SALT_KEY": "lit10367-ab-salt-0123456789abcdef", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_OTEL_V2": "true", "OTEL_BSP_SCHEDULE_DELAY": "300", "LITELLM_DISABLE_NO_REDIS_WARNING": "true", "PYTHONPATH": os.getcwd(), **extra})
    log = OUT / f"{name}-proxy.log"
    process = subprocess.Popen([".venv/bin/python", "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(PROXY_PORT), "--num_workers", "2"], env=environment, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    began = time.time()
    while time.time() - began < 420:
        if process.poll() is not None:
            break
        try:
            if httpx.get(f"http://127.0.0.1:{PROXY_PORT}/health/liveliness", timeout=2, trust_env=False).status_code == 200 and log.read_text(errors="replace").count("Started server process") >= 2:
                print(f"[{name}] up in {time.time() - began:.0f}s", flush=True)
                return process
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"[{name}] did not start: {log.read_text(errors='replace')[-1200:]}")
def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not listening(PROXY_PORT):
            return
        time.sleep(0.5)
def api(method, path, body=None, key=MASTER, headers=None):
    return httpx.request(method, f"http://127.0.0.1:{PROXY_PORT}{path}", headers={"Authorization": f"Bearer {key}", **(headers or {})}, json=body, timeout=120, trust_env=False)
def psql(query):
    return sh(f"docker exec {PG} psql -U postgres -d litellm -At -c {json.dumps(query)}").stdout.strip()
CHAT = {"model": "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}
ENDPOINTS = {
    "chat": ("/v1/chat/completions", CHAT, False),
    "chat-stream": ("/v1/chat/completions", {**CHAT, "stream": True}, True),
    "responses": ("/v1/responses", {"model": "gpt-4o-mini", "max_output_tokens": 16, "input": "Reply with ok."}, False),
    "messages": ("/v1/messages", {"model": "claude-haiku", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
}
IDS = {
    "none": ({}, {}),
    "user": ({"user": "caller-a"}, {}),
    "metadata-user-dot-id": ({"metadata": {"user.id": "caller-a"}}, {}),
    "metadata-user-id": ({"metadata": {"user_id": "caller-a"}}, {}),
    "metadata-trace-user-id": ({"metadata": {"trace_user_id": "caller-a"}}, {}),
    "metadata-json": ({"metadata": json.dumps({"user.id": "caller-a"})}, {}),
    "baggage": ({}, {"baggage": "user.id=caller-a"}),
    "end-user-header": ({}, {"x-litellm-end-user-id": "caller-a"}),
}
def call(config, cell, path, body, key, headers, stream):
    trace = uuid.uuid4().hex
    all_headers = {"Authorization": f"Bearer {key}", "traceparent": f"00-{trace}-{uuid.uuid4().hex[:16]}-01", **headers}
    with httpx.Client(trust_env=False, timeout=180) as client:
        if stream:
            with client.stream("POST", f"http://127.0.0.1:{PROXY_PORT}{path}", headers=all_headers, json=body) as reply:
                "".join(reply.iter_text())
                status, response_headers = reply.status_code, dict(reply.headers)
        else:
            reply = client.post(f"http://127.0.0.1:{PROXY_PORT}{path}", headers=all_headers, json=body)
            status, response_headers = reply.status_code, dict(reply.headers)
    return {"config": config, "cell": cell, "status": status, "trace": trace, "call_id": response_headers.get("x-litellm-call-id"), "response_headers": sorted(name for name in response_headers if name.startswith("x-litellm") and "-duration" not in name)}
def drive(config, key):
    rows = []
    for name, (path, body, stream) in ENDPOINTS.items():
        for how, (extra, headers) in (IDS.items() if name == "chat" else {"user": IDS["user"], "metadata-user-dot-id": IDS["metadata-user-dot-id"]}.items()):
            rows.append(call(config, f"{name}/{how}", path, {**body, **extra}, key, headers, stream))
    return rows
def settle(rows):
    time.sleep(10)
    ids = [row["call_id"] for row in rows if row["call_id"]]
    wanted = ",".join(f"'{value}'" for value in ids)
    for _ in range(60):
        found = psql(f"""select count(distinct litellm_call_id) from "LiteLLM_SpendLogs" where litellm_call_id in ({wanted})""")
        if found == str(len(set(ids))):
            break
        time.sleep(1)
    print("spend rows", found, "of", len(set(ids)), flush=True)
    for row in rows:
        spans = [span for span in SPANS if span["trace"] == row["trace"]]
        row["spans"] = sorted((span["name"].split(" ")[0], span["kind"], span["status"], tuple(span["keys"]), span["user_id"], span["path"]) for span in spans)
        row["spend"] = psql(f"""select coalesce(end_user,''), coalesce("user",''), call_type, spend > 0, total_tokens > 0, (select string_agg(k, ',' order by k) from jsonb_object_keys(metadata) k) from "LiteLLM_SpendLogs" where litellm_call_id = '{row['call_id']}' limit 1""")
    return rows
def main():
    server = ThreadingHTTPServer(("127.0.0.1", SINK_PORT), Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_postgres()
    results = []
    extras = {}
    try:
        process = start_proxy("plain", "", {})
        try:
            key = api("POST", "/key/generate", {"key_alias": f"plain-{uuid.uuid4().hex[:6]}"}).json()["key"]
            rows = drive("plain", key)
            for customer, field in ((f"label-{uuid.uuid4().hex[:8]}", "user.id"), (f"limited-{uuid.uuid4().hex[:8]}", "user_id")):
                assert api("POST", "/customer/new", {"user_id": customer, "rpm_limit": 1}).status_code == 200
                statuses = [api("POST", "/v1/chat/completions", {**CHAT, "metadata": {field: customer}}, key).status_code for _ in range(12)]
                extras[f"customer rpm_limit=1 via metadata.{field} has a 429"] = 429 in statuses
            team = api("POST", "/team/new", {"team_alias": f"t-{uuid.uuid4().hex[:6]}"}).json()["team_id"]
            for name, variables in (
                ("langfuse", {"langfuse_public_key": "pk-x", "langfuse_secret_key": "sk-x", "langfuse_host": "http://127.0.0.1:9"}),
                ("signoz-unknown", {"signoz_ingestion_key": "x", "signoz_foo": "y"}),
                ("mixed-unknown", {"langfuse_public_key": "pk-x", "not_a_var": "y"}),
            ):
                callback = "signoz" if name.startswith("signoz") else "langfuse_otel"
                response = api("POST", f"/team/{team}/callback", {"callback_name": callback, "callback_type": "success", "callback_vars": variables})
                extras[f"team callback {name}"] = (response.status_code, re.sub(r"[0-9a-f-]{32,}", "<id>", response.text)[:200])
            read_back = api("GET", f"/team/{team}/callback")
            extras["team callback read back"] = (read_back.status_code, re.sub(r"[0-9a-f-]{32,}", "<id>", read_back.text)[:300])
            results += settle(rows)
        finally:
            stop(process)
        process = start_proxy("otel", f'litellm_settings:\n  callbacks: ["otel"]\n  provider_url_destination_allowed_hosts: ["127.0.0.1:{SINK_PORT}"]\n', {"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": f"http://127.0.0.1:{SINK_PORT}/operator/v1/traces"})
        try:
            key = api("POST", "/key/generate", {"key_alias": f"otel-{uuid.uuid4().hex[:6]}"}).json()["key"]
            rows = drive("otel", key)
            logging = [{"callback_name": "langfuse_otel", "callback_type": "success", "callback_vars": {"langfuse_public_key": "pk-t", "langfuse_secret_key": "sk-t", "langfuse_host": f"http://127.0.0.1:{SINK_PORT}/tenant"}}]
            team = api("POST", "/team/new", {"team_alias": f"t-{uuid.uuid4().hex[:6]}", "metadata": {"logging": logging}}).json()["team_id"]
            team_key = api("POST", "/key/generate", {"team_id": team, "key_alias": f"team-{uuid.uuid4().hex[:6]}"}).json()["key"]
            rows += drive("otel+team-langfuse", team_key)
            results += settle(rows)
        finally:
            stop(process)
        process = start_proxy("langfuse", 'litellm_settings:\n  callbacks: ["langfuse_otel"]\n', {"LANGFUSE_HOST": f"http://127.0.0.1:{SINK_PORT}/operator", "LANGFUSE_PUBLIC_KEY": "pk-o", "LANGFUSE_SECRET_KEY": "sk-o"})
        try:
            key = api("POST", "/key/generate", {"key_alias": f"lf-{uuid.uuid4().hex[:6]}"}).json()["key"]
            results += settle(drive("langfuse", key))
        finally:
            stop(process)
        for name in ("plain", "otel", "langfuse"):
            text = (OUT / f"{name}-proxy.log").read_text(errors="replace")
            extras[f"{name} log tracebacks"] = text.count("Traceback")
    finally:
        sh(f"docker rm -fv {PG}")
    for row in results:
        row.pop("trace"), row.pop("call_id")
    (OUT / "result.json").write_text(json.dumps({"rows": results, "extras": extras}, indent=0, default=list))
    print("cells", len(results), "statuses", sorted({row["status"] for row in results}), "spans", sum(len(row["spans"]) for row in results), flush=True)
    print(json.dumps(extras, indent=0, default=list)[:1800], flush=True)
if __name__ == "__main__":
    main()

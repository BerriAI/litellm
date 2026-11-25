"""One live leg from the current tree: real OpenAI and Anthropic, real Mixpanel ingest behind a recorder, 4 workers."""
import base64, gzip, json, os, re, signal, socket, subprocess, sys, time, uuid
from collections import Counter
from pathlib import Path
import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

LABEL, PORT = sys.argv[1], int(sys.argv[2])
TOOLS = Path.home() / "lit10367/live"
OUT = TOOLS / LABEL
OUT.mkdir(parents=True, exist_ok=True)
MASTER = "sk-lit10367-live-master"
SALT = "lit10367-live-salt-0123456789abcdef"
PG = f"lit10367-live-pg-{LABEL}"
PG_PORT, MITM_PORT, PROXY_PORT = PORT, PORT + 1, PORT + 2
TOKEN = os.environ["MIXPANEL_TEST_TOKEN"]
DB = f"postgresql://postgres:postgres@127.0.0.1:{PG_PORT}/litellm"
BASE_CONFIG = """model_list:
  - model_name: gpt-4o-mini
    litellm_params: {model: openai/gpt-4o-mini, api_key: os.environ/OPENAI_API_KEY}
  - model_name: claude-haiku
    litellm_params: {model: anthropic/claude-haiku-5-5, api_key: os.environ/ANTHROPIC_API_KEY}
  - model_name: broken
    litellm_params: {model: openai/gpt-4o-mini, api_key: sk-not-a-real-key}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  user_api_key_cache_ttl: 2
"""
ENV_DROP = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "MIXPANEL_TEST_TOKEN", "MIXPANEL_PROJECT_TOKEN", "MIXPANEL_REGION", "MIXPANEL_ENDPOINT")
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
def start_mitm():
    log = OUT / "wire.jsonl"
    log.write_text("")
    environment = {key: value for key, value in os.environ.items() if key not in ("HTTPS_PROXY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_LICENSE", "MIXPANEL_TEST_TOKEN")}
    environment["MITM_LOG"] = str(log)
    process = subprocess.Popen(
        [str(TOOLS / "mitm-venv/bin/mitmdump"), "-q", "-p", str(MITM_PORT), "--listen-host", "127.0.0.1", "--set", f"confdir={TOOLS / 'mitm-conf'}", "-s", str(TOOLS / "addon.py")],
        env=environment, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if listening(MITM_PORT) and (TOOLS / "mitm-conf/mitmproxy-ca-cert.pem").exists():
            return process, log
        time.sleep(0.5)
    raise RuntimeError("mitmdump did not start")
def start_proxy(name, settings, extra):
    config = OUT / f"{name}.yaml"
    config.write_text(BASE_CONFIG + settings)
    environment = {key: value for key, value in os.environ.items() if key not in ENV_DROP}
    environment.update({
        "DATABASE_URL": DB, "LITELLM_MASTER_KEY": MASTER, "LITELLM_SALT_KEY": SALT, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_OTEL_V2": "true", "OTEL_BSP_SCHEDULE_DELAY": "500", "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
        "HTTPS_PROXY": f"http://127.0.0.1:{MITM_PORT}", "REQUESTS_CA_BUNDLE": str(TOOLS / "mitm-conf/mitmproxy-ca-cert.pem"),
        "NO_PROXY": "127.0.0.1,localhost,api.openai.com,api.anthropic.com,license.litellm.ai,raw.githubusercontent.com",
        "PYTHONPATH": os.getcwd(), **extra,
    })
    environment = {key: value for key, value in environment.items() if value is not None}
    log = OUT / f"{name}-proxy.log"
    process = subprocess.Popen(
        [".venv/bin/python", "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(PROXY_PORT), "--num_workers", "4"],
        env=environment, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    began = time.time()
    while time.time() - began < 420:
        if process.poll() is not None:
            break
        try:
            if httpx.get(f"http://127.0.0.1:{PROXY_PORT}/health/liveliness", timeout=2, trust_env=False).status_code == 200 and log.read_text(errors="replace").count("Started server process") >= 4:
                print(f"[{name}] up in {time.time() - began:.0f}s", flush=True)
                return process, log
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"[{name}] proxy did not start: {log.read_text(errors='replace')[-1500:]}")
def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for _ in range(30):
        if not listening(PROXY_PORT):
            return
        time.sleep(0.5)
def api(method, path, body=None, key=MASTER):
    return httpx.request(method, f"http://127.0.0.1:{PROXY_PORT}{path}", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120, trust_env=False)
CALLS = {
    "chat": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
    "chat-stream": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "Reply with ok."}]}, True),
    "responses": ("/v1/responses", {"model": "gpt-4o-mini", "max_output_tokens": 16, "input": "Reply with ok."}, False),
    "messages": ("/v1/messages", {"model": "claude-haiku", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
    "messages-stream": ("/v1/messages", {"model": "claude-haiku", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "Reply with ok."}]}, True),
}
def send(cells, cell, path, body, key, stream=False):
    trace = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {key}", "traceparent": f"00-{trace}-{uuid.uuid4().hex[:16]}-01"}
    with httpx.Client(trust_env=False, timeout=180) as client:
        if stream:
            with client.stream("POST", f"http://127.0.0.1:{PROXY_PORT}{path}", headers=headers, json=body) as reply:
                text = "".join(reply.iter_text())
                status = reply.status_code
        else:
            reply = client.post(f"http://127.0.0.1:{PROXY_PORT}{path}", headers=headers, json=body)
            text, status = reply.text, reply.status_code
    row = {"cell": cell, "trace": trace, "status": status, "path": path, "stream": stream, "caller": (body.get("user") or (body.get("metadata") or {}).get("user.id")), "body_head": text[:160]}
    cells.append(row)
    return row
def identified(body, how, caller):
    return {**body, "user": caller} if how == "user" else {**body, "metadata": {"user.id": caller}}
def drive(name, key, cells):
    for call, (path, body, stream) in CALLS.items():
        for how in ("user", "metadata-user-dot-id"):
            caller = f"lit10367-live-{LABEL}-{uuid.uuid4().hex[:10]}"
            send(cells, f"{name}/{call}/{how}", path, identified(body, how, caller), key, stream)
    caller = f"lit10367-live-{LABEL}-{uuid.uuid4().hex[:10]}"
    send(cells, f"{name}/failed-provider-call", "/v1/chat/completions", {"model": "broken", "max_tokens": 8, "user": caller, "messages": [{"role": "user", "content": "hi"}]}, key)
def team_logging():
    return [{"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": TOKEN, "mixpanel_region": "us"}}]
def decode(log):
    def value(any_value):
        kind = any_value.WhichOneof("value")
        return getattr(any_value, kind) if kind in ("string_value", "int_value", "bool_value", "double_value") else None
    spans, posts = [], []
    for line in open(log):
        record = json.loads(line)
        raw = base64.b64decode(record["body_b64"])
        if record["headers"].get("content-encoding") == "gzip":
            raw = gzip.decompress(raw)
        request = ExportTraceServiceRequest()
        request.ParseFromString(raw)
        count = 0
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    count += 1
                    attributes = {item.key: value(item.value) for item in span.attributes}
                    spans.append({"trace": span.trace_id.hex(), "span": span.span_id.hex(), "parent": span.parent_span_id.hex(), "name": span.name, "status": span.status.code, "user_id": attributes.get("user.id"), "operation": attributes.get("gen_ai.operation.name"), "ingest": record["status"]})
        posts.append({"host": record["host"], "path": record["path"], "status": record["status"], "resp": record["resp"], "spans": count, "basic": record["has_basic_auth"]})
    return spans, posts
def report(cells, spans, posts):
    by_trace = {}
    for span in spans:
        by_trace.setdefault(span["trace"], []).append(span)
    lines = []
    names_without_id = Counter()
    for cell in cells:
        found = by_trace.get(cell["trace"], [])
        auth_ids = {span["span"] for span in found if span["name"].startswith("auth ")}
        outside_auth = [span for span in found if span["parent"] not in auth_ids]
        missing = [span["name"] for span in outside_auth if span["user_id"] != cell["caller"]]
        names_without_id.update(missing)
        model = [span for span in found if span["operation"]]
        row = {"cell": cell["cell"], "http": cell["status"], "spans": len(found), "without_id": len(missing), "model_span_has_id": bool(model) and all(span["user_id"] == cell["caller"] for span in model), "ingest": sorted({span["ingest"] for span in found}), "missing_names": missing[:4]}
        lines.append(row)
        print(json.dumps(row), flush=True)
    print("cells", len(lines), "http", dict(Counter(row["http"] for row in lines)), "model span has id", sum(row["model_span_has_id"] for row in lines), "spans", sum(row["spans"] for row in lines), "spans after auth without the caller id", sum(row["without_id"] for row in lines), flush=True)
    print("span names without the caller id:", dict(names_without_id), flush=True)
    print("ingest posts by host/status:", dict(Counter((post["host"], post["status"]) for post in posts)), flush=True)
    print("ingest replies:", sorted({post["resp"][:80] for post in posts})[:5], flush=True)
    print("posts with basic auth:", sum(post["basic"] for post in posts), "of", len(posts), flush=True)
    (OUT / "report.json").write_text(json.dumps({"rows": lines, "posts": posts, "cells": cells}))
def main():
    start_postgres()
    mitm, log = start_mitm()
    cells = []
    try:
        try:
            proxy, _ = start_proxy("operator", 'litellm_settings:\n  callbacks: ["mixpanel"]\n', {"MIXPANEL_PROJECT_TOKEN": TOKEN})
        except RuntimeError as failed:
            proxy = None
            print("[operator] did not start:", str(failed)[-400:].replace(TOKEN, "<token>"), flush=True)
        if proxy is not None:
            try:
                key = api("POST", "/key/generate", {"key_alias": f"operator-{uuid.uuid4().hex[:6]}"}).json()["key"]
                drive("operator", key, cells)
                time.sleep(8)
            finally:
                stop(proxy)
        proxy, _ = start_proxy("tenant", "", {})
        try:
            team = api("POST", "/team/new", {"team_alias": f"mixpanel-{uuid.uuid4().hex[:6]}", "metadata": {"logging": team_logging()}}).json()["team_id"]
            team_key = api("POST", "/key/generate", {"team_id": team, "key_alias": f"team-{uuid.uuid4().hex[:6]}"}).json()["key"]
            own_key = api("POST", "/key/generate", {"key_alias": f"key-{uuid.uuid4().hex[:6]}", "metadata": {"logging": team_logging()}}).json()["key"]
            drive("team", team_key, cells)
            drive("key", own_key, cells)
            limited = f"lit10367-limited-{uuid.uuid4().hex[:8]}"
            label_only = f"lit10367-label-{uuid.uuid4().hex[:8]}"
            for customer in (limited, label_only):
                assert api("POST", "/customer/new", {"user_id": customer, "rpm_limit": 1}).status_code == 200
            plain = api("POST", "/key/generate", {"key_alias": f"plain-{uuid.uuid4().hex[:6]}"}).json()["key"]
            chat = CALLS["chat"][1]
            limits = {
                "metadata.user.id": [api("POST", "/v1/chat/completions", {**chat, "metadata": {"user.id": label_only}}, plain).status_code for _ in range(16)],
                "metadata.user_id": [api("POST", "/v1/chat/completions", {**chat, "metadata": {"user_id": limited}}, plain).status_code for _ in range(16)],
            }
            print("customer rpm_limit=1:", json.dumps(limits), flush=True)
            refused = {
                token: api("POST", "/key/generate", {"key_alias": f"bad-{uuid.uuid4().hex[:6]}", "metadata": {"logging": [{"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": token}}]}}).status_code
                for token in ("   ", "None", "***REDACTED***")
            }
            print("unsendable token on /key/generate:", json.dumps(refused), flush=True)
            blank_region = {
                repr(region): api("POST", "/key/generate", {"key_alias": f"region-{uuid.uuid4().hex[:6]}", "metadata": {"logging": [{"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": TOKEN, "mixpanel_region": region}}]}}).status_code
                for region in ("", "   ", "None", "eu", "mars")
            }
            print("mixpanel_region on /key/generate:", json.dumps(blank_region), flush=True)
            body_keys = {}
            for name, extra in (("mixpanel_project_token", {"mixpanel_project_token": "caller-token"}), ("mixpanel_region", {"mixpanel_region": "eu"}), ("newrelic_region", {"newrelic_region": "eu"})):
                reply = api("POST", "/v1/chat/completions", {**chat, **extra}, plain)
                body_keys[name] = (reply.status_code, "" if reply.status_code == 200 else reply.text[:140])
            print("request body key on /v1/chat/completions:", json.dumps(body_keys), flush=True)
            team_writes = {}
            for form in ("list", "json-string"):
                for token in ("", "***REDACTED***"):
                    entries_value = [{"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": token}}]
                    logging_value = entries_value if form == "list" else json.dumps(entries_value)
                    created = api("POST", "/team/new", {"team_alias": f"tw-{uuid.uuid4().hex[:6]}", "metadata": {"logging": logging_value}})
                    target = api("POST", "/team/new", {"team_alias": f"tw-{uuid.uuid4().hex[:6]}"}).json().get("team_id")
                    updated = api("POST", "/team/update", {"team_id": target, "metadata": {"logging": logging_value}})
                    team_writes[f"{form} token={token!r}"] = {"/team/new": created.status_code, "/team/update": updated.status_code}
            print("unsendable token on team writers:", json.dumps(team_writes), flush=True)
            time.sleep(8)
        finally:
            stop(proxy)
        proxy, _ = start_proxy("v2off", "", {"LITELLM_OTEL_V2": None})
        try:
            entries = {
                "mixpanel, no callback_vars": {"callback_name": "mixpanel", "callback_type": "success"},
                "mixpanel, token": {"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": TOKEN}},
                "langfuse, no callback_vars": {"callback_name": "langfuse", "callback_type": "success"},
            }
            v2off = {}
            for name, entry in entries.items():
                reply = api("POST", "/key/generate", {"key_alias": f"v2off-{uuid.uuid4().hex[:6]}", "metadata": {"logging": [entry]}})
                v2off[name] = (reply.status_code, reply.text[:160].replace(TOKEN, "<token>") if reply.status_code != 200 else "")
            print("LITELLM_OTEL_V2 off, /key/generate:", json.dumps(v2off), flush=True)
            v2off_team = {}
            for form in ("list", "json-string"):
                entries_value = [entries["mixpanel, token"]]
                logging_value = entries_value if form == "list" else json.dumps(entries_value)
                reply = api("POST", "/team/new", {"team_alias": f"v2off-{uuid.uuid4().hex[:6]}", "metadata": {"logging": logging_value}})
                v2off_team[form] = (reply.status_code, reply.text[:160].replace(TOKEN, "<token>") if reply.status_code != 200 else "")
            print("LITELLM_OTEL_V2 off, /team/new mixpanel with token:", json.dumps(v2off_team), flush=True)
        finally:
            stop(proxy)
        for name in ("operator", "tenant"):
            if not (OUT / f"{name}-proxy.log").exists():
                continue
            text = (OUT / f"{name}-proxy.log").read_text(errors="replace")
            print(f"[{name}] log: MIXPANEL_PROJECT_TOKEN-not-found={text.count('MIXPANEL_PROJECT_TOKEN not found')} Traceback={text.count('Traceback')} ", flush=True)
    finally:
        os.killpg(mitm.pid, signal.SIGKILL)
        sh(f"docker rm -fv {PG}")
    spans, posts = decode(log)
    report(cells, spans, posts)
main()

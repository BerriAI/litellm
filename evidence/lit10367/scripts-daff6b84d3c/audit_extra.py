"""Audit legs on the head: real OpenAI and Anthropic SDK clients, a team region change while traffic flows, and
chaos against Mixpanel (outage mid burst, slow destination, a killed worker, a restart mid burst). Real providers,
real Mixpanel ingest behind a recorder that can answer 503 or stall, real Postgres, 4 workers."""
import asyncio, base64, gzip, json, os, signal, socket, statistics, subprocess, sys, threading, time, uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

LABEL, PORT = sys.argv[1], int(sys.argv[2])
TOOLS = Path.home() / "lit10367/live"
OUT = TOOLS / LABEL
OUT.mkdir(parents=True, exist_ok=True)
MASTER = "sk-lit10367-audit-master"
SALT = "lit10367-audit-salt-0123456789abcdef"
PG = f"lit10367-audit-pg-{LABEL}"
PG_PORT, MITM_PORT, PROXY_PORT = PORT, PORT + 1, PORT + 2
TOKEN = os.environ["MIXPANEL_TEST_TOKEN"]
DB = f"postgresql://postgres:postgres@127.0.0.1:{PG_PORT}/litellm"
MODE = Path(f"/dev/shm/lit10367-mp-mode-{LABEL}")
BASE = f"http://127.0.0.1:{PROXY_PORT}"
BASE_CONFIG = """model_list:
  - model_name: gpt-4o-mini
    litellm_params: {model: openai/gpt-4o-mini, api_key: os.environ/OPENAI_API_KEY}
  - model_name: claude-haiku
    litellm_params: {model: anthropic/claude-haiku-5-5, api_key: os.environ/ANTHROPIC_API_KEY}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  user_api_key_cache_ttl: 2
litellm_settings:
  callbacks: ["mixpanel"]
"""
ENV_DROP = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "MIXPANEL_TEST_TOKEN", "MIXPANEL_PROJECT_TOKEN", "MIXPANEL_REGION", "MIXPANEL_ENDPOINT")
REPORT = {}
def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)
def listening(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0
def say(*parts):
    print(*parts, flush=True)
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
    environment.update({"MITM_LOG": str(log), "MITM_MODE_FILE": str(MODE)})
    process = subprocess.Popen(
        [str(TOOLS / "mitm-venv/bin/mitmdump"), "-q", "-p", str(MITM_PORT), "--listen-host", "127.0.0.1", "--set", f"confdir={TOOLS / 'mitm-conf'}", "-s", str(TOOLS / "addon_chaos.py")],
        env=environment, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if listening(MITM_PORT):
            return process, log
        time.sleep(0.5)
    raise RuntimeError("mitmdump did not start")
def start_proxy(name):
    config = OUT / f"{name}.yaml"
    config.write_text(BASE_CONFIG)
    environment = {key: value for key, value in os.environ.items() if key not in ENV_DROP}
    environment.update({
        "DATABASE_URL": DB, "LITELLM_MASTER_KEY": MASTER, "LITELLM_SALT_KEY": SALT, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_OTEL_V2": "true", "OTEL_BSP_SCHEDULE_DELAY": "500", "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
        "MIXPANEL_PROJECT_TOKEN": TOKEN,
        "HTTPS_PROXY": f"http://127.0.0.1:{MITM_PORT}", "REQUESTS_CA_BUNDLE": str(TOOLS / "mitm-conf/mitmproxy-ca-cert.pem"),
        "NO_PROXY": "127.0.0.1,localhost,api.openai.com,api.anthropic.com,license.litellm.ai,raw.githubusercontent.com",
        "PYTHONPATH": os.getcwd(),
    })
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
            if httpx.get(f"{BASE}/health/liveliness", timeout=2, trust_env=False).status_code == 200 and log.read_text(errors="replace").count("Started server process") >= 4:
                say(f"[{name}] up in {time.time() - began:.0f}s")
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
def api(method, path, body=None, key=MASTER, headers=None):
    return httpx.request(method, f"{BASE}{path}", headers={"Authorization": f"Bearer {key}", **(headers or {})}, json=body, timeout=180, trust_env=False)
def set_mode(value):
    MODE.write_text(value)
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
                    spans.append({"trace": span.trace_id.hex(), "span": span.span_id.hex(), "parent": span.parent_span_id.hex(), "name": span.name, "user_id": attributes.get("user.id"), "operation": attributes.get("gen_ai.operation.name"), "ingest": record["status"], "host": record["host"], "mode": record.get("mode"), "ts": record["ts"]})
        posts.append({"host": record["host"], "status": record["status"], "mode": record.get("mode"), "spans": count, "basic": record["has_basic_auth"], "ts": record["ts"]})
    return spans, posts
def landed(spans):
    """Spans Mixpanel accepted (200), keyed by trace, plus span ids accepted more than once."""
    accepted = [span for span in spans if span["ingest"] == 200]
    per_trace = defaultdict(list)
    for span in accepted:
        per_trace[span["trace"]].append(span)
    duplicates = [key for key, count in Counter((span["trace"], span["span"]) for span in accepted).items() if count > 1]
    return per_trace, duplicates
def after_auth(trace_spans):
    auth_ids = {span["span"] for span in trace_spans if span["name"].startswith("auth ")}
    return [span for span in trace_spans if span["parent"] not in auth_ids and span["name"] != "redis.get auth_cache"]
CALLS = {
    "chat": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
    "chat-stream": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "Reply with ok."}]}, True),
    "responses": ("/v1/responses", {"model": "gpt-4o-mini", "max_output_tokens": 16, "input": "Reply with ok."}, False),
    "messages": ("/v1/messages", {"model": "claude-haiku", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
    "messages-stream": ("/v1/messages", {"model": "claude-haiku", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "Reply with ok."}]}, True),
}
def send(key, call, caller, extra_headers=None):
    path, body, stream = CALLS[call]
    trace = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {key}", "traceparent": f"00-{trace}-{uuid.uuid4().hex[:16]}-01", **(extra_headers or {})}
    began = time.time()
    try:
        with httpx.Client(trust_env=False, timeout=180) as client:
            if stream:
                with client.stream("POST", f"{BASE}{path}", headers=headers, json={**body, "user": caller}) as reply:
                    "".join(reply.iter_text())
                    status = reply.status_code
            else:
                status = client.post(f"{BASE}{path}", headers=headers, json={**body, "user": caller}).status_code
    except httpx.HTTPError as failed:
        status = f"error:{type(failed).__name__}"
    return {"call": call, "caller": caller, "trace": trace, "status": status, "seconds": round(time.time() - began, 2)}
def burst(key, label, size=30):
    calls = list(CALLS)
    with ThreadPoolExecutor(size) as pool:
        rows = list(pool.map(lambda index: send(key, calls[index % len(calls)], f"lit10367-{label}-{index}-{uuid.uuid4().hex[:6]}"), range(size)))
    return rows
def score(rows, spans, label):
    per_trace, duplicates = landed(spans)
    traces = {row["trace"] for row in rows}
    ok = [row for row in rows if row["status"] == 200]
    landed_traces = [row for row in ok if per_trace.get(row["trace"])]
    missing_id = sum(1 for row in landed_traces for span in after_auth(per_trace[row["trace"]]) if span["user_id"] != row["caller"])
    model_with_id = sum(1 for row in landed_traces if any(span["operation"] and span["user_id"] == row["caller"] for span in per_trace[row["trace"]]))
    seconds = sorted(row["seconds"] for row in rows)
    result = {
        "requests": dict(Counter(str(row["status"]) for row in rows)),
        "p50_s": statistics.median(seconds), "p95_s": seconds[int(len(seconds) * 0.95) - 1],
        "traces_with_spans_accepted": len(landed_traces), "of_200": len(ok),
        "spans_accepted": sum(len(per_trace[row["trace"]]) for row in landed_traces),
        "model_span_with_caller_id": model_with_id,
        "after_auth_spans_without_caller_id": missing_id,
        "duplicate_span_ids_accepted": sum(1 for trace, _ in duplicates if trace in traces),
    }
    REPORT[label] = result
    say(f"{label}:", json.dumps(result))
    return result
def sdk_cells(key):
    from openai import AsyncOpenAI, OpenAI
    import anthropic
    rows = []
    def record(cell, caller, run):
        try:
            outcome = run()
            rows.append({"cell": cell, "caller": caller, "ok": True, "id": str(outcome)[:60]})
        except Exception as failed:  # noqa: BLE001
            rows.append({"cell": cell, "caller": caller, "ok": False, "id": f"{type(failed).__name__}: {str(failed)[:160]}"})
    openai_sync = OpenAI(api_key=key, base_url=f"{BASE}/v1", http_client=httpx.Client(trust_env=False, timeout=120))
    anthropic_sync = anthropic.Anthropic(api_key=key, base_url=BASE, http_client=httpx.Client(trust_env=False, timeout=120))
    message = [{"role": "user", "content": "Reply with ok."}]
    def caller():
        return f"lit10367-sdk-{uuid.uuid4().hex[:10]}"
    def chat_stream(who):
        stream = openai_sync.chat.completions.create(model="gpt-4o-mini", max_tokens=8, messages=message, user=who, stream=True)
        return len([chunk for chunk in stream])
    def anthropic_stream(who):
        with anthropic_sync.messages.stream(model="claude-haiku", max_tokens=8, messages=message, metadata={"user_id": who}) as stream:
            for _ in stream.text_stream:
                pass
            return stream.get_final_message().id
    for cell, run in (
        ("openai-sync/chat", lambda who: openai_sync.chat.completions.create(model="gpt-4o-mini", max_tokens=8, messages=message, user=who).id),
        ("openai-sync/chat-stream", chat_stream),
        ("openai-sync/responses", lambda who: openai_sync.responses.create(model="gpt-4o-mini", max_output_tokens=16, input="Reply with ok.", user=who).id),
        ("anthropic-sync/messages", lambda who: anthropic_sync.messages.create(model="claude-haiku", max_tokens=8, messages=message, metadata={"user_id": who}).id),
        ("anthropic-sync/messages-stream", anthropic_stream),
    ):
        who = caller()
        record(cell, who, lambda run=run, who=who: run(who))
    async def async_cells():
        openai_async = AsyncOpenAI(api_key=key, base_url=f"{BASE}/v1", http_client=httpx.AsyncClient(trust_env=False, timeout=120))
        anthropic_async = anthropic.AsyncAnthropic(api_key=key, base_url=BASE, http_client=httpx.AsyncClient(trust_env=False, timeout=120))
        out = []
        async def one(cell, coroutine_factory):
            who = caller()
            try:
                result = await coroutine_factory(who)
                out.append({"cell": cell, "caller": who, "ok": True, "id": str(result)[:60]})
            except Exception as failed:  # noqa: BLE001
                out.append({"cell": cell, "caller": who, "ok": False, "id": f"{type(failed).__name__}: {str(failed)[:160]}"})
        async def chat(who):
            return (await openai_async.chat.completions.create(model="gpt-4o-mini", max_tokens=8, messages=message, user=who)).id
        async def chat_stream_async(who):
            stream = await openai_async.chat.completions.create(model="gpt-4o-mini", max_tokens=8, messages=message, user=who, stream=True)
            return len([chunk async for chunk in stream])
        async def responses(who):
            return (await openai_async.responses.create(model="gpt-4o-mini", max_output_tokens=16, input="Reply with ok.", user=who)).id
        async def messages(who):
            return (await anthropic_async.messages.create(model="claude-haiku", max_tokens=8, messages=message, metadata={"user_id": who})).id
        async def messages_stream(who):
            async with anthropic_async.messages.stream(model="claude-haiku", max_tokens=8, messages=message, metadata={"user_id": who}) as stream:
                async for _ in stream.text_stream:
                    pass
                return (await stream.get_final_message()).id
        await asyncio.gather(*(one(cell, factory) for cell, factory in (
            ("openai-async/chat", chat), ("openai-async/chat-stream", chat_stream_async), ("openai-async/responses", responses),
            ("anthropic-async/messages", messages), ("anthropic-async/messages-stream", messages_stream),
        )))
        return out
    rows += asyncio.run(async_cells())
    return rows
def score_sdk(rows, spans):
    per_trace, duplicates = landed(spans)
    by_caller = defaultdict(set)
    for span in spans:
        if span["ingest"] == 200 and span["user_id"]:
            by_caller[span["user_id"]].add(span["trace"])
    results = []
    for row in rows:
        traces = by_caller.get(row["caller"], set())
        trace_spans = [span for trace in traces for span in per_trace[trace]]
        results.append({
            "cell": row["cell"], "client_ok": row["ok"], "response": row["id"], "traces": len(traces), "spans": len(trace_spans),
            "after_auth_without_id": sum(1 for span in after_auth(trace_spans) if span["user_id"] != row["caller"]),
            "model_span_with_id": any(span["operation"] and span["user_id"] == row["caller"] for span in trace_spans),
        })
    REPORT["sdk"] = results
    for result in results:
        say("sdk:", json.dumps(result))
def workers_of(process):
    children = sh(f"pgrep -P {process.pid}").stdout.split()
    return [pid for pid in children if "python" in sh(f"ps -o args= -p {pid}").stdout]
def main():
    start_postgres()
    set_mode("up")
    mitm, wire = start_mitm()
    proxy = None
    try:
        proxy, log = start_proxy("operator")
        key = api("POST", "/key/generate", {"key_alias": f"audit-{uuid.uuid4().hex[:6]}"}).json()["key"]
        sdk_rows = sdk_cells(key)
        team_entry = lambda region: [{"callback_name": "mixpanel", "callback_type": "success", "callback_vars": {"mixpanel_project_token": TOKEN, "mixpanel_region": region}}]
        team = api("POST", "/team/new", {"team_alias": f"audit-{uuid.uuid4().hex[:6]}", "metadata": {"logging": team_entry("us")}}).json()["team_id"]
        team_key = api("POST", "/key/generate", {"team_id": team, "key_alias": f"audit-team-{uuid.uuid4().hex[:6]}"}).json()["key"]
        flips = []
        for index in range(20):
            if index == 6:
                updated = api("POST", "/team/update", {"team_id": team, "metadata": {"logging": team_entry("eu")}})
                flip_at = time.time()
                say("team region us -> eu:", updated.status_code)
            row = send(team_key, "chat", f"lit10367-flip-{index}-{uuid.uuid4().hex[:6]}")
            row["index"], row["t"] = index, time.time()
            flips.append(row)
            time.sleep(0.5)
        time.sleep(8)
        baseline = burst(key, "baseline")
        time.sleep(8)
        set_mode("down")
        outage = burst(key, "outage")
        time.sleep(8)
        set_mode("up")
        say("destination back up; waiting 90s for exporter retries")
        time.sleep(90)
        set_mode("slow")
        slow = burst(key, "slow")
        time.sleep(15)
        set_mode("up")
        time.sleep(30)
        before = workers_of(proxy)
        if before:
            os.kill(int(before[0]), signal.SIGKILL)
        after_kill = [send(key, "chat", f"lit10367-kill-{index}-{uuid.uuid4().hex[:6]}") for index in range(16)]
        time.sleep(10)
        REPORT["worker_kill"] = {"workers_before": len(before), "killed": before[:1], "requests_after": dict(Counter(str(row["status"]) for row in after_kill)), "workers_10s_later": len(workers_of(proxy))}
        say("worker kill:", json.dumps(REPORT["worker_kill"]))
        restart_rows = []
        def restart_burst():
            restart_rows.extend(burst(key, "restart"))
        runner = threading.Thread(target=restart_burst)
        runner.start()
        time.sleep(1.5)
        os.kill(proxy.pid, signal.SIGTERM)
        runner.join(timeout=240)
        for _ in range(60):
            if proxy.poll() is not None:
                break
            time.sleep(1)
        exited_cleanly = proxy.poll() is not None
        stop(proxy)
        proxy, _ = start_proxy("operator-restarted")
        time.sleep(15)
        REPORT["restart_exited_on_sigterm"] = exited_cleanly
    finally:
        if proxy is not None:
            stop(proxy)
        time.sleep(2)
        os.killpg(mitm.pid, signal.SIGKILL)
        sh(f"docker rm -fv {PG}")
        MODE.unlink(missing_ok=True)
    spans, posts = decode(wire)
    score_sdk(sdk_rows, spans)
    per_trace, _ = landed(spans)
    REPORT["region_flip"] = [{"index": row["index"], "status": row["status"], "after_flip_s": round(row["t"] - flip_at, 1) if row["index"] >= 6 else None, "hosts": sorted({span["host"] for span in spans if span["trace"] == row["trace"]}), "ingest": sorted({span["ingest"] for span in spans if span["trace"] == row["trace"]})} for row in flips]
    for row in REPORT["region_flip"]:
        say("flip:", json.dumps(row))
    score(baseline, spans, "burst/baseline")
    score(outage, spans, "burst/outage-8s")
    score(slow, spans, "burst/slow-4s")
    score(restart_rows, spans, "burst/sigterm-mid-burst")
    REPORT["posts"] = dict(Counter(f"{post['host']} {post['mode']} {post['status']}" for post in posts))
    say("posts by host/mode/status:", json.dumps(REPORT["posts"]))
    for name in ("operator", "operator-restarted"):
        path = OUT / f"{name}-proxy.log"
        if path.exists():
            text = path.read_text(errors="replace")
            say(f"[{name}] log: Traceback={text.count('Traceback')} child-died={text.count('died')} export-failures={text.count('Failed to export')}")
    (OUT / "report.json").write_text(json.dumps(REPORT, default=str))
def worker_pids(log):
    import re
    return re.findall(r"Started server process \[(\d+)\]", log.read_text(errors="replace"))
def worker_kill_only():
    start_postgres()
    set_mode("up")
    mitm, wire = start_mitm()
    proxy = None
    try:
        proxy, log = start_proxy("worker-kill")
        key = api("POST", "/key/generate", {"key_alias": f"wk-{uuid.uuid4().hex[:6]}"}).json()["key"]
        before = worker_pids(log)
        warm = [send(key, "chat", f"lit10367-wkwarm-{index}-{uuid.uuid4().hex[:6]}") for index in range(8)]
        victim = int(before[0])
        os.kill(victim, signal.SIGKILL)
        killed_at = time.time()
        rows = []
        for index in range(24):
            row = send(key, ["chat", "chat-stream", "responses", "messages", "messages-stream"][index % 5], f"lit10367-wk-{index}-{uuid.uuid4().hex[:6]}")
            row["t"] = round(time.time() - killed_at, 1)
            rows.append(row)
        time.sleep(15)
        after = worker_pids(log)
        alive = [pid for pid in after if os.path.exists(f"/proc/{pid}")]
        REPORT["worker_kill"] = {"workers_started_before": before, "killed": victim, "victim_alive": os.path.exists(f"/proc/{victim}"), "requests_after": dict(Counter(str(row["status"]) for row in rows)), "warm": dict(Counter(str(row["status"]) for row in warm)), "worker_starts_total": len(after), "alive_workers_15s_later": len(alive), "log_mentions": {word: log.read_text(errors="replace").count(word) for word in ("died", "Child process", "Traceback")}}
        say("worker kill:", json.dumps(REPORT["worker_kill"]))
    finally:
        if proxy is not None:
            stop(proxy)
        time.sleep(2)
        os.killpg(mitm.pid, signal.SIGKILL)
        sh(f"docker rm -fv {PG}")
        MODE.unlink(missing_ok=True)
    spans, _ = decode(wire)
    score(rows, spans, "after-worker-kill")
    (OUT / "report.json").write_text(json.dumps(REPORT, default=str))
if len(sys.argv) > 3 and sys.argv[3] == "worker-kill":
    worker_kill_only()
else:
    main()

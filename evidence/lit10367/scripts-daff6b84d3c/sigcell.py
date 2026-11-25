"""SIGTERM mid-burst on a proxy with the generic OTel v2 callback and a local OTLP sink, 4 workers, no Mixpanel.
Counts how many answered requests still got their LLM call span (gen_ai.operation.name) after shutdown."""
import json, os, signal, subprocess, sys, threading, time, uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path.home() / "lit10367/live"))
import ab
from ab import BASE, DB, MASTER, OUT, PG, PROXY_PORT, SINK_PORT, SPANS, Sink, ThreadingHTTPServer, httpx, start_postgres, stop
CALLS = {
    "chat": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "messages": [{"role": "user", "content": "Reply with ok."}]}, False),
    "chat-stream": ("/v1/chat/completions", {"model": "gpt-4o-mini", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "Reply with ok."}]}, True),
    "responses": ("/v1/responses", {"model": "gpt-4o-mini", "max_output_tokens": 16, "input": "Reply with ok."}, False),
    "responses-stream": ("/v1/responses", {"model": "gpt-4o-mini", "max_output_tokens": 16, "stream": True, "input": "Reply with ok."}, True),
}
def start(name):
    config = OUT / f"{name}.yaml"
    config.write_text(BASE + 'litellm_settings:\n  callbacks: ["otel"]\n')
    drop = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "MIXPANEL_PROJECT_TOKEN")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"DATABASE_URL": DB, "LITELLM_MASTER_KEY": MASTER, "LITELLM_SALT_KEY": "lit10367-ab-salt-0123456789abcdef", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "DISABLE_SCHEMA_UPDATE": "true", "LITELLM_OTEL_V2": "true", "OTEL_BSP_SCHEDULE_DELAY": "500", "LITELLM_DISABLE_NO_REDIS_WARNING": "true", "OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": f"http://127.0.0.1:{SINK_PORT}/operator/v1/traces", "PYTHONPATH": os.getcwd()})
    log = OUT / f"{name}-proxy.log"
    process = subprocess.Popen([".venv/bin/python", "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(PROXY_PORT), "--num_workers", "4"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    began = time.time()
    while time.time() - began < 420:
        try:
            if httpx.get(f"http://127.0.0.1:{PROXY_PORT}/health/liveliness", timeout=2, trust_env=False).status_code == 200 and log.read_text(errors="replace").count("Started server process") >= 4:
                return process, log
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(log.read_text(errors="replace")[-800:])
def send(key, call, index):
    path, body, stream = CALLS[call]
    trace = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {key}", "traceparent": f"00-{trace}-{uuid.uuid4().hex[:16]}-01"}
    try:
        with httpx.Client(trust_env=False, timeout=180) as client:
            if stream:
                with client.stream("POST", f"http://127.0.0.1:{PROXY_PORT}{path}", headers=headers, json={**body, "user": f"sig-{index}"}) as reply:
                    "".join(reply.iter_text())
                    status = reply.status_code
            else:
                status = client.post(f"http://127.0.0.1:{PROXY_PORT}{path}", headers=headers, json={**body, "user": f"sig-{index}"}).status_code
    except httpx.HTTPError as failed:
        status = f"error:{type(failed).__name__}"
    return {"trace": trace, "status": status, "call": call}
def burst(key, size=30):
    calls = list(CALLS)
    with ThreadPoolExecutor(size) as pool:
        return list(pool.map(lambda i: send(key, calls[i % len(calls)], i), range(size)))
def score(rows):
    llm = {s["trace"] for s in SPANS if "gen_ai.operation.name" in s["keys"]}
    any_span = {s["trace"] for s in SPANS}
    ok = [r for r in rows if r["status"] == 200]
    return {"status": dict(Counter(str(r["status"]) for r in rows)), "answered_with_llm_span": sum(r["trace"] in llm for r in ok), "answered_with_any_span": sum(r["trace"] in any_span for r in ok), "answered": len(ok), "missing_by_call": dict(Counter(r["call"] for r in ok if r["trace"] not in llm))}
def main():
    server = ThreadingHTTPServer(("127.0.0.1", SINK_PORT), Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_postgres()
    try:
        process, log = start("sig")
        key = httpx.post(f"http://127.0.0.1:{PROXY_PORT}/key/generate", headers={"Authorization": f"Bearer {MASTER}"}, json={"key_alias": f"sig-{uuid.uuid4().hex[:6]}"}, trust_env=False, timeout=60).json()["key"]
        baseline = burst(key)
        time.sleep(8)
        rows = []
        runner = threading.Thread(target=lambda: rows.extend(burst(key)))
        runner.start()
        time.sleep(1.5)
        os.kill(process.pid, signal.SIGTERM)
        runner.join(timeout=240)
        for _ in range(60):
            if process.poll() is not None:
                break
            time.sleep(1)
        exited = process.poll() is not None
        stop(process)
        time.sleep(5)
    finally:
        subprocess.run(f"docker rm -fv {PG}", shell=True, capture_output=True)
    result = {"baseline": score(baseline), "sigterm": score(rows), "exited_on_sigterm": exited, "tracebacks": log.read_text(errors="replace").count("Traceback"), "cancelled": log.read_text(errors="replace").count("CancelledError")}
    (OUT / "sig.json").write_text(json.dumps(result))
    print("RESULT", json.dumps(result), flush=True)
main()

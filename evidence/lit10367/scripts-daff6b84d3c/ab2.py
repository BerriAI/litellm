"""Sad bodies, a concurrent burst and a paused database, on a proxy with OTel V2 and no Mixpanel. Compares status and error text."""
import collections, json, re, subprocess, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path.home() / "lit10367/live"))
import threading
import ab
from ab import CHAT, MASTER, PG, PROXY_PORT, SINK_PORT, Sink, ThreadingHTTPServer, api, httpx, psql, start_postgres, start_proxy, stop
BIG = json.dumps({"pad": "x" * 1_000_000, "user.id": "caller-a"})
SAD = {
    "metadata-not-json": ({"metadata": "not json"}, {}),
    "metadata-int": ({"metadata": 5}, {}),
    "metadata-list": ({"metadata": ["x"]}, {}),
    "metadata-null": ({"metadata": None}, {}),
    "metadata-big-json-string": ({"metadata": BIG}, {}),
    "metadata-user-dict-session": ({"metadata": {"user_id": {"session_id": "s1"}}}, {}),
    "metadata-user-json-session": ({"metadata": {"user_id": json.dumps({"session_id": "s1", "device_id": "d"})}}, {}),
    "metadata-user-dot-id-dict": ({"metadata": {"user.id": {"a": 1}}}, {}),
    "user-int": ({"user": 5}, {}),
    "user-list": ({"user": ["a"]}, {}),
    "user-empty": ({"user": ""}, {}),
    "user-5kb": ({"user": "u" * 5000}, {}),
    "litellm-metadata-not-json": ({"litellm_metadata": "nope"}, {}),
    "baggage-control-char": ({}, {"baggage": "user.id=a%0Ab"}),
    "baggage-malformed": ({}, {"baggage": ";;;==="}),
    "customer-header-empty": ({}, {"x-litellm-customer-id": ""}),
}
def scrub(text):
    text = re.sub(r"[0-9a-f]{16,}", "<hex>", text)
    return re.sub(r"req_[0-9A-Za-z]+|chatcmpl-[0-9A-Za-z]+|\d{10,}|\"created\":\d+", "<id>", text)
def body_summary(response):
    if response.status_code == 200:
        return ""
    return scrub(response.text)[:240]
def main():
    server = ThreadingHTTPServer(("127.0.0.1", SINK_PORT), Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_postgres()
    out = {}
    try:
        process = start_proxy("sad", f'litellm_settings:\n  callbacks: ["otel"]\n', {"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": f"http://127.0.0.1:{SINK_PORT}/operator/v1/traces"})
        try:
            key = api("POST", "/key/generate", {"key_alias": f"sad-{uuid.uuid4().hex[:6]}"}).json()["key"]
            for name, (extra, headers) in SAD.items():
                reply = api("POST", "/v1/chat/completions", {**CHAT, **extra}, key=key, headers=headers)
                out[f"sad/{name}"] = (reply.status_code, body_summary(reply))
            unauth = api("POST", "/v1/chat/completions", {**CHAT, "metadata": {"user.id": "x"}}, key="sk-not-a-key", headers={"baggage": "user.id=x"})
            out["sad/unauthenticated"] = (unauth.status_code, body_summary(unauth))
            def one(index):
                name = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
                body = {"/v1/chat/completions": CHAT, "/v1/messages": {"model": "claude-haiku", "max_tokens": 8, "messages": CHAT["messages"]}, "/v1/responses": {"model": "gpt-4o-mini", "max_output_tokens": 16, "input": "Reply with ok."}}[name]
                return api("POST", name, {**body, "user": f"burst-{index}"}, key=key).status_code
            with ThreadPoolExecutor(30) as pool:
                out["burst/30 concurrent"] = sorted(collections.Counter(pool.map(one, range(30))).items())
            paused = subprocess.run(f"docker pause {PG}", shell=True, capture_output=True, text=True)
            state = subprocess.run(f"docker inspect -f '{{{{.State.Paused}}}}' {PG}", shell=True, capture_output=True, text=True)
            out["pause check"] = f"rc={paused.returncode} paused={state.stdout.strip()} err={paused.stderr.strip()[:80]}"
            with ThreadPoolExecutor(8) as pool:
                out["postgres paused/8 requests"] = sorted(collections.Counter(pool.map(one, range(8))).items())
            subprocess.run(f"docker unpause {PG}", shell=True, capture_output=True)
            time.sleep(2)
            out["after unpause/1 request"] = one(0)
        finally:
            subprocess.run(f"docker unpause {PG}", shell=True, capture_output=True)
            stop(process)
        out["log tracebacks"] = (ab.OUT / "sad-proxy.log").read_text(errors="replace").count("Traceback")
    finally:
        subprocess.run(f"docker rm -fv {PG}", shell=True, capture_output=True)
    (ab.OUT / "sad.json").write_text(json.dumps(out, indent=0, default=list))
    print(json.dumps({k: (v if not isinstance(v, (list, tuple)) or len(str(v)) < 90 else str(v)[:90]) for k, v in out.items()}, indent=0), flush=True)
main()

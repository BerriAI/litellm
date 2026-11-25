"""Postgres pause/unpause recovery, repeated: seconds from unpause to the first 200 on a proxy with OTel V2 and no Mixpanel."""
import collections, json, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path.home() / "lit10367/live"))
import ab
from ab import CHAT, PG, SINK_PORT, Sink, ThreadingHTTPServer, api, start_postgres, start_proxy, stop
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
def main():
    server = ThreadingHTTPServer(("127.0.0.1", SINK_PORT), Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_postgres()
    rounds = []
    try:
        process = start_proxy("pause", 'litellm_settings:\n  callbacks: ["otel"]\n', {"OTEL_EXPORTER": "otlp_http", "OTEL_ENDPOINT": f"http://127.0.0.1:{SINK_PORT}/operator/v1/traces"})
        try:
            key = api("POST", "/key/generate", {"key_alias": f"pause-{uuid.uuid4().hex[:6]}"}).json()["key"]
            def one(index):
                return api("POST", "/v1/chat/completions", {**CHAT, "user": f"pause-{index}"}, key=key).status_code
            for round_index in range(ROUNDS):
                with ThreadPoolExecutor(10) as pool:
                    warm = sorted(collections.Counter(pool.map(one, range(10))).items())
                subprocess.run(f"docker pause {PG}", shell=True, capture_output=True)
                with ThreadPoolExecutor(8) as pool:
                    paused = sorted(collections.Counter(pool.map(one, range(8))).items())
                subprocess.run(f"docker unpause {PG}", shell=True, capture_output=True)
                unpaused_at = time.time()
                probes = []
                first_ok = None
                while time.time() - unpaused_at < 60:
                    time.sleep(1)
                    status = one(100 + len(probes))
                    probes.append((round(time.time() - unpaused_at, 1), status))
                    if status == 200 and first_ok is None:
                        first_ok = probes[-1][0]
                        break
                row = {"round": round_index, "warm": warm, "paused": paused, "probe_at_2s": next((s for t, s in probes if t >= 2), None), "first_200_s": first_ok, "probes": probes}
                rounds.append(row)
                print(json.dumps(row), flush=True)
                time.sleep(15)
        finally:
            subprocess.run(f"docker unpause {PG}", shell=True, capture_output=True)
            stop(process)
    finally:
        subprocess.run(f"docker rm -fv {PG}", shell=True, capture_output=True)
    log = (ab.OUT / "pause-proxy.log").read_text(errors="replace")
    summary = {"first_200_s": [r["first_200_s"] for r in rounds], "status_at_2s": [r["probe_at_2s"] for r in rounds], "engine_reconnects": log.count("Attempting Prisma DB reconnect"), "not_connected": log.count("ClientNotConnectedError")}
    (ab.OUT / "pause.json").write_text(json.dumps({"rounds": rounds, "summary": summary}))
    print("SUMMARY", json.dumps(summary), flush=True)
main()

"""Parse spans.jsonl (one compact span JSON per line), group by trace_id, and
assert the behavior each of the four PRs introduced. Prints a PASS/FAIL matrix."""

import json
import os
import sys
from datetime import datetime

SERVER = "Received Proxy Server Request"
FAILED = "Failed Proxy Server Request"

# trace_id (32 hex) -> expectation spec
EXPECT = {
    "00000000000000000000000000000001": {"label": "T01 chat 200 (team)", "code": 200, "otel": "OK",
        "team_on": [SERVER, "litellm_request", "raw_gen_ai_request"]},
    "00000000000000000000000000000002": {"label": "T02 chat 400 invalid-json", "code": 400, "otel": "ERROR"},
    "00000000000000000000000000000003": {"label": "T03 chat 401 bad-key", "code": 401, "otel": "ERROR"},
    "00000000000000000000000000000004": {"label": "T04 chat 400 unknown-model", "code": 400, "otel": "ERROR"},
    "00000000000000000000000000000005": {"label": "T05 chat 429 (team)", "code": 429, "otel": "ERROR",
        "team_on": [SERVER, FAILED]},
    "00000000000000000000000000000006": {"label": "T06 chat 500 (team)", "code": 500, "otel": "ERROR"},
    "00000000000000000000000000000007": {"label": "T07 messages 200 (team)", "code": 200, "otel": "OK"},
    "00000000000000000000000000000008": {"label": "T08 admin 200 key/generate", "code": 200, "otel": "OK"},
    "00000000000000000000000000000009": {"label": "T09 admin 500 key/generate", "code": 500, "otel": "ERROR"},
    "00000000000000000000000000000010": {"label": "T10 admin 422 key/generate", "code": 422, "otel": "ERROR"},
    "00000000000000000000000000000011": {"label": "T11 guardrail block 400", "code": 400, "otel": "ERROR",
        "guardrail": {"guardrail_status": "guardrail_intervened",
                       "guardrail_action": "GUARDRAIL_INTERVENED",
                       "categories": ["Fiduciary Advice", "VIOLENCE"]}},
    "00000000000000000000000000000012": {"label": "T12 guardrail allow 200", "code": 200, "otel": "OK",
        "guardrail": {"guardrail_status": "success"}},
}

def _read(path, default=""):
    try:
        with open(os.path.join(os.path.dirname(__file__), path)) as f:
            return f.read().strip()
    except OSError:
        return default


# Written by setup.sh; the team_alias is fixed by that script.
TEAM_ID = os.getenv("TEAM_ID") or _read(".team_id")
TEAM_ALIAS = os.getenv("TEAM_ALIAS", "otel-verify-team")


def norm_tid(raw):
    return raw[2:] if raw.startswith("0x") else raw


def duration_ms(span):
    try:
        a = datetime.fromisoformat(span["start_time"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(span["end_time"].replace("Z", "+00:00"))
        return (b - a).total_seconds() * 1000
    except Exception:
        return None


def main():
    traces = {}
    for line in open(sys.argv[1] if len(sys.argv) > 1 else "spans.jsonl"):
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        tid = norm_tid(s["context"]["trace_id"])
        traces.setdefault(tid, []).append(s)

    all_pass = True
    for tid, spec in EXPECT.items():
        spans = traces.get(tid, [])
        by_name = {}
        for s in spans:
            by_name.setdefault(s["name"], s).update() if False else by_name.setdefault(s["name"], s)
        checks = []

        # --- #28405: status code / route / path / duration on SERVER span ---
        server = next((s for s in spans if s["name"] == SERVER), None)
        if server is None:
            checks.append(("#28405 SERVER span present", False, "missing"))
        else:
            a = server.get("attributes") or {}
            code = a.get("http.response.status_code")
            checks.append(("#28405 http.response.status_code", code == spec["code"] and isinstance(code, int),
                           f"{code!r}"))
            checks.append(("#28405 url.path present", bool(a.get("url.path")), a.get("url.path")))
            checks.append(("#28405 http.route present", bool(a.get("http.route")), a.get("http.route")))
            d = duration_ms(server)
            checks.append(("#28405 duration>0", d is not None and d > 0, f"{d:.1f}ms" if d else d))
            ostat = (server.get("status") or {}).get("status_code")
            checks.append(("#28405 otel status", ostat == spec["otel"], ostat))

        # --- #28273: team attrs on the listed spans ---
        for name in spec.get("team_on", []):
            sp = next((s for s in spans if s["name"] == name), None)
            if sp is None:
                checks.append((f"#28273 team attrs on {name}", False, "span missing"))
                continue
            a = sp.get("attributes") or {}
            ok = a.get("metadata.user_api_key_team_id") == TEAM_ID and \
                 a.get("metadata.user_api_key_team_alias") == TEAM_ALIAS
            checks.append((f"#28273 team attrs on {name}", ok,
                           f"{a.get('metadata.user_api_key_team_id')}/{a.get('metadata.user_api_key_team_alias')}"))

        # --- #28362 + #28364: guardrail span ---
        g = spec.get("guardrail")
        if g is not None:
            gs = next((s for s in spans if s["name"] == "guardrail"), None)
            if gs is None:
                checks.append(("#28364 guardrail span emitted", False, "missing"))
            else:
                a = gs.get("attributes") or {}
                checks.append(("#28364 guardrail span emitted", True, "present"))
                checks.append(("#28364 guardrail_status",
                               a.get("guardrail_status") == g["guardrail_status"], a.get("guardrail_status")))
                # #28362: guardrail_response must be valid JSON, not a python repr
                gr = a.get("guardrail_response")
                json_ok = False
                detail = "absent"
                if gr is not None:
                    try:
                        parsed = json.loads(gr)
                        json_ok = isinstance(parsed, (dict, list)) and "'" not in gr
                        detail = "valid JSON dict" if json_ok else f"not-json: {gr[:40]}"
                    except Exception:
                        detail = f"NOT JSON (python repr?): {gr[:40]}"
                checks.append(("#28362 guardrail_response is JSON", json_ok, detail))
                if "guardrail_action" in g:
                    checks.append(("#28364 guardrail_action",
                                   a.get("guardrail_action") == g["guardrail_action"], a.get("guardrail_action")))
                if "categories" in g:
                    cats_raw = a.get("guardrail_violation_categories")
                    cats_ok = False
                    try:
                        cats_ok = sorted(json.loads(cats_raw)) == sorted(g["categories"])
                    except Exception:
                        pass
                    checks.append(("#28364 violation_categories", cats_ok, cats_raw))

        trace_pass = all(ok for _, ok, _ in checks)
        all_pass = all_pass and trace_pass
        spans_seen = ",".join(sorted({s["name"] for s in spans}))
        print(f"\n{'='*78}\n{spec['label']}  [trace …{tid[-4:]}]   {'PASS' if trace_pass else 'FAIL'}")
        print(f"  spans: {spans_seen}")
        for name, ok, detail in checks:
            print(f"    [{'OK' if ok else 'XX'}] {name:42} {detail}")

    print(f"\n{'='*78}\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

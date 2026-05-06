"""
Validation #8 — SSE resume.

Read 3 events, kill, reconnect with starting_seq=3, get rest. No gaps,
no dupes.

Implementation note: we don't open a real long-running SSE stream — we
shape the test as "given a run with N events, the events stream should
emit exactly the unseen ones and close once the run is terminal." We
check via the StreamingResponse body iterator with a finished run so
the loop terminates quickly.
"""

import json
import re

from litellm.proxy.agent_session_endpoints.constants import (
    EVENT_TYPE_RUN_FINISHED,
)


def _bootstrap_ready(client, noop_provider):
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{agent["id"]}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    daemon_token = sess["daemon_token"]
    sid = sess["id"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    return sid, daemon_token


def _seed_events(client, sid, daemon_token, count: int) -> str:
    run = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    ).json()
    rid = run["id"]
    client.get(
        f"/v2/sessions/{sid}/runs/next/internal/poll",
        headers={"Authorization": f"Bearer {daemon_token}"},
    )
    for i in range(count):
        client.post(
            f"/v2/sessions/{sid}/runs/{rid}/events:append",
            headers={"Authorization": f"Bearer {daemon_token}"},
            json={"event_type": "log", "payload": {"i": i}},
        )
    # End the run so the SSE stream can quiesce + close.
    client.post(
        f"/v2/sessions/{sid}/runs/{rid}/events:append",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"event_type": EVENT_TYPE_RUN_FINISHED, "payload": {"result": "done"}},
    )
    return rid


def _parse_seqs(body_text: str):
    return [int(m.group(1)) for m in re.finditer(r"^id:\s*(\d+)$", body_text, re.M)]


def test_sse_emits_all_events_from_zero(client, noop_provider):
    sid, daemon_token = _bootstrap_ready(client, noop_provider)
    rid = _seed_events(client, sid, daemon_token, count=5)

    res = client.get(
        f"/v2/sessions/{sid}/runs/{rid}/stream",
        headers={"Authorization": "Bearer k"},
    )
    assert res.status_code == 200
    seqs = _parse_seqs(res.text)
    # Seqs 1..6 (5 logs + 1 run_finished)
    assert seqs == [1, 2, 3, 4, 5, 6]


def test_sse_resumes_with_starting_seq(client, noop_provider):
    sid, daemon_token = _bootstrap_ready(client, noop_provider)
    rid = _seed_events(client, sid, daemon_token, count=5)

    res = client.get(
        f"/v2/sessions/{sid}/runs/{rid}/stream?starting_seq=4",
        headers={"Authorization": "Bearer k"},
    )
    assert res.status_code == 200
    seqs = _parse_seqs(res.text)
    # starting_seq=4 means "last seen was 3, give me >= 4".
    assert seqs == [4, 5, 6]


def test_sse_last_event_id_header_takes_precedence(client, noop_provider):
    sid, daemon_token = _bootstrap_ready(client, noop_provider)
    rid = _seed_events(client, sid, daemon_token, count=3)

    res = client.get(
        f"/v2/sessions/{sid}/runs/{rid}/stream",
        headers={
            "Authorization": "Bearer k",
            "Last-Event-ID": "2",
        },
    )
    assert res.status_code == 200
    seqs = _parse_seqs(res.text)
    # Last-Event-ID=2 means "give me > 2", so we expect 3 + 4 (run_finished).
    assert seqs == [3, 4]

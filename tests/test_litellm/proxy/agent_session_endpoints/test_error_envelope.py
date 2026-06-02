"""
Verify the standardized {error: {code, message, status, details?}} envelope
applies to /v2/* endpoints (HTTPException + ValidationError) and that
non-/v2 paths keep FastAPI's default detail shape.
"""

import pytest


def test_404_emits_envelope_for_missing_session(client, noop_provider):
    """A 404 from the v2 surface should be wrapped in the envelope."""
    res = client.get(
        "/v2/sessions/does-not-exist",
        headers={"Authorization": "Bearer k"},
    )
    # find_unique returns None -> ownership assert raises HTTPException(404).
    assert res.status_code == 404
    body = res.json()
    assert "error" in body and "detail" not in body
    assert body["error"]["status"] == 404
    assert body["error"]["code"] == "not_found"
    assert isinstance(body["error"]["message"], str)


def test_validation_error_wraps_in_envelope(client, noop_provider):
    """A 422 ValidationError from a missing required body field should
    arrive as ``{error: {code: validation_error, details: [...]}}``."""
    # POST /v2/agents requires `name` + `model`; omit both.
    res = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={},
    )
    assert res.status_code == 422
    body = res.json()
    assert "error" in body and "detail" not in body
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["status"] == 422
    assert isinstance(body["error"].get("details"), list)
    # Validation details preserve the per-field {loc, msg, type} shape.
    assert any("name" in str(d.get("loc", [])) for d in body["error"]["details"])


def test_http_status_code_to_envelope_code_mapping(client, noop_provider):
    """409 should map to ``conflict``, not a generic name."""
    # Create an agent + session, then try to POST with missing agent_id
    # path -> simpler: trigger a 409 by creating two runs back-to-back
    # on the same session.
    agent = client.post(
        "/v2/agents",
        headers={"Authorization": "Bearer k"},
        json={"name": "t", "model": "gpt-4"},
    ).json()
    sess = client.post(
        f"/v2/agents/{agent['id']}/sessions",
        headers={"Authorization": "Bearer k"},
        json={"repos": []},
    ).json()
    sid = sess["id"]
    daemon_token = sess["daemon_token"]
    client.post(
        f"/v2/sessions/{sid}/internal/register",
        headers={"Authorization": f"Bearer {daemon_token}"},
        json={"vm_id": "i-noop"},
    )
    # First run: queued.
    client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "hi"}},
    )
    # Second run on same session while first is still queued -> 409 run_busy.
    res = client.post(
        f"/v2/sessions/{sid}/runs",
        headers={"Authorization": "Bearer k"},
        json={"prompt": {"text": "again"}},
    )
    assert res.status_code == 409
    body = res.json()
    assert body["error"]["status"] == 409
    assert body["error"]["code"] == "conflict"
    assert "run_busy" in body["error"]["message"]

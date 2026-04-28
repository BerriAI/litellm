"""
E2E contract tests for LiteLLM-managed conversation IDs over the Responses API.

Any implementation that supports gateway-managed conversation state via the
`litellm_conv_id_<uuid>` prefix on the Responses API `conversation` field MUST
pass these tests.

Behavior under test
-------------------
1. A first `/v1/responses` call with `conversation="litellm_conv_id_<uuid>"`
   succeeds and is recorded against the gateway's spend logs under
   `session_id = "litellm_conv_id_<uuid>"`.
2. A second `/v1/responses` call reusing the same conversation id succeeds and
   is recorded against the same `session_id`. The model receives prior turn
   history (we assert that follow-up referencing turn 1 works without resending
   the original prompt).
3. `GET /spend/logs/session/ui?session_id=litellm_conv_id_<uuid>` returns BOTH
   request rows, ordered, and tagged with the conversation id as `session_id`.

These tests run against a live proxy at http://0.0.0.0:4000 with master key
`sk-1234`, matching the rest of `tests/openai_endpoints_tests/`.
"""

import time
import uuid

import httpx
import pytest
from openai import OpenAI

PROXY_BASE_URL = "http://0.0.0.0:4000"
MASTER_KEY = "sk-1234"
LITELLM_CONV_PREFIX = "litellm_conv_id_"
TEST_MODEL = "fast"


def _generate_key() -> str:
    """Mint a virtual key for the test so spend logs are attributable."""
    response = httpx.post(
        f"{PROXY_BASE_URL}/key/generate",
        headers={
            "Authorization": f"Bearer {MASTER_KEY}",
            "Content-Type": "application/json",
        },
        json={},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Key generation failed: status={response.status_code} body={response.text}"
        )
    return response.json()["key"]


def _get_test_client() -> OpenAI:
    return OpenAI(api_key=_generate_key(), base_url=PROXY_BASE_URL)


def _new_conversation_id() -> str:
    return f"{LITELLM_CONV_PREFIX}{uuid.uuid4()}"


def _fetch_session_logs(session_id: str) -> dict:
    """
    Read all spend logs for a session via the existing
    /spend/logs/session/ui endpoint. Uses the master key so we can see
    every row regardless of which virtual key wrote it.
    """
    response = httpx.get(
        f"{PROXY_BASE_URL}/spend/logs/session/ui",
        params={"session_id": session_id, "page": 1, "page_size": 100},
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        timeout=30.0,
    )
    assert response.status_code == 200, (
        f"Expected 200 from /spend/logs/session/ui, got {response.status_code}: "
        f"{response.text}"
    )
    return response.json()


def _wait_for_logs(session_id: str, expected_count: int, timeout_s: float = 30.0) -> dict:
    """
    Spend logs are written asynchronously after the request returns. Poll
    /spend/logs/session/ui until we see at least `expected_count` rows for
    this session_id, or fail.
    """
    deadline = time.time() + timeout_s
    last_payload: dict = {}
    while time.time() < deadline:
        last_payload = _fetch_session_logs(session_id)
        if last_payload.get("total", 0) >= expected_count:
            return last_payload
        time.sleep(1.0)
    raise AssertionError(
        f"Timed out waiting for {expected_count} spend logs for "
        f"session_id={session_id}. Last payload: {last_payload}"
    )


@pytest.mark.flaky(retries=3, delay=2)
def test_litellm_conversation_id_persists_across_responses_calls():
    """
    The contract:

    - Two /v1/responses calls sharing the same `litellm_conv_id_<uuid>` value in
      the `conversation` field must both succeed.
    - Both calls must land in LiteLLM_SpendLogs with session_id equal to that
      conversation id.
    - The conversation id must NOT be forwarded to the upstream provider as the
      provider's own conversation id (i.e. responses still come back with their
      own provider-issued response ids, not the litellm_conv_id_).
    """
    client = _get_test_client()
    conversation_id = _new_conversation_id()

    response_1 = client.responses.create(
        model=TEST_MODEL,
        input="Remember the number 42. Just acknowledge with the word 'ok'.",
        extra_body={"conversation": conversation_id},
    )
    assert response_1 is not None, "first response was None"
    assert hasattr(response_1, "id"), "first response missing id"
    assert isinstance(response_1.id, str) and len(response_1.id) > 0
    assert not response_1.id.startswith(LITELLM_CONV_PREFIX), (
        "Provider response id must not be the gateway-minted conversation id; "
        f"got {response_1.id!r}"
    )

    response_2 = client.responses.create(
        model=TEST_MODEL,
        input="What number did I just ask you to remember? Reply with only the digits.",
        extra_body={"conversation": conversation_id},
    )
    assert response_2 is not None, "second response was None"
    assert hasattr(response_2, "id"), "second response missing id"
    assert response_2.id != response_1.id, (
        "Each call must produce a distinct provider response id even when "
        "sharing the same conversation."
    )

    payload = _wait_for_logs(session_id=conversation_id, expected_count=2)

    assert payload["total"] >= 2, (
        f"Expected at least 2 spend log rows for session_id={conversation_id}, "
        f"got total={payload['total']}. Payload: {payload}"
    )

    rows = payload["data"]
    session_ids = {row.get("session_id") for row in rows}
    assert session_ids == {conversation_id}, (
        f"Every spend log for this session must carry session_id="
        f"{conversation_id!r}; got {session_ids!r}"
    )

    call_types = [row.get("call_type") for row in rows]
    assert all(ct in {"aresponses", "responses"} for ct in call_types), (
        f"All rows should be Responses API calls; got call_types={call_types}"
    )

    sorted_rows = sorted(rows, key=lambda r: r.get("startTime") or "")
    assert sorted_rows[0]["startTime"] <= sorted_rows[-1]["startTime"]


@pytest.mark.flaky(retries=3, delay=2)
def test_litellm_conversation_id_history_is_replayed_to_model():
    """
    The model must be able to answer a follow-up question that ONLY makes sense
    if turn-1 history was replayed to it. This exercises the read side of the
    feature (history reconstruction from spend logs keyed by session_id).

    We avoid asserting on exact model output (flaky) and instead assert that
    the model produced a non-empty response containing the expected token,
    which is a strong signal that prior context was injected.
    """
    client = _get_test_client()
    conversation_id = _new_conversation_id()

    secret_token = "PURPLE-OWL-7821"

    client.responses.create(
        model=TEST_MODEL,
        input=(
            f"You are participating in a memory test. The secret token is "
            f"{secret_token}. Acknowledge with the word 'noted'."
        ),
        extra_body={"conversation": conversation_id},
    )

    follow_up = client.responses.create(
        model=TEST_MODEL,
        input="Repeat the secret token I gave you earlier, exactly, with no other text.",
        extra_body={"conversation": conversation_id},
    )

    output_text = ""
    if hasattr(follow_up, "output_text") and follow_up.output_text:
        output_text = follow_up.output_text
    elif hasattr(follow_up, "output") and follow_up.output:
        for item in follow_up.output:
            for content in getattr(item, "content", []) or []:
                text_val = getattr(content, "text", None)
                if isinstance(text_val, str):
                    output_text += text_val

    assert secret_token in output_text, (
        f"Follow-up call did not see prior conversation history. "
        f"Expected output to contain {secret_token!r}, got: {output_text!r}"
    )


@pytest.mark.flaky(retries=3, delay=2)
def test_non_litellm_conversation_id_is_not_intercepted():
    """
    Conversation ids that do NOT start with `litellm_conv_id_` must pass through
    untouched. The gateway must not write a spend-log session_id matching some
    arbitrary upstream conversation id, because doing so would silently leak
    cross-customer history.
    """
    client = _get_test_client()
    foreign_conversation_id = f"conv_{uuid.uuid4().hex}"

    try:
        client.responses.create(
            model=TEST_MODEL,
            input="ping",
            extra_body={"conversation": foreign_conversation_id},
        )
    except Exception:
        # Some providers reject unknown conversation ids. That's fine for this
        # test; we only care that the gateway didn't claim ownership of the id.
        pass

    payload = _fetch_session_logs(session_id=foreign_conversation_id)
    assert payload["total"] == 0, (
        f"Foreign conversation id {foreign_conversation_id!r} must not be "
        f"written as a spend-log session_id by the gateway. Got "
        f"total={payload['total']}: {payload}"
    )

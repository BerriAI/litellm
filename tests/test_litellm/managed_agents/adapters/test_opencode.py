"""Unit tests for the opencode sandbox adapter.

Covers `litellm/managed_agents/adapters/opencode.py` per the v2 contract §7
(`.claude/v2_api_contract.md`):

  - send_message: POST <sandbox_url>/session/<oc_sid>/prompt_async with
    body {parts: [{type:"text", text: content}], model}; 204 -> success;
    ConnectError -> SandboxUnreachableError; returns None.
  - list_messages: GET <sandbox_url>/session/<oc_sid>/message; normalize
    each entry into MessageRow; empty -> [].
  - stream_events: yields ("connected", {...}) first, then translated
    opencode events; drops events for other opencode_session_ids;
    ConnectError -> SandboxUnreachableError.
  - delete: DELETE <sandbox_url>/session/<oc_sid>; errors swallowed.

HTTP mocking uses `respx` (already a dev dep). Tests are async and rely on
the repo's `asyncio_mode = "auto"` setting.
"""

import json
from typing import Any, Dict, List

import httpx
import pytest
import respx

# The adapter module is being built in parallel by the adapter agent.
# Skip cleanly if it isn't here yet so the rest of the test suite still runs.
opencode_module = pytest.importorskip(
    "litellm.managed_agents.adapters.opencode"
)
adapter_base = pytest.importorskip("litellm.managed_agents.adapters.base")
types_module = pytest.importorskip("litellm.managed_agents.types")

OpencodeAdapter = opencode_module.OpencodeAdapter
SandboxUnreachableError = adapter_base.SandboxUnreachableError
MessageRow = types_module.MessageRow


SANDBOX_URL = "http://127.0.0.1:1234"
OC_SID = "oc_sid_xxx"
OUR_SID = "ses_test"


def _make_adapter() -> Any:
    """Build a fresh adapter instance.

    Adapters are stateless per the protocol docstring, so no setup is needed.
    The builder exists so a future change in constructor signature only has
    to be updated in one place.
    """
    return OpencodeAdapter()


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    """POST <sandbox_url>/session/<oc_sid>/prompt_async."""

    @pytest.mark.asyncio
    async def test_posts_correct_url_and_body(self) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(
                f"{SANDBOX_URL}/session/{OC_SID}/prompt_async"
            ).mock(return_value=httpx.Response(204))

            result = await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="Hello, who are you?",
                model="anthropic/claude-opus-4",
            )

            assert result is None
            assert route.called
            assert route.call_count == 1
            request = route.calls[0].request
            assert request.method == "POST"
            assert (
                str(request.url)
                == f"{SANDBOX_URL}/session/{OC_SID}/prompt_async"
            )

            body = json.loads(request.content.decode("utf-8"))
            assert body == {
                "parts": [{"type": "text", "text": "Hello, who are you?"}],
                "model": "anthropic/claude-opus-4",
            }

    @pytest.mark.asyncio
    async def test_204_is_success(self) -> None:
        """opencode returns 204 No Content on prompt_async success."""
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(
                f"{SANDBOX_URL}/session/{OC_SID}/prompt_async"
            ).mock(return_value=httpx.Response(204))

            # Should NOT raise. send_message returns None on success.
            result = await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="hi",
                model=None,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_connect_error_raises_sandbox_unreachable(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(
                f"{SANDBOX_URL}/session/{OC_SID}/prompt_async"
            ).mock(side_effect=httpx.ConnectError("connection refused"))

            with pytest.raises(SandboxUnreachableError):
                await adapter.send_message(
                    sandbox_url=SANDBOX_URL,
                    opencode_session_id=OC_SID,
                    content="hi",
                    model="anthropic/claude-opus-4",
                )


# ---------------------------------------------------------------------------
# list_messages
# ---------------------------------------------------------------------------


class TestListMessages:
    """GET <sandbox_url>/session/<oc_sid>/message."""

    @pytest.mark.asyncio
    async def test_normalizes_each_message_and_returns_message_rows(
        self,
    ) -> None:
        adapter = _make_adapter()

        opencode_payload: List[Dict[str, Any]] = [
            {
                "id": "msg_oc_1",
                "sessionID": OC_SID,
                "role": "user",
                "parts": [{"type": "text", "text": "refactor src/auth.py"}],
                "completedAt": "2026-05-07T15:04:05.123Z",
            },
            {
                "id": "msg_oc_2",
                "sessionID": OC_SID,
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "I'll start by reading."},
                    {
                        "type": "tool",
                        "name": "read",
                        "input": {"path": "src/auth.py"},
                        "output": "...",
                    },
                ],
                "completedAt": "2026-05-07T15:04:06.000Z",
            },
        ]

        with respx.mock as mock:
            mock.get(f"{SANDBOX_URL}/session/{OC_SID}/message").mock(
                return_value=httpx.Response(200, json=opencode_payload)
            )

            rows = await adapter.list_messages(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            )

            assert isinstance(rows, list)
            assert len(rows) == 2
            for row in rows:
                assert isinstance(row, MessageRow)
                # session_id MUST be our ses_*, never the opencode oc_sid.
                assert row.session_id == OUR_SID
                assert row.session_id != OC_SID

            # Spot-check normalization rules.
            assert rows[0].id == "msg_oc_1"
            assert rows[0].role == "user"
            assert rows[0].content == "refactor src/auth.py"
            assert rows[0].tools is None  # no tool parts on user message

            assert rows[1].id == "msg_oc_2"
            assert rows[1].role == "assistant"
            assert rows[1].content == "I'll start by reading."
            assert rows[1].tools == [
                {
                    "name": "read",
                    "input": {"path": "src/auth.py"},
                    "output": "...",
                }
            ]

    @pytest.mark.asyncio
    async def test_empty_array_returns_empty_list(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.get(f"{SANDBOX_URL}/session/{OC_SID}/message").mock(
                return_value=httpx.Response(200, json=[])
            )

            rows = await adapter.list_messages(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            )

            assert rows == []


# ---------------------------------------------------------------------------
# stream_events
# ---------------------------------------------------------------------------


def _sse_payload(events: List[Dict[str, Any]]) -> bytes:
    """Build a valid SSE response body from event dicts.

    opencode emits events on `<sandbox_url>/event` formatted as standard
    SSE: each event is a `data: <json>\n\n` chunk. The adapter is expected
    to decode each `data:` line into a dict and pass it to
    `normalize_opencode_event` after filtering by `sessionID`.
    """
    chunks = []
    for ev in events:
        chunks.append(f"data: {json.dumps(ev)}\n\n")
    return "".join(chunks).encode("utf-8")


class TestStreamEvents:
    """GET <sandbox_url>/event (SSE)."""

    @pytest.mark.asyncio
    async def test_yields_connected_first_then_translated_events(self) -> None:
        adapter = _make_adapter()

        events_on_wire = [
            {
                "type": "message.updated",
                "sessionID": OC_SID,
                "properties": {
                    "info": {"id": "msg_a4b5c6", "role": "assistant"},
                    "sessionID": OC_SID,
                },
            },
            {
                "type": "message.part.updated",
                "sessionID": OC_SID,
                "properties": {
                    "sessionID": OC_SID,
                    "part": {
                        "type": "text",
                        "messageID": "msg_a4b5c6",
                        "text": "Hello",
                    },
                },
            },
            {
                "type": "session.idle",
                "sessionID": OC_SID,
                "properties": {
                    "sessionID": OC_SID,
                    "messageID": "msg_a4b5c6",
                    "content": "Hello",
                    "completedAt": "2026-05-07T15:04:05.123Z",
                },
            },
        ]

        with respx.mock as mock:
            mock.get(f"{SANDBOX_URL}/event").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_payload(events_on_wire),
                )
            )

            collected: List[Any] = []
            async for ev in adapter.stream_events(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            ):
                collected.append(ev)

            assert len(collected) >= 1
            # First yield must be ("connected", {"session_id": <our_sid>}).
            first_type, first_data = collected[0]
            assert first_type == "connected"
            assert first_data.get("session_id") == OUR_SID

            # Subsequent translated events should appear in order.
            translated_types = [t for t, _ in collected[1:]]
            assert "message.started" in translated_types
            assert "message.text.delta" in translated_types
            assert "message.completed" in translated_types

    @pytest.mark.asyncio
    async def test_drops_events_for_other_opencode_sessions(self) -> None:
        adapter = _make_adapter()

        other_sid = "oc_sid_OTHER"
        events_on_wire = [
            # Belongs to a different opencode session — must be filtered out.
            {
                "type": "message.updated",
                "sessionID": other_sid,
                "properties": {
                    "info": {"id": "msg_other", "role": "assistant"},
                    "sessionID": other_sid,
                },
            },
            # Belongs to our session — should be yielded.
            {
                "type": "message.updated",
                "sessionID": OC_SID,
                "properties": {
                    "info": {"id": "msg_ours", "role": "assistant"},
                    "sessionID": OC_SID,
                },
            },
        ]

        with respx.mock as mock:
            mock.get(f"{SANDBOX_URL}/event").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_payload(events_on_wire),
                )
            )

            collected: List[Any] = []
            async for ev in adapter.stream_events(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            ):
                collected.append(ev)

            # We expect: [connected, message.started for msg_ours]. The
            # `msg_other` event must NOT appear.
            translated = [data for t, data in collected if t == "message.started"]
            assert len(translated) == 1
            assert translated[0].get("message_id") == "msg_ours"

            for _, data in collected:
                assert data.get("message_id") != "msg_other"

    @pytest.mark.asyncio
    async def test_connect_error_raises_sandbox_unreachable(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.get(f"{SANDBOX_URL}/event").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            with pytest.raises(SandboxUnreachableError):
                async for _ in adapter.stream_events(
                    sandbox_url=SANDBOX_URL,
                    opencode_session_id=OC_SID,
                    our_session_id=OUR_SID,
                ):
                    pass


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    """DELETE <sandbox_url>/session/<oc_sid> — best-effort, errors swallowed."""

    @pytest.mark.asyncio
    async def test_deletes_session_at_correct_url(self) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.delete(f"{SANDBOX_URL}/session/{OC_SID}").mock(
                return_value=httpx.Response(200)
            )

            result = await adapter.delete(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None
            assert route.called
            assert route.call_count == 1
            request = route.calls[0].request
            assert request.method == "DELETE"
            assert str(request.url) == f"{SANDBOX_URL}/session/{OC_SID}"

    @pytest.mark.asyncio
    async def test_swallows_connect_error(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.delete(f"{SANDBOX_URL}/session/{OC_SID}").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            # Best-effort: must NOT raise even if the sandbox is unreachable.
            result = await adapter.delete(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_swallows_non_2xx_response(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.delete(f"{SANDBOX_URL}/session/{OC_SID}").mock(
                return_value=httpx.Response(500, text="boom")
            )

            # Even on 5xx we treat delete as fire-and-forget.
            result = await adapter.delete(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None

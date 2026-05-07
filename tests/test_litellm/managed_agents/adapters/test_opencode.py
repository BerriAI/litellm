"""Unit tests for the opencode sandbox adapter.

Covers `litellm/managed_agents/adapters/opencode.py` per the v2 contract
§7 (`.claude/v2_api_contract.md`) and against the REAL opencode 1.14.41
wire shapes documented in `.claude/v2_opencode_real_responses.md`.

Coverage:
  - send_message: POST <sandbox_url>/session/<oc_sid>/prompt_async with
    body ``{parts:[{type:"text",text:content}], model:{providerID,modelID}}``;
    string-form ``"<provider>/<model>"`` is split into the object form;
    ``model`` is omitted when it has no ``/`` separator; 204 -> success;
    ConnectError -> SandboxUnreachableError; returns None.
  - list_messages: GET <sandbox_url>/session/<oc_sid>/message; normalize
    each ``{info, parts}`` envelope into MessageRow; empty -> [].
  - stream_events: yields ``("connected", ...)`` first; tracks
    ``partID -> type`` across events so ``message.part.delta`` for text
    is yielded as ``message.text.delta``; routes
    ``message.part.updated`` for tools to ``message.tool.started`` /
    ``.completed``; tracks the in-flight assistant ``message_id`` so
    ``session.idle`` (which carries no messageID in real opencode) is
    enriched before being yielded; auto-grants permission on
    ``permission.asked``; drops events for other opencode_session_ids;
    ConnectError -> SandboxUnreachableError.
  - delete: DELETE <sandbox_url>/session/<oc_sid>; errors swallowed.

HTTP mocking uses ``respx`` (already a dev dep). Tests are async and rely
on the repo's ``asyncio_mode = "auto"`` setting.
"""

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest
import respx

opencode_module = pytest.importorskip("litellm.managed_agents.adapters.opencode")
adapter_base = pytest.importorskip("litellm.managed_agents.adapters.base")
types_module = pytest.importorskip("litellm.managed_agents.types")

OpencodeAdapter = opencode_module.OpencodeAdapter
SandboxUnreachableError = adapter_base.SandboxUnreachableError
MessageRow = types_module.MessageRow


@pytest.fixture(autouse=True)
def _bypass_async_client_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """In production the adapter pulls a cached ``httpx.AsyncClient`` from
    ``get_async_httpx_client``. The cache survives across tests, so once
    a client is created its transport is fixed and ``respx`` (which
    patches new clients only) cannot intercept it. For tests we replace
    ``_get_async_client`` with a fresh-client factory so each test sees
    its own respx mock cleanly.
    """
    monkeypatch.setattr(
        opencode_module,
        "_get_async_client",
        lambda timeout: httpx.AsyncClient(timeout=timeout),
    )


SANDBOX_URL = "http://127.0.0.1:1234"
OC_SID = "ses_oc_xxx"
OUR_SID = "ses_test"

_TS_CREATED_MS = 1778172682972
_TS_COMPLETED_MS = 1778172689905


def _make_adapter() -> Any:
    """Build a fresh adapter instance.

    Adapters are stateless across calls per the protocol docstring, so
    no setup is needed. The builder exists so a future change in
    constructor signature only has to be updated in one place.
    """
    return OpencodeAdapter()


# ---------------------------------------------------------------------------
# send_message — model object split
# ---------------------------------------------------------------------------


class TestSendMessage:
    """POST <sandbox_url>/session/<oc_sid>/prompt_async."""

    @pytest.mark.asyncio
    async def test_posts_correct_url_and_splits_model_into_provider_object(
        self,
    ) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                return_value=httpx.Response(204)
            )

            result = await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="Hello, who are you?",
                model="anthropic/claude-opus-4",
            )

            assert result is None
            assert route.called
            request = route.calls[0].request
            body = json.loads(request.content.decode("utf-8"))
            # Real opencode rejects string-form ``model`` — must be object.
            assert body == {
                "parts": [{"type": "text", "text": "Hello, who are you?"}],
                "model": {
                    "providerID": "anthropic",
                    "modelID": "claude-opus-4",
                },
            }

    @pytest.mark.asyncio
    async def test_model_with_extra_slashes_keeps_modelid_intact(self) -> None:
        """Some model identifiers contain ``/`` (e.g.
        ``"openrouter/anthropic/claude-3"``). We split on the first slash
        only so the rest stays as ``modelID``."""
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                return_value=httpx.Response(204)
            )

            await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="hi",
                model="openrouter/anthropic/claude-3",
            )

            body = json.loads(route.calls[0].request.content.decode("utf-8"))
            assert body["model"] == {
                "providerID": "openrouter",
                "modelID": "anthropic/claude-3",
            }

    @pytest.mark.asyncio
    async def test_model_without_slash_is_omitted(self) -> None:
        """opencode treats a missing ``model`` field as "use default" —
        we omit the field entirely instead of guessing a providerID."""
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                return_value=httpx.Response(204)
            )

            await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="hi",
                model="just-a-name",
            )

            body = json.loads(route.calls[0].request.content.decode("utf-8"))
            assert "model" not in body
            assert body == {"parts": [{"type": "text", "text": "hi"}]}

    @pytest.mark.asyncio
    async def test_model_none_is_omitted(self) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                return_value=httpx.Response(204)
            )

            await adapter.send_message(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                content="hi",
                model=None,
            )

            body = json.loads(route.calls[0].request.content.decode("utf-8"))
            assert "model" not in body

    @pytest.mark.asyncio
    async def test_204_is_success(self) -> None:
        """opencode returns 204 No Content on prompt_async success."""
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                return_value=httpx.Response(204)
            )

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
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/prompt_async").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            with pytest.raises(SandboxUnreachableError):
                await adapter.send_message(
                    sandbox_url=SANDBOX_URL,
                    opencode_session_id=OC_SID,
                    content="hi",
                    model="anthropic/claude-opus-4",
                )


# ---------------------------------------------------------------------------
# list_messages — {info, parts} envelopes
# ---------------------------------------------------------------------------


class TestListMessages:
    """GET <sandbox_url>/session/<oc_sid>/message."""

    @pytest.mark.asyncio
    async def test_normalizes_real_shape_messages_into_message_rows(
        self,
    ) -> None:
        adapter = _make_adapter()

        # Real opencode response shape: bare array of {info, parts} envelopes.
        opencode_payload: List[Dict[str, Any]] = [
            {
                "info": {
                    "id": "msg_oc_1",
                    "sessionID": OC_SID,
                    "role": "user",
                    "time": {"created": _TS_CREATED_MS},
                },
                "parts": [
                    {
                        "type": "text",
                        "text": "refactor src/auth.py",
                        "id": "prt_1",
                    }
                ],
            },
            {
                "info": {
                    "id": "msg_oc_2",
                    "sessionID": OC_SID,
                    "role": "assistant",
                    "providerID": "opencode",
                    "modelID": "minimax-m2.5-free",
                    "time": {
                        "created": _TS_CREATED_MS,
                        "completed": _TS_COMPLETED_MS,
                    },
                    "finish": "stop",
                },
                "parts": [
                    {"type": "step-start", "snapshot": "..."},
                    {"type": "reasoning", "text": "thinking..."},
                    {"type": "text", "text": "I'll start by reading."},
                    {
                        "type": "tool",
                        "tool": "read",
                        "callID": "call_xxx",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "src/auth.py"},
                            "output": "...",
                        },
                    },
                    {"type": "step-finish", "reason": "stop"},
                ],
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
            # step-start, reasoning, step-finish are filtered out of content.
            assert rows[1].content == "I'll start by reading."
            assert rows[1].tools == [
                {
                    "name": "read",
                    "input": {"filePath": "src/auth.py"},
                    "output": "...",
                }
            ]
            assert rows[1].model == "opencode/minimax-m2.5-free"

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

    opencode emits events on ``<sandbox_url>/event`` formatted as standard
    SSE: each event is a ``data: <json>\\n\\n`` chunk.
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
            # 1. New assistant message (no completed yet).
            {
                "type": "message.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "info": {
                        "id": "msg_a4b5c6",
                        "sessionID": OC_SID,
                        "role": "assistant",
                        "time": {"created": _TS_CREATED_MS},
                    },
                },
            },
            # 2. Text part appears (registers partID -> "text").
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "part": {
                        "id": "prt_text_1",
                        "type": "text",
                        "messageID": "msg_a4b5c6",
                        "sessionID": OC_SID,
                        "text": "",
                    },
                },
            },
            # 3. Streaming text delta — must route via partID lookup.
            {
                "type": "message.part.delta",
                "properties": {
                    "sessionID": OC_SID,
                    "messageID": "msg_a4b5c6",
                    "partID": "prt_text_1",
                    "field": "text",
                    "delta": "Hello",
                },
            },
            # 4. Per-message completion via message.updated with time.completed.
            {
                "type": "message.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "info": {
                        "id": "msg_a4b5c6",
                        "sessionID": OC_SID,
                        "role": "assistant",
                        "time": {
                            "created": _TS_CREATED_MS,
                            "completed": _TS_COMPLETED_MS,
                        },
                    },
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

            translated_types = [t for t, _ in collected[1:]]
            assert "message.started" in translated_types
            assert "message.text.delta" in translated_types
            assert "message.completed" in translated_types

            # Verify the text delta carried the right content.
            text_deltas = [d for t, d in collected if t == "message.text.delta"]
            assert len(text_deltas) == 1
            assert text_deltas[0]["delta"] == "Hello"
            assert text_deltas[0]["message_id"] == "msg_a4b5c6"

    @pytest.mark.asyncio
    async def test_part_types_map_persists_across_events(self) -> None:
        """A delta arriving AFTER a tool ``message.part.updated`` must
        still find the ``text`` part registered earlier in the stream."""
        adapter = _make_adapter()

        events_on_wire = [
            # Register text part.
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "part": {
                        "id": "prt_text_1",
                        "type": "text",
                        "messageID": "msg_a4b5c6",
                        "sessionID": OC_SID,
                        "text": "",
                    },
                },
            },
            # Register tool part (different partID).
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "part": {
                        "id": "prt_tool_1",
                        "type": "tool",
                        "messageID": "msg_a4b5c6",
                        "sessionID": OC_SID,
                        "tool": "read",
                        "callID": "call_xxx",
                        "state": {
                            "status": "running",
                            "input": {"filePath": "src/auth.py"},
                        },
                    },
                },
            },
            # Text delta should still route via the original partID.
            {
                "type": "message.part.delta",
                "properties": {
                    "sessionID": OC_SID,
                    "messageID": "msg_a4b5c6",
                    "partID": "prt_text_1",
                    "field": "text",
                    "delta": "still text",
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

            text_deltas = [d for t, d in collected if t == "message.text.delta"]
            assert len(text_deltas) == 1
            assert text_deltas[0]["delta"] == "still text"

            tool_starts = [d for t, d in collected if t == "message.tool.started"]
            assert len(tool_starts) == 1
            assert tool_starts[0]["tool"] == "read"

    @pytest.mark.asyncio
    async def test_session_idle_is_enriched_with_tracked_message_id(
        self,
    ) -> None:
        """Real opencode ``session.idle`` payload only carries
        ``sessionID``. The adapter must inject the most recent assistant
        ``message_id`` from prior ``message.updated`` events."""
        adapter = _make_adapter()

        events_on_wire = [
            {
                "type": "message.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "info": {
                        "id": "msg_tracked",
                        "sessionID": OC_SID,
                        "role": "assistant",
                        "time": {"created": _TS_CREATED_MS},
                    },
                },
            },
            {
                "type": "session.idle",
                "properties": {"sessionID": OC_SID},
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

            completes = [d for t, d in collected if t == "message.completed"]
            # `session.idle` produces a ``message.completed`` enriched
            # with the tracked ``message_id``.
            assert any(d.get("message_id") == "msg_tracked" for d in completes)

    @pytest.mark.asyncio
    async def test_drops_events_for_other_opencode_sessions(self) -> None:
        adapter = _make_adapter()

        other_sid = "ses_oc_OTHER"
        events_on_wire = [
            {
                "type": "message.updated",
                "properties": {
                    "sessionID": other_sid,
                    "info": {
                        "id": "msg_other",
                        "sessionID": other_sid,
                        "role": "assistant",
                        "time": {"created": _TS_CREATED_MS},
                    },
                },
            },
            {
                "type": "message.updated",
                "properties": {
                    "sessionID": OC_SID,
                    "info": {
                        "id": "msg_ours",
                        "sessionID": OC_SID,
                        "role": "assistant",
                        "time": {"created": _TS_CREATED_MS},
                    },
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

            translated = [data for t, data in collected if t == "message.started"]
            assert len(translated) == 1
            assert translated[0].get("message_id") == "msg_ours"

            for _, data in collected:
                assert data.get("message_id") != "msg_other"

    @pytest.mark.asyncio
    async def test_permission_asked_triggers_auto_grant(self) -> None:
        """``permission.asked`` events fire a fire-and-forget POST to
        ``/session/:sid/permissions/:per_id`` body
        ``{"response":"once"}`` so tool calls don't hang."""
        adapter = _make_adapter()

        permission_id = "per_xxx"
        events_on_wire = [
            {
                "type": "permission.asked",
                "properties": {
                    "id": permission_id,
                    "sessionID": OC_SID,
                    "permission": "external_directory",
                    "patterns": ["/etc/*"],
                    "metadata": {"filepath": "/etc/hostname"},
                    "tool": {"messageID": "msg_a4b5c6", "callID": "call_xxx"},
                },
            },
        ]

        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{SANDBOX_URL}/event").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_payload(events_on_wire),
                )
            )
            grant_route = mock.post(
                f"{SANDBOX_URL}/session/{OC_SID}/permissions/{permission_id}"
            ).mock(return_value=httpx.Response(200, json=True))

            async for _ in adapter.stream_events(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            ):
                pass

            # Auto-grant fires fire-and-forget via asyncio.create_task —
            # give the loop a chance to settle outstanding tasks.
            for _ in range(5):
                if grant_route.called:
                    break
                await asyncio.sleep(0.01)

            assert grant_route.called
            grant_body = json.loads(
                grant_route.calls[0].request.content.decode("utf-8")
            )
            assert grant_body == {"response": "once"}

    @pytest.mark.asyncio
    async def test_permission_asked_for_other_session_is_ignored(self) -> None:
        """``permission.asked`` events for OTHER opencode sessions on the
        same server must NOT trigger auto-grant — otherwise a caller
        streaming /events on session A could approve permissions for
        session B.
        """
        adapter = _make_adapter()

        other_sid = "other_oc_sid"
        permission_id = "per_xxx"
        events_on_wire = [
            {
                "type": "permission.asked",
                "properties": {
                    "id": permission_id,
                    "sessionID": other_sid,  # NOT our session
                    "permission": "external_directory",
                    "patterns": ["/etc/*"],
                    "metadata": {"filepath": "/etc/hostname"},
                    "tool": {"messageID": "msg_zzz", "callID": "call_zzz"},
                },
            },
        ]

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{SANDBOX_URL}/event").mock(
                return_value=httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=_sse_payload(events_on_wire),
                )
            )
            # If the bug were present, the adapter would POST here.
            other_grant_route = mock.post(
                f"{SANDBOX_URL}/session/{other_sid}/permissions/{permission_id}"
            ).mock(return_value=httpx.Response(200, json=True))

            async for _ in adapter.stream_events(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
                our_session_id=OUR_SID,
            ):
                pass

            # Give any (incorrect) fire-and-forget task time to land.
            for _ in range(5):
                await asyncio.sleep(0.01)

            assert (
                not other_grant_route.called
            ), "auto-grant must not fire for permissions on other sessions"

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


# ---------------------------------------------------------------------------
# abort — POST <sandbox_url>/session/<oc_sid>/abort
# ---------------------------------------------------------------------------


class TestAbort:
    """POST <sandbox_url>/session/<oc_sid>/abort."""

    @pytest.mark.asyncio
    async def test_posts_to_correct_url_with_empty_body(self) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{SANDBOX_URL}/session/{OC_SID}/abort").mock(
                return_value=httpx.Response(200)
            )

            result = await adapter.abort(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None
            assert route.called
            request = route.calls[0].request
            assert request.method == "POST"
            assert str(request.url) == f"{SANDBOX_URL}/session/{OC_SID}/abort"
            # Body is empty JSON object (`{}`) per opencode contract.
            assert json.loads(request.content.decode("utf-8")) == {}

    @pytest.mark.asyncio
    async def test_204_is_success(self) -> None:
        adapter = _make_adapter()

        with respx.mock(assert_all_called=True) as mock:
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/abort").mock(
                return_value=httpx.Response(204)
            )

            result = await adapter.abort(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_connect_error_raises_sandbox_unreachable(self) -> None:
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/abort").mock(
                side_effect=httpx.ConnectError("connection refused")
            )

            with pytest.raises(SandboxUnreachableError):
                await adapter.abort(
                    sandbox_url=SANDBOX_URL,
                    opencode_session_id=OC_SID,
                )

    @pytest.mark.asyncio
    async def test_swallows_404_response(self) -> None:
        """Best-effort: a 404 from opencode (session already gone) is NOT an error."""
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/abort").mock(
                return_value=httpx.Response(404, text="not found")
            )

            # Must NOT raise — opencode 404 is treated as best-effort success.
            result = await adapter.abort(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_swallows_non_2xx_response(self) -> None:
        """Best-effort: a 500 from opencode is also swallowed."""
        adapter = _make_adapter()

        with respx.mock as mock:
            mock.post(f"{SANDBOX_URL}/session/{OC_SID}/abort").mock(
                return_value=httpx.Response(500, text="boom")
            )

            # Must NOT raise.
            result = await adapter.abort(
                sandbox_url=SANDBOX_URL,
                opencode_session_id=OC_SID,
            )

            assert result is None

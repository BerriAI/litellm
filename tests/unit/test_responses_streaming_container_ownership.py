"""
Regression for #30210.

When streaming /v1/responses goes through the proxy + Router, the router
returns a ``FallbackResponsesStreamWrapper``. That wrapper set
``self.completed_response = None`` in __init__ and never updated it,
so the proxy's container-ownership hook (which reads
``getattr(stream_response, "completed_response", None)`` via
``ProxyBaseLLMRequestProcessing.extract_completed_responses_response``)
saw None on every streaming call and silently recorded nothing —
follow-up ``GET /v1/containers/<id>/files`` then 403'd for the very
key that created the container.

Tests below stream a mocked upstream through ``Router.aresponses`` and
assert the returned stream carries the terminal chunk on
``completed_response`` so the proxy hook can walk it.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import respx

import litellm

_DELTA = {
    "type": "response.output_text.delta",
    "sequence_number": 1,
    "item_id": "msg_1",
    "output_index": 0,
    "content_index": 0,
    "delta": "hello",
    "logprobs": [],
}


def _terminal_event(event_type: str, status: str, sequence_number: int = 2, created_at: int = 1761600000):
    return {
        "type": event_type,
        "sequence_number": sequence_number,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": "gpt-5.5",
            "output": [],
            "container": {"id": "cntr_test", "type": "code_interpreter"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }


def _stream_through_router(*events):
    """Stream ``events`` from a mocked upstream through ``Router.aresponses``;
    return the router's stream object after draining it, and the chunks."""
    body = b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)

    async def _drive():
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-5.5",
                    "litellm_params": {
                        "model": "openai/gpt-5.5",
                        "api_key": "sk-test",
                        "api_base": "https://api.openai.com/v1",
                    },
                }
            ]
        )
        with respx.mock() as mock_router:
            mock_router.post("https://api.openai.com/v1/responses").mock(
                return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
            )
            stream = await router.aresponses(model="gpt-5.5", input="hi", stream=True)
            chunks = [chunk async for chunk in stream]
        return stream, chunks

    with patch.object(litellm, "disable_aiohttp_transport", True):
        litellm.in_memory_llm_clients_cache.flush_cache()
        try:
            return asyncio.run(_drive())
        finally:
            litellm.in_memory_llm_clients_cache.flush_cache()


class TestStreamWrapperCapturesTerminalEvent:
    def test_terminal_completed_event_is_recorded_on_wrapper(self):
        """The #30210 bug: a forwarded ``response.completed`` chunk used
        to leave ``completed_response`` at None on the wrapper. Verify
        it now carries the chunk."""
        stream, chunks = _stream_through_router(_DELTA, _terminal_event("response.completed", "completed"))
        assert len(chunks) == 2
        assert stream.completed_response is not None, (
            "FallbackResponsesStreamWrapper.completed_response is still None "
            "after a response.completed chunk passed through — the proxy "
            "container-ownership hook will 403 follow-up file lookups (#30210)"
        )
        assert stream.completed_response.type == "response.completed"

    def test_terminal_incomplete_event_is_recorded(self):
        stream, _ = _stream_through_router(_terminal_event("response.incomplete", "incomplete", sequence_number=1))
        assert stream.completed_response is not None
        assert stream.completed_response.type == "response.incomplete"

    def test_first_terminal_event_wins(self):
        """Real streams only emit one terminal event, but defend against
        future producers emitting more: keep the first one (the inner
        source iterator behaves the same way)."""
        stream, _ = _stream_through_router(
            _terminal_event("response.completed", "completed", sequence_number=1, created_at=1),
            _terminal_event("response.completed", "completed", sequence_number=2, created_at=2),
        )
        assert stream.completed_response.response.created_at == 1


class TestProxyOwnershipHookReadsCompletedResponse:
    """End-to-end: the proxy hook reads exactly the attribute the
    wrapper now populates. Pin that the helper still extracts the
    response correctly so the ownership recording path doesn't break."""

    def test_extract_returns_inner_response_object(self):
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        stream, _ = _stream_through_router(_terminal_event("response.completed", "completed", sequence_number=1))

        extracted = ProxyBaseLLMRequestProcessing.extract_completed_responses_response(stream)
        assert extracted is not None
        assert extracted.container["id"] == "cntr_test"


class TestSilentSkipNowLogged:
    """Reporter's secondary ask: when completed_response is None, the
    ownership hook silently dropped on the floor. Make sure the new
    warning fires so operators see a hint instead of a mute 403."""

    def test_warning_logged_when_completed_response_missing(self, caplog):
        import logging

        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        # A stream that never saw a terminal event: the same shape as the
        # pre-fix bug.
        stream = SimpleNamespace(completed_response=None)

        async def driver():
            async def inner_gen():
                yield SimpleNamespace(type="response.output_text.delta", delta="hello")

            # Patch _record_container_owners_from_responses_if_needed to
            # a noop async so the warning branch is exercised in
            # isolation.
            with patch.object(
                ProxyBaseLLMRequestProcessing,
                "_record_container_owners_from_responses_if_needed",
                new=MagicMock(),
            ):
                wrapped = ProxyBaseLLMRequestProcessing._wrap_responses_stream_for_container_ownership(
                    original_stream_response=stream,
                    wrapped_generator=inner_gen(),
                    user_api_key_dict=MagicMock(),
                )
                async for _ in wrapped:
                    pass

        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            asyncio.run(driver())

        assert any(
            "Container ownership recording skipped on streaming /v1/responses" in r.message for r in caplog.records
        ), "silent-skip warning never fired despite completed_response=None"

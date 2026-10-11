"""
Regression for #30210.

When streaming /v1/responses goes through the proxy + Router, the
streaming iterator is wrapped by ``Router._aresponses_streaming_iterator``
which returns ``FallbackResponsesStreamWrapper``. That wrapper set
``self.completed_response = None`` in __init__ and never updated it,
so the proxy's container-ownership hook (which reads
``getattr(stream_response, "completed_response", None)`` via
``ProxyBaseLLMRequestProcessing._extract_completed_responses_response``)
saw None on every streaming call and silently recorded nothing —
follow-up ``GET /v1/containers/<id>/files`` then 403'd for the very
key that created the container.

Tests below construct the wrapper from a fake async generator that
yields one terminal ``response.completed`` chunk and assert the
wrapper now carries that chunk on ``completed_response`` so the
proxy hook can walk it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_wrapper_class():
    """Pull ``FallbackResponsesStreamWrapper`` out by running
    ``Router._aresponses_streaming_iterator`` long enough to construct
    the class then return it. Mirrors how the wrapper is actually
    instantiated in production."""
    from litellm.router import Router
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    # Minimal source iterator stub with every attribute the wrapper
    # copies in __init__ (see router.py:2552-2583).
    source = SimpleNamespace(
        response=None,
        model="openai/gpt-5.5",
        logging_obj=None,
        responses_api_provider_config=None,
        start_time=None,
        litellm_metadata=None,
        custom_llm_provider="openai",
        request_data={},
        call_type="aresponses",
        _hidden_params={},
    )

    # The class is defined inside _aresponses_streaming_iterator; capture
    # it by patching FallbackResponsesStreamWrapper into a sentinel on
    # construction.
    captured = {}

    real_router_module = __import__("litellm.router", fromlist=["Router"])

    async def _drive():
        async def empty_gen():
            if False:
                yield  # pragma: no cover
            return

        router = Router(
            model_list=[
                {
                    "model_name": "openai/gpt-5.5",
                    "litellm_params": {"model": "openai/gpt-5.5", "api_key": "sk-test"},
                }
            ]
        )
        wrapped = await router._aresponses_streaming_iterator(
            response=source,  # type: ignore[arg-type]
            initial_kwargs={},
        )
        captured["wrapper_cls"] = type(wrapped)
        captured["instance"] = wrapped

    asyncio.run(_drive())
    return captured["wrapper_cls"], captured["instance"]


def _terminal_chunk(event_type: str):
    """A SimpleNamespace shaped like the openai responses-api terminal
    event chunks the wrapper inspects (.type attribute)."""
    return SimpleNamespace(
        type=event_type,
        response=SimpleNamespace(
            id="resp_test",
            output=[],
            container={"id": "cntr_test", "type": "code_interpreter"},
        ),
    )


def _non_terminal_chunk(event_type: str = "response.output_text.delta"):
    return SimpleNamespace(type=event_type, delta="hello")


class TestStreamWrapperCapturesTerminalEvent:
    def test_terminal_completed_event_is_recorded_on_wrapper(self):
        """The #30210 bug: a forwarded ``response.completed`` chunk used
        to leave ``completed_response`` at None on the wrapper. Verify
        it now carries the chunk."""
        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _non_terminal_chunk()
            yield _terminal_chunk("response.completed")

        wrapper = wrapper_cls(gen())
        # Drain the wrapper.
        out = asyncio.run(_drain(wrapper))
        assert len(out) == 2
        assert wrapper.completed_response is not None, (
            "FallbackResponsesStreamWrapper.completed_response is still None "
            "after a response.completed chunk passed through — the proxy "
            "container-ownership hook will 403 follow-up file lookups (#30210)"
        )
        assert wrapper.completed_response.type == "response.completed"

    def test_terminal_incomplete_event_is_recorded(self):
        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _terminal_chunk("response.incomplete")

        wrapper = wrapper_cls(gen())
        asyncio.run(_drain(wrapper))
        assert wrapper.completed_response is not None
        assert wrapper.completed_response.type == "response.incomplete"

    def test_terminal_failed_event_is_recorded(self):
        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _terminal_chunk("response.failed")

        wrapper = wrapper_cls(gen())
        asyncio.run(_drain(wrapper))
        assert wrapper.completed_response is not None
        assert wrapper.completed_response.type == "response.failed"

    def test_non_terminal_chunks_do_not_set_completed_response(self):
        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _non_terminal_chunk("response.output_text.delta")
            yield _non_terminal_chunk("response.code_interpreter.in_progress")

        wrapper = wrapper_cls(gen())
        asyncio.run(_drain(wrapper))
        assert (
            wrapper.completed_response is None
        ), "non-terminal chunks must not set completed_response"

    def test_first_terminal_event_wins(self):
        """Real streams only emit one terminal event, but defend against
        future producers emitting more: keep the first one (the inner
        source iterator behaves the same way)."""
        wrapper_cls, _ = _make_wrapper_class()

        first = _terminal_chunk("response.completed")
        first.response.id = "resp_first"
        second = _terminal_chunk("response.completed")
        second.response.id = "resp_second"

        async def gen():
            yield first
            yield second

        wrapper = wrapper_cls(gen())
        asyncio.run(_drain(wrapper))
        assert wrapper.completed_response.response.id == "resp_first"


class TestProxyOwnershipHookReadsCompletedResponse:
    """End-to-end: the proxy hook reads exactly the attribute the
    wrapper now populates. Pin that the helper still extracts the
    response correctly so the ownership recording path doesn't break."""

    def test_extract_returns_inner_response_object(self):
        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _terminal_chunk("response.completed")

        wrapper = wrapper_cls(gen())
        asyncio.run(_drain(wrapper))

        extracted = ProxyBaseLLMRequestProcessing._extract_completed_responses_response(
            wrapper
        )
        assert extracted is not None
        assert extracted.id == "resp_test"
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

        # Wrap a generator that produces NO terminal event so the
        # wrapper stays at completed_response=None — same shape as the
        # pre-fix bug.
        wrapper_cls, _ = _make_wrapper_class()

        async def gen():
            yield _non_terminal_chunk()

        wrapper = wrapper_cls(gen())

        async def driver():
            async def inner_gen():
                async for c in wrapper:
                    yield c

            # Patch _record_container_owners_from_responses_if_needed to
            # a noop async so the warning branch is exercised in
            # isolation.
            with patch.object(
                ProxyBaseLLMRequestProcessing,
                "_record_container_owners_from_responses_if_needed",
                new=MagicMock(),
            ):
                wrapped = ProxyBaseLLMRequestProcessing._wrap_responses_stream_for_container_ownership(
                    original_stream_response=wrapper,
                    wrapped_generator=inner_gen(),
                    user_api_key_dict=MagicMock(),
                )
                async for _ in wrapped:
                    pass

        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            asyncio.run(driver())

        assert any(
            "Container ownership recording skipped on streaming /v1/responses"
            in r.message
            for r in caplog.records
        ), "silent-skip warning never fired despite completed_response=None"


async def _drain(it):
    out = []
    async for chunk in it:
        out.append(chunk)
    return out

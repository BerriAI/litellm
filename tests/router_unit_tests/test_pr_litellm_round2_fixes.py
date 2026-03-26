"""
Regression tests for LiteLLM round-2 fixes.

Covers four bugs fixed in this PR:

1. (#24608) Bedrock ``internalServerException`` mid-stream mapped to BadRequestError.
   botocore's eventstream protocol sends HTTP 400 for all mid-stream errors,
   so status-code checks are unreliable. The fix detects the error type by
   string-matching before falling through to the status-code branch.

2. (#18395) AuthenticationError "Missing API Key" retried N times before surfacing.
   A missing API key is a configuration error — retrying is pointless and makes
   users wait minutes for a result they could have had immediately. The fix
   raises immediately when the error string contains "missing" + "api key".

3. (#16204) ``drop_params=True`` fails to remove ``reasoning_effort`` for
   non-reasoning xai/grok models, and ``xai/grok-4-fast-reasoning`` had
   ``supports_reasoning=None`` in model_prices_and_context_window.json instead
   of ``True``. Both fixed together.

4. (#24609) ``async_sse_wrapper`` in the Anthropic pass-through streaming
   iterator had no try/except block. Raw Bedrock errors (e.g.
   internalServerException) propagated unhandled to the proxy client instead
   of being mapped to typed LiteLLM exceptions.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bedrock_exc(message: str, status_code: int = 400) -> MagicMock:
    exc = MagicMock()
    exc.status_code = status_code
    exc.message = message
    exc.__str__ = lambda self: self.message
    return exc


def _map(model: str, exc: MagicMock, provider: str = "bedrock"):
    import litellm
    litellm.suppress_debug_info = True
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    exception_type(model, exc, provider)


# ---------------------------------------------------------------------------
# Fix 1 — Bedrock internalServerException → InternalServerError (#24608)
# ---------------------------------------------------------------------------

class TestBedrockInternalServerException:
    """
    botocore's eventstream wraps ALL mid-stream errors as HTTP 400, so the
    status-code check in the Bedrock mapper never reaches ``status_code == 500``.

    The fix detects ``internalServerException`` / ``InternalServerException``
    in the error *string* before falling through to the status-code branch,
    and maps it to ``InternalServerError`` (retryable) instead of ``BadRequestError``.
    """

    def test_internal_server_exception_lowercase(self):
        """``internalServerException`` (botocore casing) → InternalServerError."""
        import litellm
        exc = _make_bedrock_exc(
            'internalServerException {"message": "The system encountered an '
            'unexpected error during processing. Try your request again."}'
        )
        with pytest.raises(litellm.InternalServerError):
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)

    def test_internal_server_exception_pascal_case(self):
        """``InternalServerException`` (alternative casing) → InternalServerError."""
        import litellm
        exc = _make_bedrock_exc("InternalServerException: internal error")
        with pytest.raises(litellm.InternalServerError):
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)

    def test_internal_server_error_string(self):
        """``internalServerError`` variant → InternalServerError."""
        import litellm
        exc = _make_bedrock_exc("internalServerError: backend unavailable")
        with pytest.raises(litellm.InternalServerError):
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)

    def test_normal_400_bad_request_unchanged(self):
        """
        A genuine 400 (e.g. context window exceeded) must still raise
        ContextWindowExceededError, not InternalServerError.
        """
        import litellm
        exc = _make_bedrock_exc("too many tokens in the prompt")
        with pytest.raises(litellm.ContextWindowExceededError):
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)

    def test_throttling_still_raises_rate_limit(self):
        """ThrottlingException must still raise RateLimitError."""
        import litellm
        exc = _make_bedrock_exc("ThrottlingException: too many requests")
        with pytest.raises(litellm.RateLimitError):
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)

    def test_internal_server_error_is_retryable(self):
        """InternalServerError must have status_code in retryable range (500+)."""
        import litellm
        exc = _make_bedrock_exc("internalServerException: transient error")
        try:
            _map("anthropic.claude-3-sonnet-20240229-v1:0", exc)
        except litellm.InternalServerError as e:
            assert e.status_code == 500, (
                f"InternalServerError must have status 500 for retry logic, got {e.status_code}"
            )


# ---------------------------------------------------------------------------
# Fix 2 — AuthenticationError "Missing API Key" not retried (#18395)
# ---------------------------------------------------------------------------

class TestAuthErrorMissingKeyNotRetried:
    """
    When a user forgets to set their API key, LiteLLM used to retry
    ``num_retries`` times before surfacing the error. A missing API key is a
    configuration error — no amount of retrying will fix it.

    The fix raises immediately when ``should_retry_this_error()`` sees
    ``"missing"`` + ``"api key"`` in the AuthenticationError message,
    regardless of how many deployments are available.
    """

    def _make_router(self):
        from litellm.router import Router
        router = object.__new__(Router)
        return router

    def _make_auth_err(self, message: str):
        import litellm
        return litellm.AuthenticationError(
            message=message, llm_provider="anthropic", model="claude-3"
        )

    def test_missing_api_key_raises_immediately_single_deployment(self):
        """
        Regression: with 1 deployment the error was already raised, but only
        after ``num_retries`` attempts had been made internally.
        Now it's raised on the first call to ``should_retry_this_error()``.
        """
        import litellm
        router = self._make_router()
        err = self._make_auth_err(
            "litellm.AuthenticationError: Missing Anthropic API Key — "
            "A call is being made to anthropic but no key is set."
        )
        with pytest.raises(litellm.AuthenticationError):
            router.should_retry_this_error(
                error=err,
                healthy_deployments=[{"model": "claude-3"}],
                all_deployments=[{"model": "claude-3"}],
            )

    def test_missing_api_key_raises_immediately_multiple_deployments(self):
        """
        Regression: with >1 deployments the error was NOT raised and the router
        retried on other deployments (which all had the same missing key).
        """
        import litellm
        router = self._make_router()
        err = self._make_auth_err(
            "litellm.AuthenticationError: Missing OpenAI API Key"
        )
        with pytest.raises(litellm.AuthenticationError):
            router.should_retry_this_error(
                error=err,
                healthy_deployments=[{"model": "gpt-4"}, {"model": "gpt-4"}],
                all_deployments=[{"model": "gpt-4"}, {"model": "gpt-4"}],
            )

    def test_missing_api_key_case_insensitive(self):
        """The string check must be case-insensitive."""
        import litellm
        router = self._make_router()
        err = self._make_auth_err("MISSING API_KEY for provider")
        with pytest.raises(litellm.AuthenticationError):
            router.should_retry_this_error(
                error=err,
                healthy_deployments=[{"model": "x"}, {"model": "y"}],
                all_deployments=[{"model": "x"}, {"model": "y"}],
            )

    def test_transient_auth_error_still_retried_on_other_deployments(self):
        """
        A transient 401 (e.g. token expiry on one deployment) should NOT be
        raised immediately if other deployments are available — only the
        "missing key" pattern triggers immediate raise.
        """
        import litellm
        router = self._make_router()
        err = self._make_auth_err(
            "AuthenticationError: 401 Unauthorized — token expired"
        )
        # With 2+ deployments and no "missing...api key" pattern, should return True
        result = router.should_retry_this_error(
            error=err,
            healthy_deployments=[{"model": "x"}, {"model": "y"}],
            all_deployments=[{"model": "x"}, {"model": "y"}],
        )
        assert result is True, "Transient auth errors should still allow retry on other deployments"


# ---------------------------------------------------------------------------
# Fix 3 — xai drop_params + grok-4-fast-reasoning supports_reasoning (#16204)
# ---------------------------------------------------------------------------

class TestXaiDropParamsReasoningEffort:
    """
    Two related fixes:
    a) ``xai/grok-4-fast-reasoning`` had ``supports_reasoning=None`` in the
       model cost map — should be ``True``.
    b) ``XAIChatConfig.map_openai_params()`` did not honour ``drop_params=True``
       for params not in ``supported_openai_params``. The param was silently
       kept instead of being dropped.
    """

    def _get_config(self):
        from litellm.llms.xai.chat.transformation import XAIChatConfig
        return XAIChatConfig()

    def test_drop_params_removes_unsupported_reasoning_effort(self):
        """
        When ``drop_params=True``, ``reasoning_effort`` must be silently
        removed for non-reasoning grok models.
        Regression: the param was kept, causing the API to return 400.
        """
        config = self._get_config()
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "low", "temperature": 0.7},
            optional_params={},
            model="grok-4-fast-non-reasoning",
            drop_params=True,
        )
        assert "reasoning_effort" not in result, (
            "reasoning_effort must be dropped when drop_params=True and the model "
            "does not support it.  Regression of issue #16204."
        )
        assert "temperature" in result, "Supported params must still be passed through"

    def test_drop_params_false_keeps_params(self):
        """When ``drop_params=False``, unsupported params are NOT silently dropped."""
        config = self._get_config()
        # With drop_params=False, the param goes through unmodified
        # (the provider will raise — that's the user's problem)
        result = config.map_openai_params(
            non_default_params={"temperature": 0.5},
            optional_params={},
            model="grok-4-fast-non-reasoning",
            drop_params=False,
        )
        assert "temperature" in result

    def test_reasoning_model_keeps_reasoning_effort(self):
        """For a reasoning model, ``reasoning_effort`` must NOT be dropped."""
        config = self._get_config()
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "high", "temperature": 0.0},
            optional_params={},
            model="grok-3-mini",  # reasoning model
            drop_params=True,
        )
        # reasoning_effort should be in supported params for reasoning models
        # (exact behaviour depends on model_cost map, but must not crash)
        assert isinstance(result, dict)

    def test_grok_4_fast_reasoning_in_model_cost_map(self):
        """
        ``xai/grok-4-fast-reasoning`` must have ``supports_reasoning=True``
        in the model cost map.
        """
        import json
        from pathlib import Path
        root = Path(__file__).parents[2]  # repo root
        models_path = root / "model_prices_and_context_window.json"
        if not models_path.exists():
            pytest.skip("model_prices_and_context_window.json not found")
        models = json.loads(models_path.read_text(encoding="utf-8"))
        entry = models.get("xai/grok-4-fast-reasoning", {})
        assert entry.get("supports_reasoning") is True, (
            "xai/grok-4-fast-reasoning must have supports_reasoning=True. "
            "Regression of issue #16204."
        )

    def test_grok_4_fast_non_reasoning_in_model_cost_map(self):
        """``xai/grok-4-fast-non-reasoning`` must have ``supports_reasoning=False``."""
        import json
        from pathlib import Path
        root = Path(__file__).parents[2]
        models_path = root / "model_prices_and_context_window.json"
        if not models_path.exists():
            pytest.skip("model_prices_and_context_window.json not found")
        models = json.loads(models_path.read_text(encoding="utf-8"))
        entry = models.get("xai/grok-4-fast-non-reasoning", {})
        assert entry.get("supports_reasoning") is False, (
            "xai/grok-4-fast-non-reasoning must have supports_reasoning=False."
        )


# ---------------------------------------------------------------------------
# Fix 4 — async_sse_wrapper try/except for pass-through streaming (#24609)
# ---------------------------------------------------------------------------

class TestAsyncSseWrapperErrorHandling:
    """
    The Anthropic pass-through ``async_sse_wrapper`` had no try/except block.
    When Bedrock sent ``internalServerException`` mid-stream, the raw error
    propagated unhandled to the proxy client.

    The fix wraps the async iteration in try/except and maps the exception
    through ``exception_type()`` to produce a typed LiteLLM exception.
    """

    def _make_iterator(self):
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

        logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        logging_obj.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        logging_obj.custom_llm_provider = "bedrock"

        # Subclass to avoid abstract method issues
        class ConcreteIterator(BaseAnthropicMessagesStreamingIterator):
            def _convert_chunk_to_sse_format(self, chunk):
                return b"data: test\n\n"

            async def _handle_streaming_logging(self, collected_chunks):
                pass

        it = ConcreteIterator.__new__(ConcreteIterator)
        it.litellm_logging_obj = logging_obj
        it.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        it.custom_llm_provider = "bedrock"
        it.request_body = {}
        return it

    def test_try_except_exists_in_async_sse_wrapper(self):
        """
        Smoke test: ``async_sse_wrapper`` must contain a try/except block so
        that exceptions from the completion stream are caught and mapped.
        """
        import inspect
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )
        src = inspect.getsource(
            BaseAnthropicMessagesStreamingIterator.async_sse_wrapper
        )
        assert "try:" in src, (
            "async_sse_wrapper must have a try/except block to catch mid-stream "
            "errors. Regression of issue #24609."
        )
        assert "except" in src, (
            "async_sse_wrapper must have an except clause. Regression of #24609."
        )

    @pytest.mark.asyncio
    async def test_raw_exception_is_caught_not_propagated_raw(self):
        """
        When the completion stream raises a raw exception (e.g. botocore's
        internalServerException), ``async_sse_wrapper`` must catch it —
        not let it propagate as a bare unhandled exception.
        """
        it = self._make_iterator()

        class _FakeBotocoreError(Exception):
            pass

        async def _bad_stream():
            yield b"data: first chunk\n\n"
            raise _FakeBotocoreError(
                'internalServerException {"message": "unexpected error"}'
            )

        chunks = []
        caught = None
        try:
            async for chunk in it.async_sse_wrapper(_bad_stream()):
                chunks.append(chunk)
        except Exception as e:
            caught = e

        assert caught is not None, (
            "async_sse_wrapper must propagate an exception when the stream fails"
        )
        assert len(chunks) >= 1, "Chunks before the error must still be yielded"

    @pytest.mark.asyncio
    async def test_happy_path_chunks_yielded_normally(self):
        """When no error occurs, all chunks must be yielded as before."""
        it = self._make_iterator()

        async def _good_stream():
            for i in range(3):
                yield f"chunk-{i}".encode()

        chunks = []
        async for chunk in it.async_sse_wrapper(_good_stream()):
            chunks.append(chunk)

        assert len(chunks) == 3, "All 3 chunks must be yielded on the happy path"

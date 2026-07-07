import json

import pytest

from litellm.router_utils.fallback_event_handlers import (
    _crosses_gemini_endpoint_boundary,
    _get_model_group_custom_llm_provider,
    get_fallback_model_group,
    run_async_fallback,
)


class StreamingWrapper:
    def __init__(self):
        self._hidden_params = {"additional_headers": {}}


class FakeRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        return StreamingWrapper()


class AlwaysFailRouter:
    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        raise RuntimeError("fallback model also failed")


@pytest.mark.asyncio
async def test_run_async_fallback_adds_errors_when_opted_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert json.loads(additional_headers["x-litellm-fallback-errors"]) == [
        {
            "message": "upstream limited request",
            "type": "RuntimeError",
            "param": None,
            "code": None,
        }
    ]


@pytest.mark.asyncio
async def test_run_async_fallback_omits_errors_without_opt_in():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    additional_headers = response._hidden_params["additional_headers"]
    assert additional_headers["x-litellm-attempted-fallbacks"] == 1
    assert "x-litellm-fallback-errors" not in additional_headers


@pytest.mark.asyncio
async def test_run_async_fallback_raises_when_all_fallbacks_fail():
    with pytest.raises(RuntimeError, match="fallback model also failed"):
        await run_async_fallback(
            litellm_router=AlwaysFailRouter(),
            fallback_model_group=["fallback-model"],
            original_model_group="primary-model",
            original_exception=RuntimeError("original request failed"),
            max_fallbacks=3,
            fallback_depth=0,
            include_fallback_errors=True,
        )


class RecordingRouter:
    def __init__(self, deployments_by_model_group=None):
        self.received_kwargs = None
        self.deployments_by_model_group = deployments_by_model_group or {}

    def log_retry(self, kwargs, e):
        return kwargs

    def get_model_list(self, model_name=None, team_id=None):
        return self.deployments_by_model_group.get(model_name)

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return StreamingWrapper()


@pytest.mark.asyncio
async def test_run_async_fallback_forwards_include_fallback_errors_to_nested_call():
    """A nested fallback (multi-hop) must keep collecting errors, so the opt-in
    flag has to reach the nested async_function_with_fallbacks call."""
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        include_fallback_errors=True,
    )

    assert router.received_kwargs.get("include_fallback_errors") is True


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_forward_flag_without_opt_in():
    router = RecordingRouter()
    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert "include_fallback_errors" not in router.received_kwargs


@pytest.mark.asyncio
async def test_run_async_fallback_skips_original_model_group():
    response = await run_async_fallback(
        litellm_router=FakeRouter(),
        fallback_model_group=["primary-model", "fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("original failed"),
        max_fallbacks=3,
        fallback_depth=0,
    )

    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(
        fallbacks=fallbacks, model_group="unmatched-model"
    )

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


def _gemini_tool_call_message(tool_call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "provider_specific_fields": {"thought_signature": "VERTEX_SIG"},
                "function": {"name": "test_tool", "arguments": "{}"},
            }
        ],
    }


class TestGetModelGroupCustomLlmProvider:
    """A thought signature is endpoint-bound, so detecting a fallback's
    provider correctly is a prerequisite for knowing whether a boundary
    was crossed."""

    def test_uses_explicit_custom_llm_provider_when_present(self):
        router = RecordingRouter(
            deployments_by_model_group={
                "primary-model": [{"litellm_params": {"custom_llm_provider": "vertex_ai", "model": "gemini-3-pro-preview"}}]
            }
        )

        assert _get_model_group_custom_llm_provider(router, "primary-model") == "vertex_ai"

    def test_infers_custom_llm_provider_from_model_prefix(self):
        router = RecordingRouter(
            deployments_by_model_group={"fallback-model": [{"litellm_params": {"model": "gemini/gemini-3-pro-preview"}}]}
        )

        assert _get_model_group_custom_llm_provider(router, "fallback-model") == "gemini"

    def test_returns_none_when_model_group_has_no_deployments(self):
        router = RecordingRouter()

        assert _get_model_group_custom_llm_provider(router, "unregistered-model") is None


class TestCrossesGeminiEndpointBoundary:
    def test_true_between_vertex_ai_and_gemini(self):
        assert _crosses_gemini_endpoint_boundary("vertex_ai", "gemini") is True
        assert _crosses_gemini_endpoint_boundary("gemini", "vertex_ai") is True

    def test_false_when_providers_match(self):
        assert _crosses_gemini_endpoint_boundary("vertex_ai", "vertex_ai") is False

    def test_false_when_either_side_is_not_a_gemini_endpoint(self):
        assert _crosses_gemini_endpoint_boundary("vertex_ai", "openai") is False
        assert _crosses_gemini_endpoint_boundary(None, "gemini") is False


@pytest.mark.asyncio
async def test_run_async_fallback_strips_foreign_thought_signature_across_gemini_boundary():
    """The Router falling back from a Vertex AI Gemini deployment to its
    Google AI Studio twin (or vice versa) must not replay a thought
    signature minted by the other endpoint, or the fallback call itself
    gets rejected with a 400 'Corrupted thought signature.' error."""
    router = RecordingRouter(
        deployments_by_model_group={
            "primary-model": [{"litellm_params": {"custom_llm_provider": "vertex_ai", "model": "vertex_ai/gemini-3-pro-preview"}}],
            "fallback-model": [{"litellm_params": {"custom_llm_provider": "gemini", "model": "gemini/gemini-3-pro-preview"}}],
        }
    )
    tool_call_id = "call_abc123__thought__VERTEX_SIG"

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        messages=[_gemini_tool_call_message(tool_call_id)],
    )

    tool_call = router.received_kwargs["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == "call_abc123"
    assert "thought_signature" not in tool_call["provider_specific_fields"]


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_strip_signature_within_same_provider():
    """A fallback between two Vertex AI deployments never crosses the
    endpoint boundary, so the signature must be replayed unchanged."""
    router = RecordingRouter(
        deployments_by_model_group={
            "primary-model": [{"litellm_params": {"custom_llm_provider": "vertex_ai", "model": "vertex_ai/gemini-3-pro-preview"}}],
            "fallback-model": [{"litellm_params": {"custom_llm_provider": "vertex_ai", "model": "vertex_ai/gemini-3-flash"}}],
        }
    )
    tool_call_id = "call_abc123__thought__VERTEX_SIG"

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=RuntimeError("upstream limited request"),
        max_fallbacks=3,
        fallback_depth=0,
        messages=[_gemini_tool_call_message(tool_call_id)],
    )

    tool_call = router.received_kwargs["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == tool_call_id
    assert tool_call["provider_specific_fields"]["thought_signature"] == "VERTEX_SIG"

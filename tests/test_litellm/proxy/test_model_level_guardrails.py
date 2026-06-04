"""
Unit tests for model-level guardrails in post_call paths.

Tests verify that guardrails configured via litellm_params.guardrails on a
deployment are merged into request metadata and trigger execution for both
streaming and non-streaming post_call hooks.
"""

import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from litellm.proxy.utils import (
    _check_and_merge_model_level_guardrails,
    _merge_guardrails_with_existing,
)

# ---------------------------------------------------------------------------
# Unit tests for _check_and_merge_model_level_guardrails
# ---------------------------------------------------------------------------


class TestCheckAndMergeModelLevelGuardrails:
    """Tests for the _check_and_merge_model_level_guardrails function."""

    def test_merge_adds_model_guardrails_to_metadata(self):
        """Model-level guardrails are added to metadata.guardrails."""
        data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        mock_router = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.litellm_params.get.return_value = ["openai-moderation"]
        mock_router.get_deployment.return_value = mock_deployment

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        assert "openai-moderation" in result["metadata"]["guardrails"]
        mock_router.get_deployment.assert_called_once_with(model_id="model-uuid-123")

    def test_merge_combines_with_existing_guardrails(self):
        """Model-level guardrails merge with existing request guardrails."""
        data = {
            "model": "gpt-4",
            "metadata": {
                "model_info": {"id": "model-uuid-123"},
                "guardrails": ["existing-guardrail"],
            },
        }
        mock_router = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.litellm_params.get.return_value = ["model-guardrail"]
        mock_router.get_deployment.return_value = mock_deployment

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        assert "existing-guardrail" in result["metadata"]["guardrails"]
        assert "model-guardrail" in result["metadata"]["guardrails"]

    def test_no_duplicates_when_guardrail_already_in_metadata(self):
        """No duplicates when the same guardrail is in both model and request."""
        data = {
            "model": "gpt-4",
            "metadata": {
                "model_info": {"id": "model-uuid-123"},
                "guardrails": ["openai-moderation"],
            },
        }
        mock_router = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.litellm_params.get.return_value = ["openai-moderation"]
        mock_router.get_deployment.return_value = mock_deployment

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        assert result["metadata"]["guardrails"].count("openai-moderation") == 1

    def test_returns_data_unchanged_when_no_router(self):
        """Returns data unchanged when llm_router is None."""
        data = {"model": "gpt-4", "metadata": {}}
        result = _check_and_merge_model_level_guardrails(data=data, llm_router=None)
        assert result is data

    def test_returns_data_unchanged_when_no_model_info(self):
        """Returns data unchanged when metadata has no model_info AND the
        model alias does not resolve to a deployment."""
        data = {"model": "gpt-4", "metadata": {}}
        mock_router = MagicMock()
        # Neither the model_id lookup nor the alias-fallback lookup
        # finds a deployment.
        mock_router.get_deployment.return_value = None
        mock_router.get_deployment_by_model_group_name.return_value = None
        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )
        assert result is data

    def test_returns_data_unchanged_when_deployment_has_no_guardrails(self):
        """Returns data unchanged when deployment has no guardrails configured."""
        data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        mock_router = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.litellm_params.get.return_value = None
        mock_router.get_deployment.return_value = mock_deployment

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        assert result is data

    def test_returns_data_unchanged_when_deployment_not_found(self):
        """Returns data unchanged when router can't find the deployment."""
        data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "nonexistent-id"}},
        }
        mock_router = MagicMock()
        mock_router.get_deployment.return_value = None

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        assert result is data

    def test_returns_new_data_dict(self):
        """Returns a new top-level dict (shallow copy), not the same object."""
        data = {
            "model": "gpt-4",
            "metadata": {
                "model_info": {"id": "model-uuid-123"},
                "guardrails": ["existing"],
            },
        }
        mock_router = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.litellm_params.get.return_value = ["new-guardrail"]
        mock_router.get_deployment.return_value = mock_deployment

        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=mock_router
        )

        # Result is a different top-level dict
        assert result is not data
        # Result should have the merged guardrail
        assert "new-guardrail" in result["metadata"]["guardrails"]
        assert "existing" in result["metadata"]["guardrails"]


# ---------------------------------------------------------------------------
# Regression test: pre_call hook must run exactly once with model-level guardrails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_runs_once_with_model_level_guardrails():
    """
    A guardrail attached at the model level (litellm_params.guardrails) is
    spread into the top-level request kwargs by the router. The proxy pre-call
    loop (async_pre_call_hook) and the deployment-level hook
    (async_pre_call_deployment_hook) must together invoke async_pre_call_hook
    exactly once, not twice.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import CallTypes, UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    class CountingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="counting-guardrail",
                event_hook=GuardrailEventHooks.pre_call,
                default_on=True,
            )
            self.pre_call_count = 0

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            self.pre_call_count += 1
            return data

    guardrail = CountingGuardrail()

    with patch("litellm.callbacks", [guardrail]):
        ProxyLogging._callback_capabilities_cache.clear()
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {},
        }

        # Path A: proxy pre-call loop runs the guardrail and records that it ran
        data = await proxy_logging.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=data,
            call_type="acompletion",
        )

        # Path B: the router spreads the deployment's model-level guardrails into
        # the top-level kwargs, then litellm.acompletion fires the deployment hook
        data["guardrails"] = ["counting-guardrail"]
        await guardrail.async_pre_call_deployment_hook(data, CallTypes.acompletion)

    assert guardrail.pre_call_count == 1


@pytest.mark.asyncio
async def test_pre_call_hook_runs_once_when_hook_returns_fresh_dict():
    """
    async_pre_call_hook may return a brand-new request dict instead of mutating
    or spreading the one it received. The exactly-once marker must live on the
    data that flows downstream, so the deployment hook still skips the guardrail
    even when the proxy loop swapped in a fresh dict that never carried it.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import CallTypes, UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    class FreshDictGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="counting-guardrail",
                event_hook=GuardrailEventHooks.pre_call,
                default_on=True,
            )
            self.pre_call_count = 0

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            self.pre_call_count += 1
            return {"model": data["model"], "messages": data["messages"]}

    guardrail = FreshDictGuardrail()

    with patch("litellm.callbacks", [guardrail]):
        ProxyLogging._callback_capabilities_cache.clear()
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {},
        }

        data = await proxy_logging.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=data,
            call_type="acompletion",
        )

        data["guardrails"] = ["counting-guardrail"]
        await guardrail.async_pre_call_deployment_hook(data, CallTypes.acompletion)

    assert guardrail.pre_call_count == 1


@pytest.mark.asyncio
async def test_deployment_hook_runs_pre_call_without_proxy_loop():
    """
    Direct-SDK usage (litellm.acompletion(..., guardrails=[...]) without the
    proxy) never runs the proxy pre-call loop, so the deployment hook is the
    only place the guardrail executes and it must still run exactly once.
    """
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import CallTypes
    from litellm.types.guardrails import GuardrailEventHooks

    class CountingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="counting-guardrail",
                event_hook=GuardrailEventHooks.pre_call,
                default_on=True,
            )
            self.pre_call_count = 0

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            self.pre_call_count += 1
            return data

    guardrail = CountingGuardrail()

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "guardrails": ["counting-guardrail"],
        "metadata": {},
    }

    await guardrail.async_pre_call_deployment_hook(data, CallTypes.acompletion)

    assert guardrail.pre_call_count == 1


# ---------------------------------------------------------------------------
# Integration test: post_call_success_hook with model-level guardrails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_model_level_guardrail():
    """
    Model-level guardrails configured on a deployment should execute in
    post_call_success_hook (non-streaming path).
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    class TestGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="test-model-guardrail",
                event_hook=GuardrailEventHooks.post_call,
            )
            self.was_called = False

        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            self.was_called = True
            return response

    guardrail = TestGuardrail()

    # Mock router that returns a deployment with guardrails configured
    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["test-model-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    with (
        patch("litellm.callbacks", [guardrail]),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        response = ModelResponse(
            id="resp-1",
            choices=[
                Choices(
                    message=Message(content="Hello", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="gpt-4",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        await proxy_logging.post_call_success_hook(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

        assert guardrail.was_called is True


@pytest.mark.asyncio
async def test_post_call_success_hook_skips_guardrail_not_on_model():
    """
    Guardrails NOT configured on the model should not execute when
    no other source (request body, key, team) enables them.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    class TestGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="unrelated-guardrail",
                event_hook=GuardrailEventHooks.post_call,
            )
            self.was_called = False

        async def async_post_call_success_hook(self, data, user_api_key_dict, response):
            self.was_called = True
            return response

    guardrail = TestGuardrail()

    # Deployment has a DIFFERENT guardrail configured
    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["some-other-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    with (
        patch("litellm.callbacks", [guardrail]),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        response = ModelResponse(
            id="resp-1",
            choices=[
                Choices(
                    message=Message(content="Hello", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="gpt-4",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        await proxy_logging.post_call_success_hook(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

        assert guardrail.was_called is False


# ---------------------------------------------------------------------------
# Integration test: async_post_call_streaming_iterator_hook with model-level guardrails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_iterator_hook_runs_model_level_guardrail():
    """
    Model-level guardrails configured on a deployment should execute in
    async_post_call_streaming_iterator_hook (streaming path) — even when
    `default_on: false` and the guardrail is not in the request body.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    class TestStreamingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="test-model-guardrail",
                event_hook=GuardrailEventHooks.post_call,
            )
            self.was_called = False

        async def async_post_call_streaming_iterator_hook(
            self, user_api_key_dict, response, request_data
        ):
            self.was_called = True
            async for chunk in response:
                yield chunk

    guardrail = TestStreamingGuardrail()

    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["test-model-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    async def fake_response():
        yield "chunk-1"
        yield "chunk-2"

    with (
        patch("litellm.callbacks", [guardrail]),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        request_data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        chunks = []
        async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
            response=fake_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        ):
            chunks.append(chunk)

        assert guardrail.was_called is True
        assert chunks == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_streaming_iterator_hook_skips_guardrail_not_on_model():
    """
    Streaming guardrails NOT configured on the model (and not in the request
    body / key / team) should not execute, even after the dispatcher merge
    runs. Confirms the gate stays closed for unrelated guardrails.
    """
    from litellm.caching.caching import DualCache
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    class TestStreamingGuardrail(CustomGuardrail):
        def __init__(self):
            super().__init__(
                guardrail_name="unrelated-guardrail",
                event_hook=GuardrailEventHooks.post_call,
            )
            self.was_called = False

        async def async_post_call_streaming_iterator_hook(
            self, user_api_key_dict, response, request_data
        ):
            self.was_called = True
            async for chunk in response:
                yield chunk

    guardrail = TestStreamingGuardrail()

    # Deployment has a DIFFERENT guardrail configured
    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["some-other-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    async def fake_response():
        yield "chunk-1"

    with (
        patch("litellm.callbacks", [guardrail]),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        request_data = {
            "model": "gpt-4",
            "metadata": {"model_info": {"id": "model-uuid-123"}},
        }
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        chunks = []
        async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
            response=fake_response(),
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
        ):
            chunks.append(chunk)

        assert guardrail.was_called is False
        assert chunks == ["chunk-1"]


# ---------------------------------------------------------------------------
# Regression: pre_call ordering — _check_and_merge_model_level_guardrails
# must run BEFORE pre_call_hook so DB/UI-configured guardrails fire on
# pre_call paths (#29652; #23774 only covered post_call).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_merges_model_level_guardrails_before_pre_call_hook():
    """
    common_processing_pre_call_logic must merge model-level guardrails into
    data BEFORE proxy_logging_obj.pre_call_hook is invoked. Otherwise
    pre_call guardrails (e.g. apply_guardrail event) never see the
    UI/DB-assigned guardrail name.
    """
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    # Stub router that reports one deployment in the group with one
    # model-level guardrail. Mirrors the real proxy: at pre_call_hook time
    # model_info has been stripped by add_litellm_data_to_request (see
    # veria-ai review on PR #29654) and route_request hasn't yet populated
    # model_info.id — so the resolver has to fall back to the model alias
    # and union guardrails across all deployments in the group.
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None
    mock_router.get_model_list.return_value = [
        {"litellm_params": {"guardrails": ["my-pre-call-guardrail"]}}
    ]

    processing = ProxyBaseLLMRequestProcessing(
        data={
            "model": "my-model",
            "metadata": {},  # model_info already stripped
        }
    )

    captured_pre_call_data: dict = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):
        captured_pre_call_data.update(data)
        return data

    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = fake_pre_call_hook

    # Minimal stubs for the surrounding setup steps in
    # common_processing_pre_call_logic. We only care about the ordering
    # between _check_and_merge_model_level_guardrails and pre_call_hook.
    async def passthrough_add_litellm_data(*, data, **kwargs):
        return data

    proxy_config = MagicMock()
    proxy_config._get_hierarchical_router_settings = AsyncMock(return_value=None)

    # Stop the function before any post-pre_call_hook logic so we can keep
    # the test focused. Raising _StopAfterPreCall in the next await fires
    # right after the guardrail merge + pre_call_hook complete.
    class _StopAfterPreCall(Exception):
        pass

    proxy_config._get_hierarchical_router_settings.side_effect = _StopAfterPreCall()

    with (
        patch(
            "litellm.proxy.common_request_processing.add_litellm_data_to_request",
            side_effect=passthrough_add_litellm_data,
        ),
        patch(
            "litellm.proxy.common_request_processing.litellm.utils.function_setup",
            return_value=(MagicMock(), processing.data),
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            None,
        ),
    ):
        from litellm.proxy._types import UserAPIKeyAuth

        try:
            await processing.common_processing_pre_call_logic(
                request=MagicMock(headers={}, url=MagicMock(path="/v1/chat/completions")),
                general_settings={},
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
                proxy_logging_obj=proxy_logging,
                proxy_config=proxy_config,
                route_type="acompletion",
                version=None,
                user_model=None,
                user_temperature=None,
                user_request_timeout=None,
                user_max_tokens=None,
                user_api_base=None,
                model=None,
                llm_router=mock_router,
            )
        except _StopAfterPreCall:
            pass

    # The pre_call_hook must have received data with the model-level
    # guardrail already merged in. Before the fix, this assertion fails
    # because pre_call_hook saw the original data without merge.
    merged = (captured_pre_call_data.get("metadata") or {}).get("guardrails") or (
        captured_pre_call_data.get("guardrails") or []
    )
    assert "my-pre-call-guardrail" in merged

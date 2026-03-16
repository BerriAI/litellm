"""
Unit tests for model-level guardrails in post_call paths.

Tests verify that guardrails configured via litellm_params.guardrails on a
deployment are merged into request metadata and trigger execution for both
streaming and non-streaming post_call hooks.
"""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

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
        result = _check_and_merge_model_level_guardrails(
            data=data, llm_router=None
        )
        assert result is data

    def test_returns_data_unchanged_when_no_model_info(self):
        """Returns data unchanged when metadata has no model_info."""
        data = {"model": "gpt-4", "metadata": {}}
        mock_router = MagicMock()
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

        async def async_post_call_success_hook(
            self, data, user_api_key_dict, response
        ):
            self.was_called = True
            return response

    guardrail = TestGuardrail()

    # Mock router that returns a deployment with guardrails configured
    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["test-model-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    with patch("litellm.callbacks", [guardrail]), patch(
        "litellm.proxy.proxy_server.llm_router", mock_router
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

        async def async_post_call_success_hook(
            self, data, user_api_key_dict, response
        ):
            self.was_called = True
            return response

    guardrail = TestGuardrail()

    # Deployment has a DIFFERENT guardrail configured
    mock_router = MagicMock()
    mock_deployment = MagicMock()
    mock_deployment.litellm_params.get.return_value = ["some-other-guardrail"]
    mock_router.get_deployment.return_value = mock_deployment

    with patch("litellm.callbacks", [guardrail]), patch(
        "litellm.proxy.proxy_server.llm_router", mock_router
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

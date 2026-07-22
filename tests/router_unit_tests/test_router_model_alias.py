"""
Tests for Router model alias resolution in streaming completion

Ensures that when a model group alias is used, the resolved deployment
model name is passed to litellm.completion() in input_kwargs, not the alias.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router


def test_router_completion_uses_resolved_model_name_for_alias():
    """
    When a model group alias resolves to a deployment, the input_kwargs
    passed to litellm.completion() should use the real model name
    (e.g. "openai/gpt-4o-mini") not the alias (e.g. "my-alias").
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            }
        ],
        model_group_alias={"my-alias": "gpt-4o-mini"},
    )

    with patch.object(router, "_get_client", return_value=MagicMock()), patch(
        "litellm.completion"
    ) as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock()],
            _hidden_params={},
        )

        router._completion(
            model="my-alias",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi",
        )

        # Verify the model kwarg passed to litellm.completion is the
        # resolved deployment model, not the alias
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini", (
            f"Expected model='openai/gpt-4o-mini', got {call_kwargs['model']}"
        )


@pytest.mark.asyncio
async def test_router_acompletion_uses_resolved_model_name_for_alias():
    """
    Same as above but for the async path.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            }
        ],
        model_group_alias={"my-alias": "gpt-4o-mini"},
    )

    with patch.object(
        router, "_get_async_openai_model_client", return_value=MagicMock()
    ), patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = MagicMock(
            choices=[MagicMock()],
            _hidden_params={},
        )

        await router._acompletion(
            model="my-alias",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi",
        )

        # Verify the model kwarg passed to litellm.acompletion is the
        # resolved deployment model, not the alias
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini", (
            f"Expected model='openai/gpt-4o-mini', got {call_kwargs['model']}"
        )


def test_router_completion_records_deployment_model_in_metadata():
    """
    The resolved deployment model name should be recorded in metadata
    for logging/tracking purposes.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test",
                },
            }
        ],
        model_group_alias={"my-alias": "gpt-4o-mini"},
    )

    with patch.object(router, "_get_client", return_value=MagicMock()), patch(
        "litellm.completion"
    ) as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock()],
            _hidden_params={},
        )

        router._completion(
            model="my-alias",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi",
        )

        call_kwargs = mock_completion.call_args.kwargs
        metadata = call_kwargs.get("metadata", {}) or {}
        # The deployment model name should be recorded in metadata
        assert metadata.get("deployment") == "openai/gpt-4o-mini", (
            f"Expected deployment='openai/gpt-4o-mini', got {metadata.get('deployment')}"
        )
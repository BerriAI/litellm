"""
Tests that Router.aspeech() calls _update_kwargs_with_deployment() so
deployment metadata (model_info, model_id) is stored in kwargs for cost
calculation.

Without this call, get_router_model_id() returns None, the cost
calculator falls back to the bare model name (no pricing entry), and
spend is always $0 for custom-priced TTS deployments.

See: https://github.com/BerriAI/litellm/issues/27390
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router


def _make_tts_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "my-tts",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "fake-key",
                },
                "model_info": {
                    "id": "test-deployment-id",
                    "mode": "audio_speech",
                    "input_cost_per_character": 0.000015,
                },
            },
        ],
    )


class TestAspeechDeploymentKwargs:
    @pytest.mark.asyncio
    async def test_aspeech_calls_update_kwargs_with_deployment(self):
        """aspeech must call _update_kwargs_with_deployment so deployment
        metadata is available for cost calculation."""
        router = _make_tts_router()

        with (
            patch.object(
                router,
                "async_get_available_deployment",
                new_callable=AsyncMock,
                return_value={
                    "model_name": "my-tts",
                    "litellm_params": {
                        "model": "openai/tts-1",
                        "api_key": "fake-key",
                    },
                    "model_info": {
                        "id": "test-deployment-id",
                    },
                },
            ),
            patch.object(
                router,
                "_update_kwargs_with_deployment",
                wraps=router._update_kwargs_with_deployment,
            ) as mock_update,
            patch("litellm.aspeech", new_callable=AsyncMock, return_value=MagicMock()),
        ):
            await router.aspeech(model="my-tts", input="hello world", voice="alloy")

        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args
        assert call_kwargs.kwargs.get("function_name") == "aspeech" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "aspeech"
        )

    @pytest.mark.asyncio
    async def test_aspeech_stores_model_info_in_metadata(self):
        """After aspeech, kwargs metadata should contain model_info with
        the deployment ID needed for cost lookup."""
        router = _make_tts_router()

        deployment = {
            "model_name": "my-tts",
            "litellm_params": {
                "model": "openai/tts-1",
                "api_key": "fake-key",
            },
            "model_info": {
                "id": "test-deployment-id",
                "input_cost_per_character": 0.000015,
            },
        }
        captured_kwargs = {}

        async def capture_aspeech(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch.object(
                router,
                "async_get_available_deployment",
                new_callable=AsyncMock,
                return_value=deployment,
            ),
            patch("litellm.aspeech", side_effect=capture_aspeech),
        ):
            await router.aspeech(model="my-tts", input="hello world", voice="alloy")

        metadata = captured_kwargs.get("metadata", {})
        assert metadata.get("model_info") is not None
        assert metadata["model_info"]["id"] == "test-deployment-id"

"""
Unit tests for HPC-AI OpenAI-compatible configuration.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm import completion
from litellm.llms.hpc_ai.chat.transformation import HpcAiConfig


class TestHpcAiConfig:
    def test_validate_environment_sets_auth_header(self):
        config = HpcAiConfig()
        headers = {}
        api_key = "fake-hpc-ai-key"
        result = config.validate_environment(
            headers=headers,
            model="hpc_ai/minimax/minimax-m2.5",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    @pytest.mark.respx()
    def test_hpc_ai_completion_mock(self, respx_mock):
        litellm.disable_aiohttp_transport = True

        api_key = "fake-hpc-ai-key"
        api_base = "https://api.hpc-ai.com/inference/v1"
        model = "hpc_ai/minimax/minimax-m2.5"
        model_name = "minimax/minimax-m2.5"

        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-hpc-1",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from HPC-AI.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "total_tokens": 13,
                },
            },
            status_code=200,
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base=api_base,
        )

        assert response.choices[0].message.content == "Hello from HPC-AI."

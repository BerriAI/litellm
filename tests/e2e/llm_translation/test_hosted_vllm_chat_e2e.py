"""Live e2e: hosted_vllm OpenAI-compatible /chat/completions.

Self-hosted vLLM (or any OpenAI-compatible server) is a confirmed production
backend. Register hosted_vllm/* via /model/new and expect real completion text.
"""

from __future__ import annotations

import os

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e


class TestHostedVllmChat:
    @pytest.mark.covers(
        "llm.chat_completions.hosted_vllm.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_hosted_vllm_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        (api_base,) = require_env("HOSTED_VLLM_API_BASE")
        api_key = (os.environ.get("HOSTED_VLLM_API_KEY") or "").strip() or None
        backend = (
            os.environ.get("HOSTED_VLLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
        ).strip()
        model = f"e2e-vllm-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=f"hosted_vllm/{backend}",
                api_base=api_base,
                api_key=api_key,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"hosted_vllm chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"hosted_vllm empty content: {response}"

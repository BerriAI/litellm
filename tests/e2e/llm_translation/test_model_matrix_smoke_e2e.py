"""Vendor §6 smoke model matrix: basic chat across provider families (LIT-4778).

Each row registers a live deployment and asserts a non-empty chat completion.
This is the smoke set, not the full matrix; missing credentials hard-fail per e2e rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class SmokeModel:
    id: str
    backend: str
    params: LiteLLMParamsBody


SMOKE_MODELS: tuple[SmokeModel, ...] = (
    SmokeModel(
        id="openai-gpt-4o-mini",
        backend="openai/gpt-4o-mini",
        params=LiteLLMParamsBody(
            model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"
        ),
    ),
    SmokeModel(
        id="openai-gpt-4o",
        backend="openai/gpt-4o",
        params=LiteLLMParamsBody(model="openai/gpt-4o", api_key="os.environ/OPENAI_API_KEY"),
    ),
    SmokeModel(
        id="anthropic-haiku",
        backend="anthropic/claude-haiku-4-5",
        params=LiteLLMParamsBody(
            model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY"
        ),
    ),
    SmokeModel(
        id="bedrock-claude-haiku",
        backend="bedrock/claude-haiku",
        params=LiteLLMParamsBody(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    ),
    SmokeModel(
        id="gemini-flash",
        backend="gemini/gemini-2.5-flash",
        params=LiteLLMParamsBody(
            model="gemini/gemini-2.5-flash", api_key="os.environ/GEMINI_API_KEY"
        ),
    ),
)


class TestModelMatrixSmoke:
    @pytest.mark.covers("llm.chat_completions.openai.basic.nonstream.works")
    @pytest.mark.parametrize("smoke", SMOKE_MODELS, ids=[s.id for s in SMOKE_MODELS])
    def test_smoke_model_chat_returns_content(
        self, proxy: ProxyClient, resources: ResourceManager, smoke: SmokeModel
    ) -> None:
        model = f"e2e-smoke-{smoke.id}-{unique_marker()}"
        model_id = proxy.create_model(model, smoke.params)
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word confirmed. {unique_marker()}",
                        )
                    ],
                    max_completion_tokens=32,
                    temperature=0.0 if "gpt-4o" in smoke.backend else None,
                ),
            )
        )
        assert response.choices, f"{smoke.id}: empty choices: {response}"
        message = response.choices[0].message
        assert message is not None and (message.content or "").strip(), (
            f"{smoke.id}: empty assistant content: {response}"
        )

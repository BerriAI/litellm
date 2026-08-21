from __future__ import annotations

from dataclasses import dataclass

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatMessage, CountTokensBody, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class CountTokensProvider:
    name: str
    litellm_params: LiteLLMParamsBody


COUNT_TOKENS_PROVIDERS = (
    pytest.param(
        CountTokensProvider(
            name="anthropic",
            litellm_params=LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5",
                api_key="os.environ/ANTHROPIC_API_KEY",
            ),
        ),
        id="anthropic",
        marks=pytest.mark.covers("llm.messages.anthropic.count_tokens.nonstream.works"),
    ),
    pytest.param(
        CountTokensProvider(
            name="openai",
            litellm_params=LiteLLMParamsBody(
                model="openai/gpt-5.5",
                api_key="os.environ/OPENAI_API_KEY",
            ),
        ),
        id="openai",
        marks=pytest.mark.covers("llm.messages.openai.count_tokens.nonstream.works"),
    ),
    pytest.param(
        CountTokensProvider(
            name="bedrock",
            litellm_params=LiteLLMParamsBody(
                model="bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                aws_region_name="us-east-1",
            ),
        ),
        id="bedrock",
        marks=pytest.mark.covers("llm.messages.bedrock_invoke.count_tokens.nonstream.works"),
    ),
    pytest.param(
        CountTokensProvider(
            name="gemini",
            litellm_params=LiteLLMParamsBody(
                model="gemini/gemini-2.5-flash",
                api_key="os.environ/GEMINI_API_KEY",
            ),
        ),
        id="gemini",
        marks=pytest.mark.covers("llm.messages.gemini.count_tokens.nonstream.works"),
    ),
    pytest.param(
        CountTokensProvider(
            name="azure-openai",
            litellm_params=LiteLLMParamsBody(
                model="azure/gpt-5.5",
                api_base="os.environ/AZURE_API_BASE",
                api_key="os.environ/AZURE_API_KEY",
            ),
        ),
        id="azure-openai",
        marks=pytest.mark.covers("llm.messages.azure_openai.count_tokens.nonstream.works"),
    ),
    pytest.param(
        CountTokensProvider(
            name="vertex-ai",
            litellm_params=LiteLLMParamsBody(
                model="vertex_ai/gemini-2.5-flash",
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="us-central1",
            ),
        ),
        id="vertex-ai",
        marks=pytest.mark.covers("llm.messages.vertex.count_tokens.nonstream.works"),
    ),
)

SHORT_MESSAGE = "Count these tokens."
LONG_MESSAGE = " ".join(f"count-token-{index}" for index in range(256))


def _count_tokens(proxy: ProxyClient, key: str, model: str, content: str) -> int:
    response = unwrap(
        proxy.count_tokens(
            key,
            CountTokensBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
            ),
        )
    )
    return response.input_tokens


@pytest.mark.parametrize("provider", COUNT_TOKENS_PROVIDERS)
def test_count_tokens_returns_prompt_dependent_count(
    proxy: ProxyClient,
    resources: ResourceManager,
    provider: CountTokensProvider,
) -> None:
    model = f"e2e-count-tokens-{provider.name}-{unique_marker()}"
    model_id = proxy.create_model(model, provider.litellm_params)
    resources.defer(lambda: proxy.delete_model(model_id))
    key = resources.key(models=[model])

    short_count = _count_tokens(proxy, key, model, SHORT_MESSAGE)
    long_count = _count_tokens(proxy, key, model, LONG_MESSAGE)

    assert short_count > 0, f"{provider.name} returned a non-positive count: {short_count}"
    assert long_count > short_count, (
        f"{provider.name} did not count the larger prompt as larger: short={short_count}, long={long_count}"
    )

"""Live e2e for model-specific request features: service_tier and prompt caching.

Each case asserts the feature took effect, not just a 200.

service_tier is an OpenAI concept. The proxy forwards it and the provider echoes
the tier back on the response, so sending a non-default tier ("flex") and reading
it back off ``service_tier`` proves the param was honored end to end; litellm's own
default injection would report "default", so a "flex" echo can only come from the
request being forwarded. Bedrock and Vertex do not accept service_tier, so that
cell is OpenAI-only by design.

Prompt caching is asserted through provider prompt-cache usage tokens. The
deterministic path is explicit ``cache_control`` on an Anthropic-family model
(here Bedrock's Claude): a large cacheable prefix is sent twice and the second
call must report ``cache_read_input_tokens > 0``. OpenAI and Gemini only offer
implicit automatic caching, which does not deterministically produce a cache read
within a test window (verified: repeated >3k-token prompts kept
``prompt_tokens_details.cached_tokens`` at 0), so those caching cells are out of
scope here and covered only by the explicit-cache-control Bedrock case.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

SERVICE_TIER = "flex"
CACHE_MIN_READ_TOKENS = 1


class CacheControl(BaseModel):
    type: str = "ephemeral"


class CacheTextBlock(BaseModel):
    type: str = "text"
    text: str
    cache_control: CacheControl | None = None


class RichMessage(BaseModel):
    role: str
    content: list[CacheTextBlock]


class CacheChatBody(BaseModel):
    model: str
    messages: list[RichMessage]
    max_tokens: int


def cacheable_prefix() -> str:
    return (
        "You are a policy compliance auditor. The following corpus is the immutable "
        "reference the assistant must consult on every turn. "
    ) + ("Clause: obey all safety, formatting, and citation rules exactly. " * 400)


def post_chat(client: PassthroughClient, key: str, body: BaseModel) -> ChatResponse:
    return unwrap(
        client.gateway.transport.post(
            "/chat/completions",
            headers=client.gateway.transport.bearer(key),
            json=body,
            response_type=ChatResponse,
        )
    )


class TestServiceTier:
    @pytest.mark.covers("llm.chat_completions.openai.service_tier.works", exercised_on=[])
    def test_openai_service_tier_is_echoed(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-service-tier-{unique_marker()}"
        model_id = client.gateway.create_model(
            model, LiteLLMParamsBody(model="openai/gpt-5.5", api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.gateway.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.gateway.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="reply with one word")],
                    max_tokens=64,
                    service_tier=SERVICE_TIER,
                ),
            )
        )
        assert response.service_tier == SERVICE_TIER, (
            f"service_tier not honored: sent {SERVICE_TIER!r}, response reported "
            f"{response.service_tier!r} ({response})"
        )


class TestPromptCaching:
    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.prompt_cache_5m.nonstream.cache_hit", exercised_on=[]
    )
    def test_bedrock_cache_control_produces_cache_read(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-bedrock-cache-{unique_marker()}"
        model_id = client.gateway.create_model(
            model,
            LiteLLMParamsBody(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                aws_region_name="us-east-1",
            ),
        )
        resources.defer(lambda: client.gateway.delete_model(model_id))
        key = resources.key()

        body = CacheChatBody(
            model=model,
            max_tokens=32,
            messages=[
                RichMessage(
                    role="user",
                    content=[
                        CacheTextBlock(text=cacheable_prefix(), cache_control=CacheControl()),
                        CacheTextBlock(text="Answer in one word: acknowledged?"),
                    ],
                )
            ],
        )

        first = post_chat(client, key, body)
        assert first.usage is not None, f"first call reported no usage: {first}"

        second = post_chat(client, key, body)
        assert second.usage is not None, f"second call reported no usage: {second}"
        cache_read = second.usage.cache_read_input_tokens
        assert cache_read is not None and cache_read >= CACHE_MIN_READ_TOKENS, (
            "second identical request did not read the prompt cache: "
            f"cache_read_input_tokens={cache_read!r} (usage={second.usage})"
        )

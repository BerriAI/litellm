"""Live e2e for model-specific request features: service_tier and prompt caching.

Each case asserts the feature took effect, not just a 200.

service_tier is an OpenAI concept. The proxy forwards it and the provider echoes
the tier back on the response, so sending a non-default tier ("priority") and
reading it back off ``service_tier`` proves the param was honored end to end;
litellm's own default injection (and service_tier="auto") both report "default",
so a "priority" echo can only come from the request being forwarded. "flex" is
avoided here because it is capacity-constrained and returns a transient 429 when
flex resources are unavailable. Bedrock and Vertex do not accept service_tier, so
that cell is OpenAI-only by design.

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
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from model_matrix import BEDROCK_ANTHROPIC_CHAT, OPENAI_CHAT
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

SERVICE_TIER = "priority"
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


class CacheDirective(BaseModel):
    """litellm per-request cache control. ``no-cache`` forces the proxy to skip its
    own response cache and make a fresh provider call, so the second identical
    request actually reaches Bedrock and reads the provider prompt cache instead of
    being served the first response verbatim (which would report cache_read=0)."""

    model_config = ConfigDict(populate_by_name=True)
    no_cache: bool = Field(default=True, alias="no-cache")


class CacheChatBody(BaseModel):
    model: str
    messages: list[RichMessage]
    max_tokens: int
    cache: CacheDirective = CacheDirective()


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
    @pytest.mark.covers(
        "llm.chat_completions.openai.service_tier.nonstream.works", exercised_on=[]
    )
    def test_openai_service_tier_is_echoed(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-service-tier-{unique_marker()}"
        model_id = client.gateway.create_model(
            model,
            LiteLLMParamsBody(
                model=OPENAI_CHAT.backend, api_key="os.environ/OPENAI_API_KEY"
            ),
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
        "llm.chat_completions.bedrock_converse.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    def test_bedrock_cache_control_produces_cache_read(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-bedrock-cache-{unique_marker()}"
        model_id = client.gateway.create_model(
            model,
            LiteLLMParamsBody(
                model=BEDROCK_ANTHROPIC_CHAT.backend,
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
                        CacheTextBlock(
                            text=cacheable_prefix(), cache_control=CacheControl()
                        ),
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

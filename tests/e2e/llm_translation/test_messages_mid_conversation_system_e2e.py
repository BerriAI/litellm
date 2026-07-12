"""Live e2e: mid-conversation ``role: "system"`` handling on the Bedrock Invoke
/v1/messages path is model-aware (PRs #32578, #32831, #32882).

Models flagged ``supports_mid_conversation_system`` in the cost map (Claude 4.8+
and the 5 family) must keep a mid-conversation system reminder in place inside
``messages`` so the top-level ``system`` prefix stays byte-identical and the
prompt cache written on turn one is read back in full on turn two. Models
without the flag (Claude 4.7 and older) reject the role inside ``messages``
outright, so the proxy must hoist the reminder into the top-level ``system``
field and the call must still return a completion instead of a provider 400.

The conversation shape mirrors what Claude Code sends mid-session: a cached
system prompt, a user turn carrying its own ``cache_control`` breakpoint, a
``role: "system"`` reminder, an assistant turn, and a fresh user turn. The
message-turn breakpoint is what makes the cache assertion able to fail: a cache
entry whose prefix spans ``system`` plus message turns is invalidated when the
reminder is hoisted (the ``system`` field mutates and a turn disappears from
``messages``), while an entry ending at the system block itself would survive
the hoist and mask the regression.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, unwrap
from endpoints_client import AnthropicContentBlock, EndpointsClient
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

FLAGGED_INVOKE_MODEL = "bedrock/invoke/us.anthropic.claude-sonnet-5"
UNFLAGGED_INVOKE_MODEL = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-east-1"
CACHE_PRIMING_DEADLINE_SECONDS = 60.0
CACHE_PRIMING_INTERVAL_SECONDS = 3.0


class CacheControl(BaseModel):
    type: str = "ephemeral"


class TextBlock(BaseModel):
    type: str = "text"
    text: str
    cache_control: CacheControl | None = None


class MessageTurn(BaseModel):
    role: str
    content: list[TextBlock]


class MessagesBody(BaseModel):
    model: str
    max_tokens: int = 64
    system: list[TextBlock]
    messages: list[MessageTurn]


class MessagesUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesCompletion(BaseModel):
    role: str | None = None
    content: list[AnthropicContentBlock] = []
    usage: MessagesUsage = MessagesUsage()

    @property
    def text(self) -> str:
        return "".join(block.text or "" for block in self.content)


def _cacheable_system_block(marker: str) -> TextBlock:
    """A system prompt comfortably above Sonnet's 1024-token minimum cacheable
    size, unique per run so no other run's cache entry can satisfy the read."""
    text = " ".join(
        f"Reference paragraph {index} for run {marker}." for index in range(300)
    )
    return TextBlock(text=text, cache_control=CacheControl())


def _user_turn(text: str, *, cached: bool = False) -> MessageTurn:
    block = TextBlock(text=text, cache_control=CacheControl() if cached else None)
    return MessageTurn(role="user", content=[block])


def _system_reminder_turn() -> MessageTurn:
    return MessageTurn(
        role="system",
        content=[
            TextBlock(
                text="<system-reminder>Answer with exactly one word.</system-reminder>"
            )
        ],
    )


def _post_messages(
    client: EndpointsClient, key: str, body: MessagesBody
) -> Result[MessagesCompletion]:
    return client.gateway.transport.post(
        "/v1/messages",
        headers=client.gateway.transport.bearer(key),
        json=body,
        response_type=MessagesCompletion,
    )


def _register_invoke_deployment(
    client: EndpointsClient, resources: ResourceManager, bedrock_model: str
) -> str:
    model = f"e2e-midsys-{unique_marker()}"
    model_id = client.create_model(
        model, LiteLLMParamsBody(model=bedrock_model, aws_region_name=AWS_REGION)
    )
    resources.defer(lambda: client.delete_model(model_id))
    return model


def _first_turn_user_text(marker: str) -> str:
    """A first user turn heavy enough (hundreds of tokens) that losing its cache
    entry is unambiguous in the usage numbers, unique per attempt so priming
    retries never depend on the proxy's response cache behavior."""
    notes = " ".join(f"Session note {index} for attempt {marker}." for index in range(100))
    return f"Reply with one word.\n{notes}"


class PrimedCache(BaseModel):
    first_user_text: str
    prefix_read_tokens: int
    first_turn_creation_tokens: int

    @property
    def full_prefix_tokens(self) -> int:
        return self.prefix_read_tokens + self.first_turn_creation_tokens


def _prime_prompt_cache(
    client: EndpointsClient, key: str, model: str, system_block: TextBlock
) -> PrimedCache:
    """Send first-turn calls (fresh cache-marked user turn each attempt,
    identical system prefix) until one both reads the system prefix back from
    cache and writes its own user-turn chunk, proving the cache is live in both
    directions. Only the pre-reminder turn is ever retried here, so retries can
    never warm a mutated-prefix cache entry and mask the regression the second
    turn asserts on."""
    deadline = time.monotonic() + CACHE_PRIMING_DEADLINE_SECONDS
    while True:
        user_text = _first_turn_user_text(unique_marker())
        body = MessagesBody(
            model=model,
            system=[system_block],
            messages=[_user_turn(user_text, cached=True)],
        )
        usage = unwrap(_post_messages(client, key, body)).usage
        if usage.cache_read_input_tokens > 0 and usage.cache_creation_input_tokens > 0:
            return PrimedCache(
                first_user_text=user_text,
                prefix_read_tokens=usage.cache_read_input_tokens,
                first_turn_creation_tokens=usage.cache_creation_input_tokens,
            )
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{model}: prompt cache never became readable within "
                f"{CACHE_PRIMING_DEADLINE_SECONDS}s (last usage: {usage})"
            )
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)


class TestBedrockInvokeMidConversationSystem:
    @pytest.mark.covers(
        "llm.messages.bedrock_invoke.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _register_invoke_deployment(
            endpoints_client, resources, FLAGGED_INVOKE_MODEL
        )
        key = resources.key()
        system_block = _cacheable_system_block(unique_marker())

        primed = _prime_prompt_cache(endpoints_client, key, model, system_block)

        reminder_turn_body = MessagesBody(
            model=model,
            system=[system_block],
            messages=[
                _user_turn(primed.first_user_text, cached=True),
                _system_reminder_turn(),
                MessageTurn(role="assistant", content=[TextBlock(text="OK.")]),
                _user_turn("Reply with one word again.", cached=True),
            ],
        )
        second = unwrap(_post_messages(endpoints_client, key, reminder_turn_body))

        assert second.text.strip(), (
            f"{model}: reminder turn returned no completion text"
        )
        assert second.usage.cache_read_input_tokens >= primed.full_prefix_tokens, (
            f"{model}: turn with a mid-conversation system reminder read "
            f"{second.usage.cache_read_input_tokens} cached tokens, expected at "
            f"least the {primed.full_prefix_tokens} cached on turn one "
            f"({primed.prefix_read_tokens} system prefix + "
            f"{primed.first_turn_creation_tokens} first user turn); the reminder "
            f"was hoisted into the top-level system field, which mutates the "
            f"cached prefix and re-bills the conversation at cache-write pricing"
        )

    @pytest.mark.covers(
        "llm.messages.bedrock_invoke.mid_conversation_system.nonstream.works",
        exercised_on=[],
    )
    def test_unflagged_model_hoists_system_reminder_and_succeeds(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = _register_invoke_deployment(
            endpoints_client, resources, UNFLAGGED_INVOKE_MODEL
        )
        key = resources.key()

        body = MessagesBody(
            model=model,
            system=[TextBlock(text="You are terse.")],
            messages=[
                _user_turn(f"Say hi. Run {unique_marker()}."),
                _system_reminder_turn(),
                MessageTurn(role="assistant", content=[TextBlock(text="Hi.")]),
                _user_turn("Say bye."),
            ],
        )
        completion = unwrap(_post_messages(endpoints_client, key, body))

        assert completion.role == "assistant", (
            f"{model}: unexpected role {completion.role!r}"
        )
        assert completion.text.strip(), (
            f"{model}: conversation with a mid-conversation system reminder "
            f"returned no text; the reminder was forwarded in place to a model "
            f"that rejects role 'system' inside messages instead of being hoisted"
        )

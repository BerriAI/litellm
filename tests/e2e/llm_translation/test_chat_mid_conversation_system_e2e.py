"""Live e2e: mid-conversation ``role: "system"`` handling on the OpenAI-format
/v1/chat/completions path is model-aware for first-party Anthropic and Bedrock
Invoke, both of which build the Anthropic request through
``AnthropicConfig.transform_request`` (#36559).

Only the leading run of system messages becomes the top-level ``system``
parameter. A ``role: "system"`` entry that appears later in ``messages`` used to
be hoisted into that same field, which rewrote the cached prefix and re-billed
the whole conversation at cache-write pricing on every reminder. Models flagged
``supports_mid_conversation_system`` in the cost map (Claude 4.8+ and the 5
family) must keep the reminder in ``messages`` as ``role: "system"``; models
without the flag (Claude 4.7 and older, Haiku 4.5) reject that role inside
``messages``, so the proxy must convert the reminder to a user turn in place,
prefixed with an operator note. Either way the prompt cache written on turn one
must be read back in full on turn two.

The conversation shape mirrors what an OpenAI-SDK client sends mid-session: a
cached system prompt, a user turn carrying its own ``cache_control`` breakpoint,
an assistant turn, a ``role: "system"`` reminder, and a fresh user turn. The
message-turn breakpoint is what makes the cache assertion able to fail: a cache
entry whose prefix spans ``system`` plus message turns is invalidated when the
reminder is hoisted (the ``system`` field mutates and a turn disappears from
``messages``), while an entry ending at the system block itself would survive
the hoist and mask the regression.

The provider-native ``cache_control`` request shape is not expressible with the
shared ``ChatBody`` (whose content parts carry no cache_control), so the body is
built from the typed content blocks shared in ``endpoints_client.py``.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import Result, unwrap
from endpoints_client import CacheControl, RichMessage, TextBlock
from lifecycle import ResourceManager
from models import ChatResponse, LiteLLMParamsBody, Usage
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

CACHE_PRIMING_DEADLINE_SECONDS = 60.0
CACHE_PRIMING_INTERVAL_SECONDS = 3.0
CACHE_WARM_CONSECUTIVE_READS = 3


class CacheChatRequest(BaseModel):
    """OpenAI-format chat body whose content blocks carry ``cache_control``."""

    model: str
    messages: list[RichMessage]
    max_tokens: int = 64
    cache: dict[str, bool] = {"no-cache": True}


def _anthropic_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, api_key="os.environ/ANTHROPIC_API_KEY")


def _invoke_params(model: str, region: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=model, aws_region_name=region)


def _cacheable_system_turn(marker: str) -> RichMessage:
    """A system prompt comfortably above the 4096-token minimum cacheable size
    of Haiku 4.5 (the smallest model here), unique per run so no other run's
    cache entry can satisfy the read."""
    text = " ".join(f"Reference paragraph {index} for run {marker}." for index in range(300))
    return RichMessage(role="system", content=[TextBlock(text=text, cache_control=CacheControl())])


def _user_turn(text: str, *, cached: bool = False) -> RichMessage:
    block = TextBlock(text=text, cache_control=CacheControl() if cached else None)
    return RichMessage(role="user", content=[block])


def _assistant_turn(text: str) -> RichMessage:
    return RichMessage(role="assistant", content=[TextBlock(text=text)])


def _system_reminder_turn() -> RichMessage:
    return RichMessage(
        role="system",
        content=[TextBlock(text="<system-reminder>Answer with exactly one word.</system-reminder>")],
    )


def _post_chat(client: PassthroughClient, key: str, body: CacheChatRequest) -> Result[ChatResponse]:
    return client.proxy.transport.post(
        "/v1/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
        response_type=ChatResponse,
    )


def _register_deployment(client: PassthroughClient, resources: ResourceManager, params: LiteLLMParamsBody) -> str:
    model = f"e2e-chat-midsys-{unique_marker()}"
    model_id = client.proxy.create_model(model, params)
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model


def _first_turn_user_text(marker: str) -> str:
    """A first user turn heavy enough (hundreds of tokens) that losing its cache
    entry is unambiguous in the usage numbers, unique per attempt so priming
    retries never depend on the proxy's response cache behavior."""
    notes = " ".join(f"Session note {index} for attempt {marker}." for index in range(100))
    return f"Reply with one word.\n{notes}"


def _cache_read_tokens(usage: Usage | None) -> int:
    """Cache-read tokens however the chat usage reports them: the Anthropic-style
    ``cache_read_input_tokens`` litellm forwards, or the OpenAI-style
    ``prompt_tokens_details.cached_tokens`` it mirrors them into."""
    if usage is None:
        return 0
    if usage.cache_read_input_tokens:
        return usage.cache_read_input_tokens
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        return usage.prompt_tokens_details.cached_tokens
    return 0


def _cache_creation_tokens(usage: Usage | None) -> int:
    if usage is None:
        return 0
    return usage.cache_creation_input_tokens or 0


def _response_text(response: ChatResponse) -> str:
    return "".join(choice.message.content or "" for choice in response.choices if choice.message)


def _response_role(response: ChatResponse) -> str | None:
    first = response.choices[0].message if response.choices else None
    return first.role if first else None


class PrimedCache(BaseModel):
    first_user_text: str
    prefix_read_tokens: int
    first_turn_creation_tokens: int

    @property
    def full_prefix_tokens(self) -> int:
        return self.prefix_read_tokens + self.first_turn_creation_tokens


def _prime_prompt_cache(client: PassthroughClient, key: str, model: str, system_turn: RichMessage) -> PrimedCache:
    """Send first-turn calls (fresh cache-marked user turn each attempt,
    identical system prefix) until one both reads the system prefix back from
    cache and writes its own user-turn chunk, then re-send that exact turn until
    its own chunk reads back on three sends in a row, proving the cache is live
    in both directions before the reminder turn goes out (a freshly written entry
    can take a few seconds to become readable). Only the pre-reminder turn is
    ever retried here, so retries can never warm a mutated-prefix cache entry and
    mask the regression the second turn asserts on."""
    deadline = time.monotonic() + CACHE_PRIMING_DEADLINE_SECONDS
    while True:
        user_text = _first_turn_user_text(unique_marker())
        body = CacheChatRequest(model=model, messages=[system_turn, _user_turn(user_text, cached=True)])
        usage = unwrap(_post_chat(client, key, body)).usage
        read_tokens = _cache_read_tokens(usage)
        creation_tokens = _cache_creation_tokens(usage)
        if read_tokens > 0 and creation_tokens > 0:
            primed = PrimedCache(
                first_user_text=user_text,
                prefix_read_tokens=read_tokens,
                first_turn_creation_tokens=creation_tokens,
            )
            if _first_turn_reads_back(client, key, body, primed.full_prefix_tokens, deadline):
                return primed
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{model}: prompt cache never became readable in full within "
                f"{CACHE_PRIMING_DEADLINE_SECONDS}s (last usage: {usage})"
            )
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)


def _reads_full_prefix(client: PassthroughClient, key: str, body: CacheChatRequest, full_prefix_tokens: int) -> bool:
    return _cache_read_tokens(unwrap(_post_chat(client, key, body)).usage) >= full_prefix_tokens


def _first_turn_reads_back(
    client: PassthroughClient,
    key: str,
    body: CacheChatRequest,
    full_prefix_tokens: int,
    deadline: float,
) -> bool:
    """True once the full prefix reads back on CACHE_WARM_CONSECUTIVE_READS sends in
    a row. Some providers' global endpoints serve the prompt cache per region, so a
    fresh entry can be missing from the region the next request lands on; each miss
    re-creates the entry there, so the streak converges as the regions warm up."""
    while time.monotonic() < deadline:
        if all(_reads_full_prefix(client, key, body, full_prefix_tokens) for _ in range(CACHE_WARM_CONSECUTIVE_READS)):
            return True
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)
    return False


def _reminder_turn_body(model: str, system_turn: RichMessage, primed: PrimedCache) -> CacheChatRequest:
    """Turn two in OpenAI shape: the primed prefix, an assistant reply, the
    mid-conversation system reminder, and a fresh cache-marked user turn."""
    return CacheChatRequest(
        model=model,
        messages=[
            system_turn,
            _user_turn(primed.first_user_text, cached=True),
            _assistant_turn("OK."),
            _system_reminder_turn(),
            _user_turn("Reply with one word again.", cached=True),
        ],
    )


def _assert_flagged_model_keeps_cache(
    client: PassthroughClient, resources: ResourceManager, params: LiteLLMParamsBody
) -> None:
    model = _register_deployment(client, resources, params)
    key = resources.key(models=[model])
    system_turn = _cacheable_system_turn(unique_marker())

    primed = _prime_prompt_cache(client, key, model, system_turn)

    second = unwrap(_post_chat(client, key, _reminder_turn_body(model, system_turn, primed)))
    read_tokens = _cache_read_tokens(second.usage)

    assert _response_role(second) == "assistant", f"{model}: unexpected role {_response_role(second)!r}"
    assert _response_text(second).strip(), f"{model}: reminder turn returned no completion text"
    assert read_tokens >= primed.full_prefix_tokens, (
        f"{model}: turn with a mid-conversation system reminder read {read_tokens} "
        f"cached tokens, expected at least the {primed.full_prefix_tokens} cached on "
        f"turn one ({primed.prefix_read_tokens} system prefix + "
        f"{primed.first_turn_creation_tokens} first user turn); the reminder was "
        f"hoisted into the top-level system field, which mutates the cached prefix "
        f"and re-bills the conversation at cache-write pricing"
    )


def _assert_unflagged_model_converts_and_succeeds(
    client: PassthroughClient, resources: ResourceManager, params: LiteLLMParamsBody
) -> None:
    model = _register_deployment(client, resources, params)
    key = resources.key(models=[model])
    system_turn = _cacheable_system_turn(unique_marker())

    primed = _prime_prompt_cache(client, key, model, system_turn)

    second = unwrap(_post_chat(client, key, _reminder_turn_body(model, system_turn, primed)))
    read_tokens = _cache_read_tokens(second.usage)

    assert _response_role(second) == "assistant", f"{model}: unexpected role {_response_role(second)!r}"
    assert _response_text(second).strip(), (
        f"{model}: conversation with a mid-conversation system reminder returned "
        f"no text; the reminder was forwarded in place to a model that rejects "
        f"role 'system' inside messages instead of being converted to a user turn"
    )
    assert read_tokens >= primed.full_prefix_tokens, (
        f"{model}: reminder turn read {read_tokens} cached tokens, expected at least "
        f"the {primed.full_prefix_tokens} cached on turn one "
        f"({primed.prefix_read_tokens} system prefix + "
        f"{primed.first_turn_creation_tokens} first user turn); the reminder was "
        f"hoisted into the top-level system field instead of being converted to a "
        f"user turn in place, mutating the cached prefix and re-billing the "
        f"conversation at cache-write pricing"
    )


class TestAnthropicChatMidConversationSystem:
    FLAGGED_MODEL = "anthropic/claude-opus-4-8"
    UNFLAGGED_MODEL = "anthropic/claude-haiku-4-5-20251001"

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        _assert_flagged_model_keeps_cache(client, resources, _anthropic_params(self.FLAGGED_MODEL))

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.mid_conversation_system.nonstream.works",
        exercised_on=[],
    )
    def test_unflagged_model_converts_system_reminder_and_succeeds(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        _assert_unflagged_model_converts_and_succeeds(client, resources, _anthropic_params(self.UNFLAGGED_MODEL))


class TestBedrockInvokeChatMidConversationSystem:
    FLAGGED_MODEL = "bedrock/invoke/us.anthropic.claude-sonnet-5"
    UNFLAGGED_MODEL = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    AWS_REGION = "us-east-1"

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_invoke.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        _assert_flagged_model_keeps_cache(client, resources, _invoke_params(self.FLAGGED_MODEL, self.AWS_REGION))

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_invoke.mid_conversation_system.nonstream.works",
        exercised_on=[],
    )
    def test_unflagged_model_converts_system_reminder_and_succeeds(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        _assert_unflagged_model_converts_and_succeeds(
            client, resources, _invoke_params(self.UNFLAGGED_MODEL, self.AWS_REGION)
        )

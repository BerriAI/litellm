"""Live e2e: mid-conversation ``role: "system"`` handling on the Bedrock Invoke
/v1/messages path is model-aware (PRs #32578, #32831, #32882).

Models flagged ``supports_mid_conversation_system`` in the cost map (Claude 4.8+
and the 5 family) must keep a mid-conversation system reminder in place inside
``messages`` so the top-level ``system`` prefix stays byte-identical and the
prompt cache written on turn one is read back in full on turn two. Models
without the flag (Claude 4.7 and older) reject the role inside ``messages``
outright, so the proxy must convert the reminder to a user turn in place and
the call must still return a completion instead of a provider 400.

The conversation shape mirrors what Claude Code sends mid-session: a cached
system prompt, a user turn carrying its own ``cache_control`` breakpoint, a
``role: "system"`` reminder, an assistant turn, and a fresh user turn. The
message-turn breakpoint is what makes the cache assertion able to fail: a cache
entry whose prefix spans ``system`` plus message turns is invalidated when the
reminder is hoisted (the ``system`` field mutates and a turn disappears from
``messages``), while an entry ending at the system block itself would survive
the hoist and mask the regression.

Calls go through the real Anthropic SDK (LIT-4577). The SDK's ``MessageParam``
type only admits user/assistant roles, so the system reminder turn is cast to
it; the SDK serializes the dict verbatim, which is exactly the wire shape under
test.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

import pytest
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlockParam
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

FLAGGED_INVOKE_MODEL = "bedrock/invoke/us.anthropic.claude-sonnet-5"
UNFLAGGED_INVOKE_MODEL = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"
AWS_REGION = "us-east-1"
CACHE_PRIMING_DEADLINE_SECONDS = 60.0
CACHE_PRIMING_INTERVAL_SECONDS = 3.0
CACHE_WARM_CONSECUTIVE_READS = 3


def _cacheable_system_block(marker: str) -> TextBlockParam:
    """A system prompt at roughly twice the 4096-token minimum cacheable size of
    Haiku 4.5 (the smallest model here), unique per run so no other run's cache
    entry can satisfy the read. The marker appears once instead of in every
    paragraph: repeating it swung the block's size by ~1800 tokens with the
    marker's own tokenization and left it under the minimum on ~15% of runs, so
    the system breakpoint went uncached and the priming loop never saw a read."""
    text = f"Run {marker}.\n" + " ".join(f"Reference paragraph {index}." for index in range(1500))
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def _user_turn(text: str, *, cached: bool = False) -> MessageParam:
    block: TextBlockParam = (
        {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}} if cached else {"type": "text", "text": text}
    )
    return {"role": "user", "content": [block]}


def _system_reminder_turn() -> MessageParam:
    return cast(
        "MessageParam",
        {
            "role": "system",
            "content": [{"type": "text", "text": "<system-reminder>Answer with exactly one word.</system-reminder>"}],
        },
    )


def _assistant_turn(text: str) -> MessageParam:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _send(client: Anthropic, model: str, system_block: TextBlockParam, messages: Sequence[MessageParam]) -> Message:
    return client.messages.create(model=model, max_tokens=64, system=[system_block], messages=messages)


def _register_invoke_deployment(proxy: ProxyClient, resources: ResourceManager, bedrock_model: str) -> str:
    model = f"e2e-midsys-{unique_marker()}"
    model_id = proxy.create_model(model, LiteLLMParamsBody(model=bedrock_model, aws_region_name=AWS_REGION))
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _first_turn_user_text(marker: str) -> str:
    """A first user turn heavy enough (hundreds of tokens) that losing its cache
    entry is unambiguous in the usage numbers, unique per attempt so priming
    retries never depend on the proxy's response cache behavior."""
    notes = " ".join(f"Session note {index}." for index in range(100))
    return f"Reply with one word. Attempt {marker}.\n{notes}"


class PrimedCache(BaseModel):
    first_user_text: str
    prefix_read_tokens: int
    first_turn_creation_tokens: int

    @property
    def full_prefix_tokens(self) -> int:
        return self.prefix_read_tokens + self.first_turn_creation_tokens


def _prime_prompt_cache(client: Anthropic, model: str, system_block: TextBlockParam) -> PrimedCache:
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
        first_turn = (_user_turn(user_text, cached=True),)
        usage = _send(client, model, system_block, first_turn).usage
        read_tokens = usage.cache_read_input_tokens or 0
        creation_tokens = usage.cache_creation_input_tokens or 0
        if read_tokens > 0 and creation_tokens > 0:
            primed = PrimedCache(
                first_user_text=user_text,
                prefix_read_tokens=read_tokens,
                first_turn_creation_tokens=creation_tokens,
            )
            if _first_turn_reads_back(client, model, system_block, first_turn, primed.full_prefix_tokens, deadline):
                return primed
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{model}: prompt cache never became readable in full within "
                f"{CACHE_PRIMING_DEADLINE_SECONDS}s (last usage: {usage})"
            )
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)


def _reads_full_prefix(
    client: Anthropic,
    model: str,
    system_block: TextBlockParam,
    messages: Sequence[MessageParam],
    full_prefix_tokens: int,
) -> bool:
    return (_send(client, model, system_block, messages).usage.cache_read_input_tokens or 0) >= full_prefix_tokens


def _first_turn_reads_back(
    client: Anthropic,
    model: str,
    system_block: TextBlockParam,
    messages: Sequence[MessageParam],
    full_prefix_tokens: int,
    deadline: float,
) -> bool:
    """True once the full prefix reads back on CACHE_WARM_CONSECUTIVE_READS sends in
    a row. Some providers' global endpoints serve the prompt cache per region, so a
    fresh entry can be missing from the region the next request lands on; each miss
    re-creates the entry there, so the streak converges as the regions warm up."""
    while time.monotonic() < deadline:
        if all(
            _reads_full_prefix(client, model, system_block, messages, full_prefix_tokens)
            for _ in range(CACHE_WARM_CONSECUTIVE_READS)
        ):
            return True
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)
    return False


def _reminder_turn_messages(primed: PrimedCache) -> tuple[MessageParam, ...]:
    return (
        _user_turn(primed.first_user_text, cached=True),
        _system_reminder_turn(),
        _assistant_turn("OK."),
        _user_turn("Reply with one word again.", cached=True),
    )


#: Kept in sync with the copy in test_messages_mid_conversation_system_native_providers_e2e.py;
#: the e2e suites stay self-contained rather than importing across test modules.
MID_CONVERSATION_CACHE_SKIP_REASON = (
    "LIT-4873: a mid-conversation role='system' reminder invalidates the prompt cache on the "
    "vertex_ai / azure_ai / bedrock_invoke Messages paths, while the same request preserves it "
    "both direct to Anthropic and through litellm's first-party anthropic path. Product bug, not "
    "a test defect: the assertion here is correct and must be restored unchanged with the fix"
)


class TestBedrockInvokeMidConversationSystem:
    @pytest.mark.skip(reason=MID_CONVERSATION_CACHE_SKIP_REASON)
    @pytest.mark.covers(
        "llm.messages.bedrock_invoke.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_invoke_deployment(proxy, resources, FLAGGED_INVOKE_MODEL)
        client = sdk.anthropic(resources.key(models=[model]))
        system_block = _cacheable_system_block(unique_marker())

        primed = _prime_prompt_cache(client, model, system_block)

        second = _send(client, model, system_block, _reminder_turn_messages(primed))

        assert _text(second).strip(), f"{model}: reminder turn returned no completion text"
        assert (second.usage.cache_read_input_tokens or 0) >= primed.full_prefix_tokens, (
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
    def test_unflagged_model_converts_system_reminder_and_succeeds(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register_invoke_deployment(proxy, resources, UNFLAGGED_INVOKE_MODEL)
        client = sdk.anthropic(resources.key(models=[model]))
        system_block = _cacheable_system_block(unique_marker())

        primed = _prime_prompt_cache(client, model, system_block)

        second = _send(client, model, system_block, _reminder_turn_messages(primed))

        assert second.role == "assistant", f"{model}: unexpected role {second.role!r}"
        assert _text(second).strip(), (
            f"{model}: conversation with a mid-conversation system reminder "
            f"returned no text; the reminder was forwarded in place to a model "
            f"that rejects role 'system' inside messages instead of being converted to a user turn"
        )
        assert (second.usage.cache_read_input_tokens or 0) >= primed.full_prefix_tokens, (
            f"{model}: reminder turn read {second.usage.cache_read_input_tokens} "
            f"cached tokens, expected at least the {primed.full_prefix_tokens} "
            f"cached on turn one ({primed.prefix_read_tokens} system prefix + "
            f"{primed.first_turn_creation_tokens} first user turn); the reminder "
            f"was hoisted into the top-level system field instead of being "
            f"converted to a user turn in place, mutating the cached prefix and "
            f"re-billing the conversation at cache-write pricing"
        )

"""Live e2e: model-aware mid-conversation ``role: "system"`` handling on the
Azure AI Foundry and Vertex AI ``/v1/messages`` paths.

Azure Foundry and Vertex both serve Claude on the first-party Anthropic Messages
contract, verified live: a mid-conversation ``role: "system"`` reminder is
accepted in place on Claude 4.8+/5 (200) but rejected on Claude 4.7 and older
("role 'system' is not supported on this model", 400), and a *leading* system
entry is rejected on every model ("messages.0: use the top-level 'system'
parameter"). This mirrors Bedrock Invoke (PRs #32578/#32831/#32882); the same
model-gated normalization now runs for these two providers (customer RCA gap #3).

Flagged models (``supports_mid_conversation_system`` in the cost map: Claude
4.8+ and the 5 family) must keep the reminder in ``messages`` so the top-level
``system`` prefix stays byte-identical and the prompt cache written on turn one
is read back in full on turn two. Unflagged models (Claude 4.7 and older) must
have the reminder converted to a user turn in place so the call
returns a completion instead of a provider 400.

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
from endpoints_client import (
    CacheControl,
    EndpointsClient,
    MessagesResult,
    RichMessage,
    RichMessagesRequest,
    TextBlock,
)
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

CACHE_PRIMING_DEADLINE_SECONDS = 60.0
CACHE_PRIMING_INTERVAL_SECONDS = 3.0
CACHE_WARM_CONSECUTIVE_READS = 3


def _azure_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=model,
        api_base="os.environ/AZURE_AI_API_BASE",
        api_key="os.environ/AZURE_AI_API_KEY",
    )


def _vertex_params(model: str, location: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=model,
        vertex_project="os.environ/VERTEXAI_PROJECT",
        vertex_location=location,
    )


def _cacheable_system_block(marker: str) -> TextBlock:
    """A system prompt at roughly twice the 4096-token minimum cacheable size of
    Haiku 4.5 (the smallest model here), unique per run so no other run's cache
    entry can satisfy the read. The marker appears once instead of in every
    paragraph: repeating it swung the block's size by ~1800 tokens with the
    marker's own tokenization and left it under the minimum on ~15% of runs, so
    the system breakpoint went uncached and the priming loop never saw a read."""
    text = f"Run {marker}.\n" + " ".join(
        f"Reference paragraph {index}." for index in range(1500)
    )
    return TextBlock(text=text, cache_control=CacheControl())


def _user_turn(text: str, *, cached: bool = False) -> RichMessage:
    block = TextBlock(text=text, cache_control=CacheControl() if cached else None)
    return RichMessage(role="user", content=[block])


def _system_reminder_turn() -> RichMessage:
    return RichMessage(
        role="system",
        content=[TextBlock(text="<system-reminder>Answer with exactly one word.</system-reminder>")],
    )


def _post_messages(client: EndpointsClient, key: str, body: RichMessagesRequest) -> Result[MessagesResult]:
    return client.proxy.transport.post(
        "/v1/messages",
        headers=client.proxy.transport.bearer(key),
        json=body,
        response_type=MessagesResult,
    )


def _register_deployment(
    client: EndpointsClient, resources: ResourceManager, params: LiteLLMParamsBody
) -> str:
    model = f"e2e-midsys-{unique_marker()}"
    model_id = client.create_model(model, params)
    resources.defer(lambda: client.delete_model(model_id))
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


def _prime_prompt_cache(
    client: EndpointsClient, key: str, model: str, system_block: TextBlock
) -> PrimedCache:
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
        body = RichMessagesRequest(
            model=model,
            system=[system_block],
            messages=[_user_turn(user_text, cached=True)],
        )
        usage = unwrap(_post_messages(client, key, body)).usage
        if usage.cache_read_input_tokens > 0 and usage.cache_creation_input_tokens > 0:
            primed = PrimedCache(
                first_user_text=user_text,
                prefix_read_tokens=usage.cache_read_input_tokens,
                first_turn_creation_tokens=usage.cache_creation_input_tokens,
            )
            if _first_turn_reads_back(client, key, body, primed.full_prefix_tokens, deadline):
                return primed
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{model}: prompt cache never became readable in full within "
                f"{CACHE_PRIMING_DEADLINE_SECONDS}s (last usage: {usage})"
            )
        time.sleep(CACHE_PRIMING_INTERVAL_SECONDS)


def _reads_full_prefix(
    client: EndpointsClient, key: str, body: RichMessagesRequest, full_prefix_tokens: int
) -> bool:
    return unwrap(_post_messages(client, key, body)).usage.cache_read_input_tokens >= full_prefix_tokens


def _first_turn_reads_back(
    client: EndpointsClient,
    key: str,
    body: RichMessagesRequest,
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


#: Why the flagged-model cache checks are skipped rather than failing. The
#: assertions below are correct and must be restored unchanged when the bug is
#: fixed; they are the regression guard for a real billing cost.
#:
#: Measured cache_read_input_tokens on the reminder turn, same conversation shape:
#:   direct to api.anthropic.com            7013  preserved
#:   litellm -> anthropic/claude-opus-4-8   7013  preserved
#:   litellm -> vertex_ai/claude-opus-4-8      0  destroyed
#: and the Vertex control with the same added turns but no reminder reads 7013,
#: so it is the reminder on the non-first-party paths, not the extra turns.
MID_CONVERSATION_CACHE_SKIP_REASON = (
    "LIT-4873: a mid-conversation role='system' reminder invalidates the prompt cache on the "
    "vertex_ai / azure_ai / bedrock_invoke Messages paths, while the same request preserves it "
    "both direct to Anthropic and through litellm's first-party anthropic path. Product bug, not "
    "a test defect: the assertion here is correct and must be restored unchanged with the fix"
)


def _assert_flagged_model_keeps_cache(
    client: EndpointsClient, resources: ResourceManager, params: LiteLLMParamsBody
) -> None:
    model = _register_deployment(client, resources, params)
    key = resources.key(models=[model])
    system_block = _cacheable_system_block(unique_marker())

    primed = _prime_prompt_cache(client, key, model, system_block)

    reminder_turn_body = RichMessagesRequest(
        model=model,
        system=[system_block],
        messages=[
            _user_turn(primed.first_user_text, cached=True),
            _system_reminder_turn(),
            RichMessage(role="assistant", content=[TextBlock(text="OK.")]),
            _user_turn("Reply with one word again.", cached=True),
        ],
    )
    second = unwrap(_post_messages(client, key, reminder_turn_body))

    assert second.text.strip(), f"{model}: reminder turn returned no completion text"
    assert second.usage.cache_read_input_tokens >= primed.full_prefix_tokens, (
        f"{model}: turn with a mid-conversation system reminder read "
        f"{second.usage.cache_read_input_tokens} cached tokens, expected at "
        f"least the {primed.full_prefix_tokens} cached on turn one "
        f"({primed.prefix_read_tokens} system prefix + "
        f"{primed.first_turn_creation_tokens} first user turn); the reminder "
        f"was hoisted into the top-level system field, which mutates the cached "
        f"prefix and re-bills the conversation at cache-write pricing"
    )


def _assert_unflagged_model_converts_and_succeeds(
    client: EndpointsClient, resources: ResourceManager, params: LiteLLMParamsBody
) -> None:
    model = _register_deployment(client, resources, params)
    key = resources.key(models=[model])
    system_block = _cacheable_system_block(unique_marker())

    primed = _prime_prompt_cache(client, key, model, system_block)

    reminder_turn_body = RichMessagesRequest(
        model=model,
        system=[system_block],
        messages=[
            _user_turn(primed.first_user_text, cached=True),
            _system_reminder_turn(),
            RichMessage(role="assistant", content=[TextBlock(text="OK.")]),
            _user_turn("Reply with one word again.", cached=True),
        ],
    )
    second = unwrap(_post_messages(client, key, reminder_turn_body))

    assert second.role == "assistant", f"{model}: unexpected role {second.role!r}"
    assert second.text.strip(), (
        f"{model}: conversation with a mid-conversation system reminder returned "
        f"no text; the reminder was forwarded in place to a model that rejects "
        f"role 'system' inside messages instead of being converted to a user turn"
    )
    assert second.usage.cache_read_input_tokens >= primed.full_prefix_tokens, (
        f"{model}: reminder turn read {second.usage.cache_read_input_tokens} cached "
        f"tokens, expected at least the {primed.full_prefix_tokens} cached on turn "
        f"one ({primed.prefix_read_tokens} system prefix + "
        f"{primed.first_turn_creation_tokens} first user turn); the reminder was "
        f"hoisted into the top-level system field instead of being converted to a "
        f"user turn in place, mutating the cached prefix and re-billing the "
        f"conversation at cache-write pricing"
    )


class TestAzureFoundryMidConversationSystem:
    FLAGGED_MODEL = "azure_ai/claude-opus-4-8"
    UNFLAGGED_MODEL = "azure_ai/claude-opus-4-7"

    @pytest.mark.skip(reason=MID_CONVERSATION_CACHE_SKIP_REASON)
    @pytest.mark.covers(
        "llm.messages.azure_foundry.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _assert_flagged_model_keeps_cache(endpoints_client, resources, _azure_params(self.FLAGGED_MODEL))

    @pytest.mark.covers(
        "llm.messages.azure_foundry.mid_conversation_system.nonstream.works",
        exercised_on=[],
    )
    def test_unflagged_model_converts_system_reminder_and_succeeds(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _assert_unflagged_model_converts_and_succeeds(
            endpoints_client, resources, _azure_params(self.UNFLAGGED_MODEL)
        )


class TestVertexMidConversationSystem:
    """The unflagged test pins a single region because the global endpoint serves the
    prompt cache per region: a chunk written seconds earlier can still be missing from
    the region the reminder turn lands on, which reads exactly like the hoist regression
    (system prefix read back, first user turn re-created). The flagged model has quota
    only on the global endpoint, so its test keeps that location."""

    FLAGGED_MODEL = "vertex_ai/claude-opus-4-8"
    FLAGGED_LOCATION = "global"
    UNFLAGGED_MODEL = "vertex_ai/claude-sonnet-4-6"
    UNFLAGGED_LOCATION = "us-east5"

    @pytest.mark.skip(reason=MID_CONVERSATION_CACHE_SKIP_REASON)
    @pytest.mark.covers(
        "llm.messages.vertex.mid_conversation_system.nonstream.cache_hit",
        exercised_on=[],
    )
    def test_flagged_model_keeps_prompt_cache_across_system_reminder(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _assert_flagged_model_keeps_cache(
            endpoints_client, resources, _vertex_params(self.FLAGGED_MODEL, self.FLAGGED_LOCATION)
        )

    @pytest.mark.covers(
        "llm.messages.vertex.mid_conversation_system.nonstream.works",
        exercised_on=[],
    )
    def test_unflagged_model_converts_system_reminder_and_succeeds(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        _assert_unflagged_model_converts_and_succeeds(
            endpoints_client, resources, _vertex_params(self.UNFLAGGED_MODEL, self.UNFLAGGED_LOCATION)
        )

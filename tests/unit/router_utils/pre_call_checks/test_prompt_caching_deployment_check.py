import asyncio
import copy
import os
import functools
import json
import logging
import threading
import uuid
from typing import Final, cast

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    PROMPT_CACHE_LOOKBACK_POSITIONS,
    REDACTED_BY_LITELLM,
)
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
    _honors_one_hour_cache_ttl,
    _get_min_token_count_for_deployments,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.prompt_caching_cache import PROMPT_CACHE_PIN_TTL_SECONDS, PromptCachingCache
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt, token_counter

MODEL_GROUP_ALIAS = "my-claude-group"
OPUS_4_6_MIN_TOKENS = 4096
CALLBACK_REGISTRIES: Final = (
    "input_callback",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "callbacks",
)


@pytest.fixture(autouse=True)
def _fresh_callback_registries(monkeypatch):
    """`litellm.logging_callback_manager` keeps one callback per class, so a
    `PromptCachingDeploymentCheck` or `_SentMessagesCapture` left behind by an
    earlier test would swallow the next test's success events."""
    for registry in CALLBACK_REGISTRIES:
        monkeypatch.setattr(litellm, registry, [])


@pytest.fixture
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.fixture(autouse=True)
def _local_model_cost_map_autouse(local_model_cost_map):
    """Every test here reads `prompt_cache_min_tokens`, which only the in-repo map
    carries, so the shared local_model_cost_map fixture (conftest.py) is autouse
    for the whole file."""
    yield


def _deployments(*models: str) -> list[dict]:
    return [
        {
            "model_name": MODEL_GROUP_ALIAS,
            "litellm_params": {"model": model},
            "model_info": {"id": f"dep-{index}"},
        }
        for index, model in enumerate(models, start=1)
    ]


def _messages(word_count: int) -> list[AllMessageValues]:
    return cast(
        list[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "word " * word_count,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )


def test_get_min_token_count_for_deployments_takes_min_across_mixed_group():
    """
    A group may legally mix models whose real minimums differ, and one gate decides for every
    member. The threshold must be the lowest minimum in the group. This gate only decides whether
    the cache lookup happens, so taking the highest would skip the lookup for a prefix the Sonnet
    4.5 deployment genuinely cached and lose a hit it had earned.
    """
    assert get_prompt_cache_min_tokens(model="anthropic/claude-opus-4-5") == 4096
    assert get_prompt_cache_min_tokens(model="anthropic/claude-sonnet-4-5") == 1024

    deployments = _deployments("anthropic/claude-opus-4-5", "anthropic/claude-sonnet-4-5")

    assert _get_min_token_count_for_deployments(deployments) == 1024


def test_write_gate_is_what_prevents_a_pin_below_the_model_minimum():
    """
    The invariant the read gate relies on. A deployment can only be pinned when the cache already
    holds an entry for the prefix, and `async_log_success_event` writes entries against the real
    deployment model. Opus 4.5 never records an entry for a prefix it will not cache, so no read
    threshold is what keeps it from being pinned.
    """
    messages = _messages(word_count=1400)

    token_count = token_counter(
        messages=messages, model="anthropic/claude-opus-4-5", use_default_image_token_count=True
    )
    assert 1024 < token_count < 4096

    assert is_prompt_caching_valid_prompt(model="anthropic/claude-opus-4-5", messages=messages) is False
    assert is_prompt_caching_valid_prompt(model="anthropic/claude-sonnet-4-5", messages=messages) is True


def test_get_min_token_count_for_deployments_falls_back_to_default_for_empty_group():
    """An empty group has no member minimum to read, so it must fall back rather than crash."""
    assert _get_min_token_count_for_deployments([]) == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT


def test_bedrock_application_profile_uses_its_base_model_for_one_hour_ttl() -> None:
    standard_logging_object: Final = cast(
        StandardLoggingPayload,
        {
            "custom_llm_provider": "bedrock",
            "model": "bedrock/converse/arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/pinprofile",
        },
    )

    assert (
        _honors_one_hour_cache_ttl(
            standard_logging_object,
            {"model": None, "base_model": "bedrock/us.anthropic.claude-sonnet-5-5"},
        )
        is True
    )


def test_bedrock_application_profile_cache_hit_reads_base_model_from_metadata_model_info() -> None:
    standard_logging_object: Final = cast(
        StandardLoggingPayload,
        {
            "custom_llm_provider": "bedrock",
            "model": "bedrock/converse/arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/pinprofile",
        },
    )
    assert (
        _honors_one_hour_cache_ttl(
            standard_logging_object,
            {"metadata": {"model_info": {"base_model": "bedrock/us.anthropic.claude-sonnet-5-5"}}},
        )
        is True
    )


def test_bedrock_application_profile_top_level_base_model_wins_over_metadata_model_info() -> None:
    standard_logging_object: Final = cast(
        StandardLoggingPayload,
        {
            "custom_llm_provider": "bedrock",
            "model": "bedrock/converse/arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/pinprofile",
        },
    )
    top_level: Final = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
    in_metadata: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
    top_level_alone: Final = _honors_one_hour_cache_ttl(standard_logging_object, {"base_model": top_level})
    metadata_alone: Final = _honors_one_hour_cache_ttl(
        standard_logging_object, {"metadata": {"model_info": {"base_model": in_metadata}}}
    )
    assert top_level_alone != metadata_alone

    assert (
        _honors_one_hour_cache_ttl(
            standard_logging_object,
            {"base_model": top_level, "metadata": {"model_info": {"base_model": in_metadata}}},
        )
        is top_level_alone
    )


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_narrow_prompt_below_model_minimum():
    """
    The regression. Opus 4.6 will not cache a prefix under 4096 tokens, so a ~1400-token prompt is
    not cacheable and routing must stay free across the whole group. Previously the check resolved
    its threshold from `model`, which is the operator's group alias and matches nothing in the cost
    map, silently fell back to 1024, judged this prompt cacheable, and pinned every request to one
    deployment for a cache hit the provider was never going to serve.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    messages = _messages(word_count=1400)

    token_count = token_counter(
        messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True
    )
    assert DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT < token_count < OPUS_4_6_MIN_TOKENS

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == deployments


@pytest.mark.asyncio
async def test_async_filter_deployments_narrows_prompt_above_model_minimum():
    """
    The positive control for the regression above: once the same group's prompt clears Opus 4.6's
    real 4096-token minimum the prefix is genuinely cacheable, so the check must still pin the
    deployment that served it. Proves the fix tightened the gate rather than disabling the feature.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    messages = _messages(word_count=5000)

    token_count = token_counter(
        messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True
    )
    assert token_count > OPUS_4_6_MIN_TOKENS

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


class _ReadTrackingMessages(list[AllMessageValues]):
    def __init__(self, messages: list[AllMessageValues]) -> None:
        super().__init__(messages)
        self.read = False

    def __iter__(self):
        self.read = True
        return super().__iter__()


class _PinLookupCounter(DualCache):
    def __init__(self) -> None:
        super().__init__()
        self.pin_lookups = 0

    async def async_batch_get_cache(
        self,
        keys: list[str],
        parent_otel_span: object = None,
        local_only: bool = False,
        throttle_redis: bool = True,
        **kwargs: object,
    ):
        self.pin_lookups += 1
        return await super().async_batch_get_cache(
            keys,
            parent_otel_span=parent_otel_span,
            local_only=local_only,
            throttle_redis=throttle_redis,
            **kwargs,
        )


@pytest.mark.asyncio
async def test_async_filter_deployments_skips_prefix_hash_for_a_single_deployment():
    """
    With one healthy deployment there is nothing to pin to, so the check must hand the group
    back without hashing the prefix or probing the pin cache: on a 400k-token Claude Code
    prompt that hash alone is ~30 ms of GIL-holding work per request.
    """
    cache = _PinLookupCounter()
    check = PromptCachingDeploymentCheck(cache=cache, group_has_alternatives=lambda group: True)
    deployments = _deployments("anthropic/claude-opus-4-6")
    messages = _messages(word_count=5000)
    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-1", messages=messages, tools=None)
    tracked_messages = _ReadTrackingMessages(messages)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=tracked_messages
    )

    assert filtered == deployments
    assert cache.pin_lookups == 0
    assert tracked_messages.read is False

    two = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    assert await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=two, messages=messages
    ) == [two[0]]
    assert cache.pin_lookups == 1


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_pin_when_target_order_is_set():
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")
    messages = _messages(word_count=5000)

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
        request_kwargs={"_target_order": 2},
    )

    assert filtered == deployments


_RETRIES_ON_DEP_1: Final = tuple({"deployment_id": "dep-1"} for _ in range(4))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_in_request",
    [
        {"_excluded_deployment_ids": ["dep-2"]},
        {"_retry_skipped_deployment_ids": ("dep-2",)},
        {"metadata": {"previous_models": [{"deployment_id": "dep-2"}], "request_retry_count": 1}},
        {"litellm_metadata": {"previous_models": ({"deployment_id": "dep-2"},), "request_retry_count": 1}},
        {"metadata": {"previous_models": _RETRIES_ON_DEP_1, "request_retry_count": 5}},
    ],
    ids=["excluded", "retry_skipped", "retried", "retried_litellm_metadata", "retries_rolled_over"],
)
async def test_async_filter_deployments_does_not_pin_to_a_deployment_that_failed_this_request(
    failed_in_request: dict[str, object],
):
    cache: Final = DualCache()
    check: Final = PromptCachingDeploymentCheck(cache=cache)
    messages: Final = _messages(word_count=5000)
    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    assert await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=_deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6"),
        messages=messages,
        request_kwargs=failed_in_request,
    ) == _deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_metadata",
    [
        {"previous_models": _RETRIES_ON_DEP_1, "request_retry_count": 4},
        {"previous_models": "dep-2"},
        {"previous_models": [None, "dep-2", {"deployment_id": None}], "request_retry_count": 3},
    ],
    ids=["every_retry_on_record", "malformed_breadcrumbs", "malformed_records"],
)
async def test_async_filter_deployments_keeps_the_pin_while_every_retry_is_on_record(
    retry_metadata: dict[str, object],
):
    cache: Final = DualCache()
    check: Final = PromptCachingDeploymentCheck(cache=cache)
    messages: Final = _messages(word_count=5000)
    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered: Final = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=_deployments("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-6"),
        messages=messages,
        request_kwargs={"metadata": retry_metadata},
    )

    assert [deployment["model_info"]["id"] for deployment in filtered] == ["dep-2"]


@pytest.mark.asyncio
async def test_async_filter_deployments_narrows_for_group_whose_model_minimum_is_lower():
    """
    Same ~1400-token prompt that must not pin an Opus 4.6 group, on an Opus 4.8 group whose real
    minimum is 1024. Here the prefix is cacheable and the check must pin. Proves the threshold is
    resolved per-model from the deployments rather than tightened for everyone.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments("anthropic/claude-opus-4-8", "anthropic/claude-opus-4-8")
    messages = _messages(word_count=1400)

    assert get_prompt_cache_min_tokens(model="anthropic/claude-opus-4-8") == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


AUTO_CACHING_MODEL = "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_replayed_redacted_thinking_block_still_records_and_pins():
    """
    A model that returns no reasoning summary (gpt-5.x through the /v1/messages bridge, Anthropic with
    redacted reasoning) hands the client a `redacted_thinking` block, and the client replays it on every
    later turn. The token count behind `is_prompt_caching_valid_prompt` raised on that block, the helper
    swallowed it to False, and the check neither recorded the serving deployment nor pinned it, so the
    conversation bounced across the group and paid a cache write on each deployment.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    model = "openai/gpt-5.6-sol"
    deployments = _deployments(model, model, model)
    messages = cast(
        list[AllMessageValues],
        [
            *_messages(word_count=3000),
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "litellm_encrypted_reasoning:" + "Z" * 400},
                    {"type": "text", "text": "Draw from the box labeled Mixed."},
                ],
            },
            {"role": "user", "content": "Restate that in one sentence."},
        ],
    )

    assert is_prompt_caching_valid_prompt(model=model, messages=messages) is True

    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": {
                "call_type": "anthropic_messages",
                "custom_llm_provider": "anthropic",
                "model": model,
                "messages": messages,
                "model_id": "dep-2",
                "model_map_information": {"model_map_key": model, "model_map_value": None},
            }
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


def _auto_caching_messages() -> list[AllMessageValues]:
    """A prompt over the model minimum that carries no client cache_control."""
    return cast(
        list[AllMessageValues],
        [
            {"role": "system", "content": "word " * 3000},
            {"role": "user", "content": "hello"},
        ],
    )


def _affinity_messages(messages: list[AllMessageValues]) -> list[AllMessageValues]:
    """The messages the check keys deployment affinity on, for a group of `AUTO_CACHING_MODEL`."""
    return AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages,
        models=(AUTO_CACHING_MODEL,),
    )


class _SentMessagesCapture(CustomLogger):
    def __init__(self, litellm_call_id: str):
        self.litellm_call_id = litellm_call_id
        self.messages: list[AllMessageValues] | None = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("litellm_call_id") != self.litellm_call_id:
            return
        standard_logging_object = kwargs.get("standard_logging_object")
        if standard_logging_object is not None:
            self.messages = standard_logging_object["messages"]


async def _eventually(predicate, timeout: float = 10.0):
    """Success callbacks run as tasks, so give the write a bounded window to land."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(0.05)
    return predicate()


@pytest.mark.asyncio
async def test_affinity_key_matches_the_messages_auto_caching_actually_sends(monkeypatch, local_model_cost_map):
    """
    The regression. `enable_anthropic_prompt_caching` injects cache_control inside
    `litellm.acompletion`, which runs after routing, so at filter time the messages carried no
    marker, `extract_cacheable_prefix` returned [], the key was None, and the check no-opped on
    every request. Routing must derive the same key the success event writes from the messages the
    request was actually sent with, otherwise auto-injected caching gets no affinity at all.
    """
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    call_id: Final = str(uuid.uuid4())
    capture = _SentMessagesCapture(call_id)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    messages = _auto_caching_messages()

    await litellm.acompletion(
        model=AUTO_CACHING_MODEL,
        messages=copy.deepcopy(messages),
        mock_response="ok",
        api_key="sk-fake",
        litellm_call_id=call_id,
    )
    sent_messages = await _eventually(lambda: capture.messages)
    assert sent_messages is not None

    routing_key = PromptCachingCache.get_prompt_caching_cache_key(_affinity_messages(messages), None)

    assert routing_key is not None
    assert routing_key == PromptCachingCache.get_prompt_caching_cache_key(sent_messages, None)


@pytest.mark.asyncio
async def test_repeated_auto_cached_prefix_pins_to_one_deployment(monkeypatch, local_model_cost_map):
    """
    End to end over the router: identical requests with no client cache_control must stop bouncing
    across a multi-deployment group once one deployment has cached the prefix. Bedrock and Anthropic
    caches are per account and region, so every bounce paid the cache write premium and never read.
    """
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": AUTO_CACHING_MODEL, "api_key": "sk-fake"},
                "model_info": {"id": model_id},
            }
            for model_id in ("dep-1", "dep-2")
        ],
        optional_pre_call_checks=["prompt_caching"],
    )
    messages = _auto_caching_messages()

    first = await router.acompletion(model=MODEL_GROUP_ALIAS, messages=messages, mock_response="ok")
    served_by = first._hidden_params["model_id"]

    affinity_key = PromptCachingCache.get_prompt_caching_cache_key(_affinity_messages(messages), None)
    assert await _eventually(lambda: router.prompt_caching_pin_cache.get_cache(key=affinity_key)) is not None

    subsequent = [
        (await router.acompletion(model=MODEL_GROUP_ALIAS, messages=messages, mock_response="ok"))._hidden_params[
            "model_id"
        ]
        for _ in range(4)
    ]

    assert subsequent == [served_by] * 4


@pytest.mark.asyncio
async def test_per_request_enable_prompt_caching_reaches_the_affinity_key(monkeypatch, local_model_cost_map):
    """
    `enable_prompt_caching` turns auto-injection on for a single request while the global flag stays
    off, so routing has to read it too. Ignore it and the key comes off unmarked messages, which is
    never what the request goes on to send, and the pin is lost for every per-key enablement.
    """
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    messages = _auto_caching_messages()

    sent = AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages, models=(AUTO_CACHING_MODEL,), enable_prompt_caching=True
    )
    assert sent != messages
    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=sent, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
        request_kwargs={"enable_prompt_caching": True},
    )

    assert filtered == [deployments[1]]


@pytest.mark.asyncio
async def test_claude_code_one_shot_subagent_does_not_reuse_an_auto_injected_affinity_key(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    messages = cast(list[AllMessageValues], [{"role": "user", "content": "unique " * 3000}])
    request_kwargs = {
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.263; cc_is_subagent=true;",
            }
        ],
        "proxy_server_request": {"headers": {"user-agent": "claude-cli/2.1.263 (external, cli)"}},
    }
    auto_injected_messages = AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages,
        models=(AUTO_CACHING_MODEL,),
    )
    assert auto_injected_messages != messages
    await PromptCachingCache(cache=cache).async_add_model_id(
        model_id="dep-2", messages=auto_injected_messages, tools=None
    )

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
        request_kwargs=request_kwargs,
    )

    assert filtered == deployments


@pytest.mark.asyncio
async def test_root_cache_control_does_not_reuse_an_auto_injected_affinity_key(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    messages = _auto_caching_messages()
    auto_injected_messages = AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages,
        models=(AUTO_CACHING_MODEL,),
    )
    assert auto_injected_messages != messages
    await PromptCachingCache(cache=cache).async_add_model_id(
        model_id="dep-2", messages=auto_injected_messages, tools=None
    )

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
        request_kwargs={"cache_control": {"type": "ephemeral"}},
    )

    assert filtered == deployments


@pytest.mark.asyncio
async def test_tool_marked_cache_control_keeps_routing_off_another_requests_prefix(monkeypatch, local_model_cost_map):
    """
    Tools carrying the client's own cache_control make auto-injection stand down, so this request
    will not carry litellm's breakpoints. Routing must see the tools as well. Ignore them and it
    keys off the injected prefix, pinning the request to whichever deployment cached a different,
    tool-less request whose prefix it can never actually reuse.
    """
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    messages = _auto_caching_messages()
    cache_marked_tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    await PromptCachingCache(cache=cache).async_add_model_id(
        model_id="dep-2", messages=_affinity_messages(messages), tools=None
    )

    without_tools = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=messages
    )
    assert without_tools == [deployments[1]]

    with_tools = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
        request_kwargs={"tools": cache_marked_tools},
    )

    assert with_tools == deployments


def test_client_supplied_cache_control_keeps_its_own_prefix_boundary(monkeypatch, local_model_cost_map):
    """
    Auto-injection stands down when the client marks its own breakpoints, so the affinity key must
    keep keying off the client's boundary. Injecting on top would push the boundary to the trailing
    turn and break affinity for prompts that already worked.
    """
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    messages = cast(
        list[AllMessageValues],
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "word " * 3000, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "user", "content": "hello"},
        ],
    )

    for_key = _affinity_messages(messages)

    assert for_key is messages
    assert PromptCachingCache.extract_cacheable_prefix(for_key) == messages[:1]


@pytest.mark.asyncio
async def test_wildcard_route_resolves_underlying_model_minimum(local_model_cost_map):
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_key": "sk-fake"},
                "model_info": {"id": "wild-1"},
            }
        ]
    )

    deployments = await router.async_get_healthy_deployments(model="anthropic/claude-opus-4-6", request_kwargs={})

    assert deployments[0]["litellm_params"]["model"] == "anthropic/claude-opus-4-6"
    assert _get_min_token_count_for_deployments(deployments) == 4096


@pytest.mark.asyncio
async def test_async_filter_deployments_counts_the_prompt_off_the_event_loop():
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("anthropic/claude-fable-5")
    check = PromptCachingDeploymentCheck(cache=DualCache())
    deployments = _deployments("anthropic/claude-fable-5", "anthropic/claude-fable-5")
    messages = cast(list[AllMessageValues], [{"role": "user", "content": text * 100}])

    result, took, lags = await timed_with_loop_lags(
        lambda: check.async_filter_deployments(
            model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=messages
        )
    )

    assert result == deployments
    assert_loop_stayed_free(took, lags)


@pytest.mark.asyncio
async def test_async_log_success_event_counts_the_prompt_off_the_event_loop():
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("anthropic/claude-fable-5")
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    messages = cast(
        list[AllMessageValues],
        [{"role": "user", "content": [{"type": "text", "text": text * 100, "cache_control": {"type": "ephemeral"}}]}],
    )
    standard_logging_object = {
        "call_type": "acompletion",
        "custom_llm_provider": "anthropic",
        "model": "anthropic/claude-fable-5",
        "messages": messages,
        "model_id": "dep-1",
        "model_map_information": {"model_map_key": "anthropic/claude-fable-5", "model_map_value": None},
    }

    _, took, lags = await timed_with_loop_lags(
        lambda: check.async_log_success_event(
            kwargs={"standard_logging_object": standard_logging_object},
            response_obj=None,
            start_time=None,
            end_time=None,
        )
    )

    assert await PromptCachingCache(cache=cache).async_get_model_id(messages=messages, tools=None) == {
        "model_id": "dep-1"
    }
    assert_loop_stayed_free(took, lags)


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + seconds


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_llm_provider", "model", "model_map_value", "pinned_past_five_minutes"),
    [
        pytest.param(
            "anthropic",
            "claude-sonnet-4-5",
            {"cache_creation_input_token_cost_above_1hr": 6e-06},
            True,
            id="one-hour-cache-writes-priced",
        ),
        pytest.param(
            "anthropic",
            "claude-sonnet-4-5",
            {"cache_creation_input_token_cost_above_1hr": None},
            False,
            id="no-one-hour-cache-writes",
        ),
        pytest.param("anthropic", "claude-sonnet-4-5", None, False, id="model-not-in-cost-map"),
        pytest.param(
            "anthropic",
            "claude-sonnet-4-5",
            {"cache_creation_input_token_cost_above_1hr": 0.0},
            False,
            id="one-hour-cache-writes-priced-at-zero",
        ),
        pytest.param(
            "bedrock",
            "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"cache_creation_input_token_cost_above_1hr": None},
            True,
            id="bedrock-forwards-1h-from-the-base-model-entry",
        ),
        pytest.param(
            "bedrock",
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123",
            {"cache_creation_input_token_cost_above_1hr": 6.6e-06},
            False,
            id="bedrock-drops-1h-for-a-profile-arn-whatever-its-base-model",
        ),
    ],
)
async def test_one_hour_pin_follows_the_deployments_one_hour_cache_support(
    custom_llm_provider: str,
    model: str,
    model_map_value: dict[str, float | None] | None,
    pinned_past_five_minutes: bool,
):
    clock: Final = _ManualClock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock))
    check: Final = PromptCachingDeploymentCheck(cache=cache)
    messages: Final = cast(
        list[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "word " * 6000, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                ],
            }
        ],
    )

    await check.async_log_success_event(
        kwargs={
            "standard_logging_object": {
                "call_type": "acompletion",
                "custom_llm_provider": custom_llm_provider,
                "model": model,
                "messages": messages,
                "model_id": "dep-1",
                "model_map_information": {"model_map_key": model, "model_map_value": model_map_value},
            }
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    pins: Final = PromptCachingCache(cache=cache)
    assert await pins.async_get_model_id(messages=messages, tools=None) == {"model_id": "dep-1"}

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)

    assert (await pins.async_get_model_id(messages=messages, tools=None) is not None) is pinned_past_five_minutes


LONG_PROMPT = "word " * 3000
ONE_PIXEL_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _turn(*messages: dict) -> list[AllMessageValues]:
    return cast(list[AllMessageValues], list(messages))


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _marked(text: str) -> dict:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


@pytest.mark.asyncio
async def test_pin_survives_the_breakpoint_moving_to_the_next_turn():
    """
    The regression. Claude Code marks only the newest user message each turn, so the last breakpoint
    moves forward every turn. The key hashed the prefix up to that moving breakpoint, markers
    included, so no turn after the first ever found the pin the previous turn wrote, and a
    multi-deployment group re-rolled the deployment mid-session, paying a cache write on a
    deployment whose provider cache held nothing of the conversation.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    turn_one = _turn({"role": "user", "content": [_marked(LONG_PROMPT)]})
    turn_two = _turn(
        {"role": "user", "content": [_text(LONG_PROMPT)]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_marked("next")]},
    )

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=turn_one, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=turn_two
    )

    assert filtered == [deployments[1]]


@pytest.mark.asyncio
async def test_pin_survives_the_marked_message_coming_back_as_string_content():
    """
    Claude Code sends the message that carries a breakpoint as a one-block content list and re-sends
    it next turn as plain string content once the marker has moved on. The provider caches both
    shapes identically, so the key has to as well, or the walk-back never lands on the turn-one write.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    turn_one = _turn(
        {"role": "system", "content": [_marked(LONG_PROMPT)]},
        {"role": "user", "content": [_marked("hello")]},
    )
    turn_two = _turn(
        {"role": "system", "content": LONG_PROMPT},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": [_marked("again")]},
    )

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-1", messages=turn_one, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=turn_two
    )

    assert filtered == [deployments[0]]


@pytest.mark.asyncio
async def test_lookback_stops_where_the_provider_cache_stops():
    """
    Anthropic finds a cached prefix at most PROMPT_CACHE_LOOKBACK_POSITIONS block positions behind a
    breakpoint, the breakpoint block included. Probing further would pin to a deployment whose cache
    the provider will not consult, and probing less would drop pins the provider still honors.
    """
    prompt_cache = PromptCachingCache(cache=DualCache())
    await prompt_cache.async_add_model_id(
        model_id="dep-1", messages=_turn({"role": "user", "content": [_marked("block 0")]}), tools=None
    )

    def turn_with_blocks_after(count: int) -> list[AllMessageValues]:
        later = [_text(f"block {index}") for index in range(1, count)] + [_marked(f"block {count}")]
        return _turn({"role": "user", "content": [_text("block 0"), *later]})

    inside_window = turn_with_blocks_after(PROMPT_CACHE_LOOKBACK_POSITIONS - 1)
    past_window = turn_with_blocks_after(PROMPT_CACHE_LOOKBACK_POSITIONS)

    assert await prompt_cache.async_get_model_id(messages=inside_window, tools=None) == {"model_id": "dep-1"}
    assert prompt_cache.get_model_id(messages=inside_window, tools=None) == {"model_id": "dep-1"}
    assert await prompt_cache.async_get_model_id(messages=past_window, tools=None) is None
    assert prompt_cache.get_model_id(messages=past_window, tools=None) is None


@pytest.mark.asyncio
async def test_a_run_of_tool_blocks_counts_as_one_lookback_position():
    """
    The provider counts consecutive tool_use blocks as one lookback position, and consecutive
    tool_result blocks as one, in both the Anthropic and the OpenAI message shapes. An agent turn that
    fans out into many tool calls would otherwise push the previous breakpoint out of the window
    after a single turn, which is exactly when the conversation is longest and the cache matters most.
    """
    prompt_cache = PromptCachingCache(cache=DualCache())
    await prompt_cache.async_add_model_id(
        model_id="dep-1", messages=_turn({"role": "user", "content": [_marked("task")]}), tools=None
    )
    fan_out = PROMPT_CACHE_LOOKBACK_POSITIONS + 5

    def anthropic_shaped(tool_use_type: str, tool_result_type: str) -> list[AllMessageValues]:
        return _turn(
            {"role": "user", "content": [_text("task")]},
            {
                "role": "assistant",
                "content": [
                    {"type": tool_use_type, "id": f"call-{index}", "name": "read", "input": {"index": index}}
                    for index in range(fan_out)
                ],
            },
            {
                "role": "user",
                "content": [
                    *(
                        {"type": tool_result_type, "tool_use_id": f"call-{index}", "content": "ok"}
                        for index in range(fan_out)
                    ),
                    _marked("continue"),
                ],
            },
        )

    openai_shaped = _turn(
        {"role": "user", "content": [_text("task")]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call-{index}", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                for index in range(fan_out)
            ],
        },
        *({"role": "tool", "tool_call_id": f"call-{index}", "content": "ok"} for index in range(fan_out)),
        {"role": "user", "content": [_marked("continue")]},
    )

    assert await prompt_cache.async_get_model_id(messages=anthropic_shaped("tool_use", "tool_result"), tools=None) == {
        "model_id": "dep-1"
    }
    assert await prompt_cache.async_get_model_id(messages=openai_shaped, tools=None) == {"model_id": "dep-1"}
    assert await prompt_cache.async_get_model_id(messages=anthropic_shaped("text", "text"), tools=None) is None


@pytest.mark.asyncio
async def test_an_edited_earlier_block_does_not_inherit_the_pin():
    """
    Every key must bind the whole prefix before its block, not the block alone, or a conversation
    that repeats a pinned block after an edit walks back onto a cache the provider no longer holds.
    """
    prompt_cache = PromptCachingCache(cache=DualCache())
    await prompt_cache.async_add_model_id(
        model_id="dep-1", messages=_turn({"role": "user", "content": [_marked("original")]}), tools=None
    )
    edited = _turn(
        {"role": "user", "content": [_text("edited")]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_marked("original")]},
    )

    assert await prompt_cache.async_get_model_id(messages=edited, tools=None) is None


@pytest.mark.asyncio
async def test_swapped_roles_do_not_inherit_the_pin():
    """The message envelope is part of what the provider caches, so the same blocks under other roles key apart."""
    prompt_cache = PromptCachingCache(cache=DualCache())
    pinned = _turn(
        {"role": "user", "content": [_text("question")]},
        {"role": "assistant", "content": [_marked("answer")]},
    )
    swapped = _turn(
        {"role": "assistant", "content": [_text("question")]},
        {"role": "user", "content": [_marked("answer")]},
    )
    await prompt_cache.async_add_model_id(model_id="dep-1", messages=pinned, tools=None)

    assert await prompt_cache.async_get_model_id(messages=pinned, tools=None) == {"model_id": "dep-1"}
    assert await prompt_cache.async_get_model_id(messages=swapped, tools=None) is None


@pytest.mark.asyncio
async def test_raw_bytes_in_a_block_hash_instead_of_failing_the_request():
    """A block carrying raw bytes must key like any other block rather than raising out of the router filter."""
    prompt_cache = PromptCachingCache(cache=DualCache())
    binary_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b"\xff\xfe"}}
    turn = _turn({"role": "user", "content": [binary_block, _marked("describe")]})
    await prompt_cache.async_add_model_id(model_id="dep-1", messages=turn, tools=None)

    assert await prompt_cache.async_get_model_id(messages=turn, tools=None) == {"model_id": "dep-1"}


class _BrokenBatchReadCache(DualCache):
    async def async_batch_get_cache(self, keys, parent_otel_span=None, local_only=False, **kwargs):
        return None


@pytest.mark.asyncio
async def test_a_failed_batch_read_pins_nothing():
    """DualCache answers None rather than a list when the batch read raises, and routing must fall through."""
    prompt_cache = PromptCachingCache(cache=_BrokenBatchReadCache())

    assert (
        await prompt_cache.async_get_model_id(messages=_turn({"role": "user", "content": [_marked("x")]}), tools=None)
        is None
    )


@pytest.mark.asyncio
async def test_pin_matches_when_the_success_event_truncated_an_image_payload(monkeypatch, local_model_cost_map):
    """
    The success event only ever sees the standard logging payload, whose long base64 data URIs are
    replaced by size placeholders, while routing sees the raw request. Hashing the raw bytes on the
    read side would key every image-carrying session past its own pin.
    """
    call_id: Final = str(uuid.uuid4())
    capture = _SentMessagesCapture(call_id)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    image = {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG}}
    turn_one = _turn({"role": "user", "content": [image, _marked(LONG_PROMPT)]})

    await litellm.acompletion(
        model=AUTO_CACHING_MODEL,
        messages=copy.deepcopy(turn_one),
        mock_response="ok",
        api_key="sk-fake",
        litellm_call_id=call_id,
    )
    logged = await _eventually(lambda: capture.messages)
    assert logged is not None
    assert logged != turn_one

    cache = DualCache()
    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=logged, tools=None)
    turn_two = _turn(
        {"role": "user", "content": [image, _text(LONG_PROMPT)]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_marked("next")]},
    )
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)

    filtered = await PromptCachingDeploymentCheck(cache=cache).async_filter_deployments(
        model=MODEL_GROUP_ALIAS, healthy_deployments=deployments, messages=turn_two
    )

    assert filtered == [deployments[1]]


@pytest.mark.asyncio
async def test_claude_code_style_session_stays_on_one_deployment_across_turns(local_model_cost_map):
    """
    End to end over the router with a client that marks only the newest user message each turn, the
    way Claude Code does. Every turn has to land on the deployment that served the first one.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": AUTO_CACHING_MODEL, "api_key": "sk-fake"},
                "model_info": {"id": model_id},
            }
            for model_id in (f"dep-{number}" for number in range(1, 7))
        ],
        optional_pre_call_checks=["prompt_caching"],
    )
    user_turns = [LONG_PROMPT, *(f"follow-up {number}" for number in range(1, 9))]
    history: list[AllMessageValues] = []
    served: list[str] = []
    for text in user_turns:
        request = cast(list[AllMessageValues], [*history, {"role": "user", "content": [_marked(text)]}])
        response = await router.acompletion(model=MODEL_GROUP_ALIAS, messages=request, mock_response="ok")
        served.append(response._hidden_params["model_id"])
        pin_key = PromptCachingCache.get_prompt_caching_cache_key(request, None)
        assert await _eventually(functools.partial(router.prompt_caching_pin_cache.get_cache, key=pin_key)) is not None
        history = [*history, {"role": "user", "content": [_text(text)]}, {"role": "assistant", "content": "ok"}]

    assert served == [served[0]] * len(user_turns)


@pytest.fixture
def anthropic_messages():
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 500,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_router_with_prompt_caching(anthropic_messages):
    """
    if prompt caching supported model called with prompt caching valid prompt,
    then 2nd call should go to the same model.
    """
    from litellm.router import Router

    router = Router(
        model_list=[
            {
                "model_name": "claude-model",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                    "mock_response": "The sky is blue.",
                },
            },
            {
                "model_name": "claude-model",
                "litellm_params": {
                    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
                    "mock_response": "The sky is green.",
                },
            },
        ],
        optional_pre_call_checks=["prompt_caching"],
    )

    response = await router.acompletion(
        messages=anthropic_messages,
        model="claude-model",
        mock_response="The sky is blue.",
    )
    print("response=", response)

    initial_model_id = response._hidden_params["model_id"]

    cache = PromptCachingCache(
        cache=router.prompt_caching_pin_cache,
    )

    cached_model_id = await _eventually(lambda: cache.get_model_id(messages=anthropic_messages, tools=None))

    assert cached_model_id is not None
    prompt_caching_cache_key = PromptCachingCache.get_prompt_caching_cache_key(messages=anthropic_messages, tools=None)
    print(f"prompt_caching_cache_key: {prompt_caching_cache_key}")
    assert cached_model_id["model_id"] == initial_model_id

    new_messages = anthropic_messages + [{"role": "user", "content": "What is the weather in SF?"}]

    for _ in range(20):
        response = await router.acompletion(
            messages=new_messages,
            model="claude-model",
            mock_response="The sky is blue.",
        )
        print("response=", response)

        assert response._hidden_params["model_id"] == initial_model_id


_OPENAI_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "word " * 3000,
        "parameters": {"type": "object", "properties": {}},
    },
    "cache_control": {"type": "ephemeral"},
}
_ANTHROPIC_TOOL: Final = {
    "name": "get_weather",
    "description": "word " * 3000,
    "input_schema": {"type": "object", "properties": {}},
    "cache_control": {"type": "ephemeral"},
}


def _success_kwargs(
    *,
    call_type: str,
    model: str,
    model_id: str,
    request_messages: list,
    request_kwargs: dict,
) -> dict:
    """The success event shape: `messages` is what standard logging produced, `system`/`tools` stay raw."""
    return {
        "standard_logging_object": {
            "call_type": call_type,
            "model": model,
            "model_id": model_id,
            "custom_llm_provider": "anthropic",
            "model_map_information": {"model_map_key": model, "model_map_value": None},
            "messages": StandardLoggingPayloadSetup.append_system_prompt_messages(
                kwargs=request_kwargs, messages=copy.deepcopy(request_messages)
            ),
        },
        "system": request_kwargs.get("system"),
        "tools": request_kwargs.get("tools"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type,request_messages,request_kwargs",
    [
        pytest.param(
            "acompletion",
            [{"role": "user", "content": "weather?"}],
            {"tools": [_OPENAI_TOOL]},
            id="chat_tool_breakpoint_only",
        ),
        pytest.param(
            "anthropic_messages",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "word " * 3000, "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ],
            {"system": "you are terse"},
            id="anthropic_messages_string_system_and_marked_user_block",
        ),
        pytest.param(
            "anthropic_messages",
            [{"role": "user", "content": "hi"}],
            {
                "system": [
                    {
                        "type": "text",
                        "text": "word " * 3000,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ]
            },
            id="anthropic_messages_list_system_breakpoint_only",
        ),
        pytest.param(
            "anthropic_messages",
            [{"role": "user", "content": "weather?"}],
            {"tools": [_ANTHROPIC_TOOL]},
            id="anthropic_messages_tool_breakpoint_only",
        ),
        pytest.param(
            "acompletion",
            [{"role": "user", "content": "weather?"}],
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "word " * 3000,
                            "parameters": {"type": "object", "properties": {}},
                            "cache_control": {"type": "ephemeral"},
                        },
                    }
                ]
            },
            id="chat_nested_function_cache_control_only",
        ),
        pytest.param(
            "anthropic_messages",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "word " * 3000, "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ],
            {},
            id="control_message_breakpoint_no_system",
        ),
    ],
)
async def test_affinity_key_covers_the_tools_and_system_prompt_the_provider_sees(
    call_type, request_messages, request_kwargs
):
    """
    The write side keys the pin on what standard logging logged plus the raw `system`/`tools`
    kwargs, and the read side must rebuild the same prefix from the raw request. Anything the
    provider caches that the key ignores (tools, the /v1/messages system prompt) either writes no
    pin or writes one the lookup can never compute.
    """
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)

    await check.async_log_success_event(
        kwargs=_success_kwargs(
            call_type=call_type,
            model=AUTO_CACHING_MODEL,
            model_id="dep-1",
            request_messages=request_messages,
            request_kwargs=request_kwargs,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=cast(list[AllMessageValues], request_messages),
        request_kwargs=request_kwargs,
    )

    assert filtered == [deployments[0]]


@pytest.mark.asyncio
async def test_a_different_tool_set_does_not_share_the_pin():
    """Tools hash into the prefix, so a same-messages request with other tools must not read the pin."""
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    request_messages = [{"role": "user", "content": "weather?"}]
    other_tool = {**_OPENAI_TOOL, "function": {**_OPENAI_TOOL["function"], "name": "get_time"}}

    await check.async_log_success_event(
        kwargs=_success_kwargs(
            call_type="acompletion",
            model=AUTO_CACHING_MODEL,
            model_id="dep-1",
            request_messages=request_messages,
            request_kwargs={"tools": [_OPENAI_TOOL]},
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=cast(list[AllMessageValues], request_messages),
        request_kwargs={"tools": [other_tool]},
    )

    assert filtered == deployments


@pytest.mark.asyncio
async def test_a_pin_written_without_tools_is_read_when_tools_is_an_empty_list():
    """`tools=None` on the write and `tools=[]` on the read are the same empty seed."""
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    deployments = _deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)
    request_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "word " * 3000, "cache_control": {"type": "ephemeral"}}],
        }
    ]

    await check.async_log_success_event(
        kwargs=_success_kwargs(
            call_type="anthropic_messages",
            model=AUTO_CACHING_MODEL,
            model_id="dep-1",
            request_messages=request_messages,
            request_kwargs={},
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=cast(list[AllMessageValues], request_messages),
        request_kwargs={"tools": []},
    )

    assert filtered == [deployments[0]]


class _LoggedMessagesProbe(CustomLogger):
    def __init__(self) -> None:
        self.logged_messages: list[object] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.logged_messages.append((kwargs["standard_logging_object"]["messages"], kwargs["messages"]))


@pytest.mark.asyncio
async def test_pin_is_written_with_message_logging_off_while_loggers_still_see_redacted_messages(
    monkeypatch, local_model_cost_map
):
    """
    `turn_off_message_logging` redacted the shared success-event payload before the check read it, so the
    pin was keyed off `redacted-by-litellm`, never written, and a cacheable conversation bounced across the
    group. The pin must land whatever the flag says, while a real logger alongside still sees only the
    redacted messages.
    """
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    probe = _LoggedMessagesProbe()
    monkeypatch.setattr(litellm, "callbacks", [probe])
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": "anthropic/claude-opus-4-6", "api_key": "sk-fake"},
                "model_info": {"id": model_id},
            }
            for model_id in ("dep-1", "dep-2")
        ],
        optional_pre_call_checks=["prompt_caching"],
    )
    messages = _messages(word_count=5000)

    first = await router.acompletion(model=MODEL_GROUP_ALIAS, messages=messages, mock_response="ok")
    served_by = first._hidden_params["model_id"]

    affinity_key = PromptCachingCache.get_prompt_caching_cache_key(messages, None)
    pin = await _eventually(lambda: router.prompt_caching_pin_cache.get_cache(key=affinity_key))
    assert pin == {"model_id": served_by}, pin

    subsequent = [
        (await router.acompletion(model=MODEL_GROUP_ALIAS, messages=messages, mock_response="ok"))._hidden_params[
            "model_id"
        ]
        for _ in range(4)
    ]
    assert subsequent == [served_by] * 4

    await _eventually(lambda: len(probe.logged_messages) == 5)
    redacted = [{"role": "user", "content": "redacted-by-litellm"}]
    assert probe.logged_messages == [(redacted, redacted)] * 5, probe.logged_messages


RESPONSES_BRIDGED_MODEL: Final = "anthropic/claude-sonnet-4-6"
RESPONSES_NATIVE_MODEL: Final = "openai/gpt-5.4-mini"
_RESPONSES_TOOL: Final = {
    "type": "function",
    "name": "get_time",
    "parameters": {"type": "object", "properties": {}},
}


def _marked_input_text(text: str) -> dict[str, object]:
    return {"type": "input_text", "text": text, "cache_control": {"type": "ephemeral"}}


def _marked_responses_input() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": [_marked_input_text(LONG_PROMPT)]},
        {"role": "user", "content": "hello"},
    ]


def _responses_router(model: str) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": model, "api_key": "sk-fake"},
                "model_info": {"id": model_id},
            }
            for model_id in ("dep-1", "dep-2")
        ],
        optional_pre_call_checks=["prompt_caching"],
    )


async def _serve_response(router: litellm.Router, request_input: object) -> str:
    response = await router.aresponses(
        model=MODEL_GROUP_ALIAS,
        input=copy.deepcopy(request_input),
        instructions="Be terse.",
        tools=[_RESPONSES_TOOL],
        mock_response="ok",
    )
    return response._hidden_params["model_id"]


async def _deployments_serving_repeats(router: litellm.Router, request_input: object) -> set[str]:
    await _serve_response(router, request_input)
    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    return {await _serve_response(router, request_input) for _ in range(8)}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [RESPONSES_BRIDGED_MODEL, RESPONSES_NATIVE_MODEL], ids=["bridged", "native"])
async def test_repeated_responses_request_pins_to_one_deployment(model: str):
    assert len(await _deployments_serving_repeats(_responses_router(model), _marked_responses_input())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [RESPONSES_BRIDGED_MODEL, RESPONSES_NATIVE_MODEL], ids=["bridged", "native"])
async def test_responses_thread_stays_on_the_deployment_holding_its_prefix(model: str):
    router = _responses_router(model)
    first_turn = _marked_responses_input()
    pinned_deployment = await _serve_response(router, first_turn)
    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)

    later_turns = (
        [*first_turn, {"role": "assistant", "content": "ok"}, {"role": "user", "content": f"question {turn}"}]
        for turn in range(8)
    )

    assert {await _serve_response(router, turn) for turn in later_turns} == {pinned_deployment}


@pytest.mark.asyncio
async def test_responses_request_pins_after_prompt_caching_is_turned_off_and_on_again():
    router = _responses_router(RESPONSES_NATIVE_MODEL)
    await _serve_response(router, _marked_responses_input())
    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    router.update_settings(optional_pre_call_checks=[])
    router.update_settings(optional_pre_call_checks=["prompt_caching"])
    router.prompt_caching_pin_cache.in_memory_cache.flush_cache()

    assert len(await _deployments_serving_repeats(router, _marked_responses_input())) == 1


def _team_tagged_router(model: str) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": model, "api_key": "sk-fake", "tags": [team]},
                "model_info": {"id": model_id},
            }
            for model_id, team in (("dep-1", "team-a"), ("dep-2", "team-b"))
        ],
        optional_pre_call_checks=["prompt_caching"],
        enable_tag_filtering=True,
    )


async def _serve_team_response(router: litellm.Router, team: str) -> str:
    response = await router.aresponses(
        model=MODEL_GROUP_ALIAS,
        input=_marked_responses_input(),
        instructions="Be terse.",
        tools=[_RESPONSES_TOOL],
        litellm_metadata={"tags": [team]},
        mock_response="ok",
    )
    return response._hidden_params["model_id"]


async def _serve_team_completion(router: litellm.Router, team: str) -> str:
    response = await router.acompletion(
        model=MODEL_GROUP_ALIAS,
        messages=_messages(word_count=5000),
        metadata={"tags": [team]},
        mock_response="ok",
    )
    return response._hidden_params["model_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, serve",
    [(RESPONSES_NATIVE_MODEL, _serve_team_response), (RESPONSES_BRIDGED_MODEL, _serve_team_completion)],
    ids=["responses", "chat"],
)
async def test_prompt_cache_pin_never_overrides_the_request_tags(model: str, serve):
    router = _team_tagged_router(model)
    assert await serve(router, "team-a") == "dep-1"
    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)

    assert await serve(router, "team-b") == "dep-2"


@pytest.mark.asyncio
async def test_retry_moves_off_the_pinned_deployment_that_failed():
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {
                    "model": RESPONSES_BRIDGED_MODEL,
                    "api_key": "sk-fake",
                    "mock_response": mock_response,
                    "weight": weight,
                },
                "model_info": {"id": model_id},
            }
            for model_id, mock_response, weight in (
                ("dep-1", "litellm.InternalServerError", 1),
                ("dep-2", "ok", 1000),
            )
        ],
        optional_pre_call_checks=["prompt_caching"],
        num_retries=3,
        retry_after=0,
        disable_cooldowns=True,
    )
    messages: Final = _messages(word_count=5000)
    await PromptCachingCache(cache=router.prompt_caching_pin_cache).async_add_model_id(
        model_id="dep-1", messages=messages, tools=None
    )

    response: Final = await router.acompletion(model=MODEL_GROUP_ALIAS, messages=messages)

    assert response._hidden_params["model_id"] == "dep-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("reused_as", ["metadata", "litellm_metadata"])
async def test_pin_holds_for_a_metadata_dict_reused_after_a_request_failed_on_the_pinned_deployment(reused_as: str):
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": RESPONSES_BRIDGED_MODEL, "api_key": "sk-fake", "weight": weight},
                "model_info": {"id": model_id},
            }
            for model_id, weight in (("dep-1", 1000), ("dep-2", 1))
        ],
        optional_pre_call_checks=["prompt_caching"],
        num_retries=6,
        retry_after=0,
        allowed_fails=100,
    )
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="dep-1",
        original_exception=litellm.RateLimitError(
            message="upstream 429", llm_provider="anthropic", model=RESPONSES_BRIDGED_MODEL
        ),
        exception_status=429,
        cooldown_time=60,
    )
    reused_metadata: Final[dict[str, object]] = {}
    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(
            model=MODEL_GROUP_ALIAS,
            messages=[{"role": "user", "content": "hello"}],
            metadata=reused_metadata,
            mock_response="litellm.InternalServerError",
        )
    router.cooldown_cache.cooldown_store.evict_local("dep-1")
    messages: Final = _messages(word_count=5000)
    await PromptCachingCache(cache=router.prompt_caching_pin_cache).async_add_model_id(
        model_id="dep-2", messages=messages, tools=None
    )

    response: Final = await router.acompletion(
        model=MODEL_GROUP_ALIAS, messages=messages, mock_response="ok", **{reused_as: reused_metadata}
    )

    assert response._hidden_params["model_id"] == "dep-2"


@pytest.mark.asyncio
async def test_responses_mark_on_the_input_item_pins_to_one_deployment():
    request_input = [
        {"role": "system", "content": LONG_PROMPT, "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "hello"},
    ]

    assert len(await _deployments_serving_repeats(_responses_router(RESPONSES_BRIDGED_MODEL), request_input)) == 1


async def _read_responses_pin(
    check: PromptCachingDeploymentCheck,
    request: dict[str, object],
    models: tuple[str, ...] = (AUTO_CACHING_MODEL, AUTO_CACHING_MODEL),
) -> tuple[str, ...]:
    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=_deployments(*models),
        messages=None,
        request_kwargs=request,
    )
    return tuple(str(deployment["model_info"]["id"]) for deployment in filtered)


async def _log_responses_success(
    check: PromptCachingDeploymentCheck,
    logged_messages: object,
    metadata: dict[str, object],
    model_id: str = "dep-1",
    model_map_value: dict[str, float | None] | None = None,
    call_type: str = "aresponses",
) -> None:
    event = _success_kwargs(
        call_type=call_type,
        model=AUTO_CACHING_MODEL,
        model_id=model_id,
        request_messages=logged_messages,
        request_kwargs={},
    )
    await check.async_log_success_event(
        kwargs={
            **event,
            "standard_logging_object": {
                **event["standard_logging_object"],
                "model_map_information": {"model_map_key": AUTO_CACHING_MODEL, "model_map_value": model_map_value},
            },
            "litellm_params": {"litellm_metadata": metadata},
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )


async def _pinned_deployment_ids(
    first_request: dict[str, object], logged_messages: object, next_request: dict[str, object]
) -> tuple[str, ...]:
    check = PromptCachingDeploymentCheck(cache=DualCache())
    metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**first_request, "litellm_metadata": metadata})
    await _log_responses_success(check, logged_messages, metadata)
    return await _read_responses_pin(check, {**next_request, "litellm_metadata": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logged_messages",
    [
        _marked_responses_input(),
        [
            {"role": "system", "content": "Be terse."},
            {"role": "system", "content": LONG_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        ],
        [{"role": "user", "content": REDACTED_BY_LITELLM}],
    ],
    ids=["client_input", "bridged_chat_messages", "redacted"],
)
async def test_responses_pin_does_not_depend_on_what_the_success_event_logged(
    logged_messages: list[dict[str, object]],
):
    request = {"input": _marked_responses_input(), "instructions": "Be terse.", "tools": [_RESPONSES_TOOL]}

    pinned = await _pinned_deployment_ids(request, logged_messages, request)

    assert pinned == ("dep-1",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_pin",
    [["forged-redis-key", 3600], {"cache_key": "forged-redis-key", "marked_ttl_seconds": 3600}],
    ids=["array", "object"],
)
async def test_responses_pin_sent_in_the_request_metadata_is_never_written(forged_pin: object):
    cache = DualCache()

    await _log_responses_success(
        PromptCachingDeploymentCheck(cache=cache),
        _marked_responses_input(),
        {"responses_prompt_cache_pin": forged_pin},
    )

    assert not cache.in_memory_cache.cache_dict


@pytest.mark.asyncio
async def test_interleaved_responses_requests_each_pin_their_own_prompt():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    terse = {"input": _marked_responses_input(), "instructions": "Be terse."}
    verbose = {"input": _marked_responses_input(), "instructions": "Be verbose."}
    terse_metadata: dict[str, object] = {}
    verbose_metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**terse, "litellm_metadata": terse_metadata})
    await _read_responses_pin(check, {**verbose, "litellm_metadata": verbose_metadata})

    await _log_responses_success(check, _marked_responses_input(), terse_metadata, model_id="dep-1")
    await _log_responses_success(check, _marked_responses_input(), verbose_metadata, model_id="dep-2")

    assert await _read_responses_pin(check, {**terse, "litellm_metadata": {}}) == ("dep-1",)
    assert await _read_responses_pin(check, {**verbose, "litellm_metadata": {}}) == ("dep-2",)


@pytest.mark.asyncio
async def test_retried_responses_request_leaves_no_pin_in_flight_once_it_succeeds():
    pins_in_flight = InMemoryCache()
    check = PromptCachingDeploymentCheck(cache=DualCache(), responses_pins_in_flight=pins_in_flight)
    metadata: dict[str, object] = {}
    request = {"input": _marked_responses_input(), "litellm_metadata": metadata}
    await _read_responses_pin(check, request)
    await _read_responses_pin(check, request)

    await _log_responses_success(check, _marked_responses_input(), metadata)

    assert not pins_in_flight.cache_dict


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["acompact_responses", "acreate_interaction"])
async def test_request_sharing_the_responses_input_shape_leaves_no_pin_behind(call_type: str):
    cache = DualCache()
    pins_in_flight = InMemoryCache()
    check = PromptCachingDeploymentCheck(cache=cache, responses_pins_in_flight=pins_in_flight)
    metadata: dict[str, object] = {}
    await _read_responses_pin(check, {"input": _marked_responses_input(), "litellm_metadata": metadata})
    assert pins_in_flight.cache_dict

    await _log_responses_success(check, _marked_responses_input(), metadata, call_type=call_type)

    assert not pins_in_flight.cache_dict
    assert not cache.in_memory_cache.cache_dict


@pytest.mark.asyncio
async def test_responses_request_served_by_the_only_healthy_deployment_pins_it():
    cache = _PinLookupCounter()
    check = PromptCachingDeploymentCheck(cache=cache, group_has_alternatives=lambda group: True)
    request = {"input": _marked_responses_input(), "instructions": "Be terse."}
    only_healthy = [_deployments(AUTO_CACHING_MODEL, AUTO_CACHING_MODEL)[1]]
    metadata: dict[str, object] = {}
    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=only_healthy,
        messages=None,
        request_kwargs={**request, "litellm_metadata": metadata},
    )
    assert filtered == only_healthy
    assert cache.pin_lookups == 0

    await _log_responses_success(check, _marked_responses_input(), metadata, model_id="dep-2")

    assert await _read_responses_pin(check, {**request, "litellm_metadata": {}}) == ("dep-2",)


@pytest.mark.asyncio
async def test_responses_request_to_a_single_deployment_group_skips_the_prefix_hash():
    cache = _PinLookupCounter()
    pins_in_flight = InMemoryCache()
    check = PromptCachingDeploymentCheck(
        cache=cache, responses_pins_in_flight=pins_in_flight, group_has_alternatives=lambda group: False
    )
    deployments = _deployments(AUTO_CACHING_MODEL)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=None,
        request_kwargs={"input": _marked_responses_input(), "litellm_metadata": {}},
    )

    assert filtered == deployments
    assert cache.pin_lookups == 0
    assert not pins_in_flight.cache_dict


@pytest.mark.asyncio
async def test_responses_request_to_a_single_deployment_router_group_writes_no_pin(monkeypatch):
    router = litellm.Router(
        model_list=[
            {
                "model_name": MODEL_GROUP_ALIAS,
                "litellm_params": {"model": RESPONSES_NATIVE_MODEL, "api_key": "sk-fake"},
                "model_info": {"id": "dep-1"},
            }
        ],
        optional_pre_call_checks=["prompt_caching"],
    )
    logged_after_the_check: list[str] = []

    class _LoggedAfterTheCheck(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            logged_after_the_check.append(kwargs["litellm_call_id"])

    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, _LoggedAfterTheCheck()])

    assert await _serve_response(router, _marked_responses_input()) == "dep-1"

    assert await _eventually(lambda: logged_after_the_check)
    assert not router.prompt_caching_pin_cache.in_memory_cache.cache_dict


def _distinct_marked_responses_input(index: int) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": [_marked_input_text(f"prompt {index} {LONG_PROMPT}")]},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_responses_prompts_served_while_a_peer_cools_down_stay_pinned_once_it_recovers():
    router = _responses_router(RESPONSES_NATIVE_MODEL)
    pins = router.prompt_caching_pin_cache.in_memory_cache.cache_dict
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="dep-1",
        original_exception=litellm.RateLimitError(
            message="upstream 429", llm_provider="openai", model=RESPONSES_NATIVE_MODEL
        ),
        exception_status=429,
        cooldown_time=60,
    )
    prompts = [_distinct_marked_responses_input(index) for index in range(8)]
    for prompt in prompts:
        pins_before = len(pins)
        assert await _serve_response(router, prompt) == "dep-2"
        assert await _eventually(lambda before=pins_before: len(pins) > before)
    router.cooldown_cache.cooldown_store.evict_local("dep-1")
    unpinned = {
        (
            await router.aresponses(model=MODEL_GROUP_ALIAS, input=f"unpinned {index}", mock_response="ok")
        )._hidden_params["model_id"]
        for index in range(16)
    }
    assert unpinned == {"dep-1", "dep-2"}

    assert {await _serve_response(router, prompt) for prompt in prompts} == {"dep-2"}


@pytest.mark.asyncio
async def test_responses_with_other_instructions_are_a_different_prefix():
    pinned = await _pinned_deployment_ids(
        {"input": _marked_responses_input(), "instructions": "Be terse."},
        _marked_responses_input(),
        {"input": _marked_responses_input(), "instructions": "Be verbose."},
    )

    assert len(pinned) == 2


@pytest.mark.asyncio
async def test_responses_with_another_tool_set_are_a_different_prefix():
    other_tool = {**_RESPONSES_TOOL, "name": "get_weather"}

    pinned = await _pinned_deployment_ids(
        {"input": _marked_responses_input(), "tools": [_RESPONSES_TOOL]},
        _marked_responses_input(),
        {"input": _marked_responses_input(), "tools": [other_tool]},
    )

    assert len(pinned) == 2


@pytest.mark.asyncio
async def test_responses_custom_tools_with_another_grammar_are_a_different_prefix():
    custom_tool: Final = {
        "type": "custom",
        "name": "run_sql",
        "description": "Run a query",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: SELECT"},
    }
    other_grammar: Final = {**custom_tool, "format": {"type": "grammar", "syntax": "lark", "definition": "start: UPDATE"}}

    pinned = await _pinned_deployment_ids(
        {"input": _marked_responses_input(), "tools": [custom_tool]},
        _marked_responses_input(),
        {"input": _marked_responses_input(), "tools": [other_grammar]},
    )

    assert len(pinned) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parameters",
    [{"type": "object", "properties": {}}, {"properties": {"city": {"type": "string"}}}],
    ids=["typed_parameters", "untyped_parameters"],
)
async def test_chat_and_responses_requests_sending_the_same_tool_share_one_prefix(parameters: dict[str, object]):
    described_tool: Final = {**_RESPONSES_TOOL, "description": "Get the time", "parameters": parameters}
    chat_tool: Final = {
        "type": "function",
        "function": {key: value for key, value in described_tool.items() if key != "type"},
    }
    router: Final = _responses_router(RESPONSES_BRIDGED_MODEL)
    await router.acompletion(
        model=MODEL_GROUP_ALIAS,
        messages=[
            {"role": "system", "content": [{**_marked_input_text(LONG_PROMPT), "type": "text"}]},
            {"role": "user", "content": "hello"},
        ],
        tools=[chat_tool],
        mock_response="ok",
    )
    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    chat_pin_keys: Final = set(router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    router.prompt_caching_pin_cache.in_memory_cache.flush_cache()

    await router.aresponses(
        model=MODEL_GROUP_ALIAS, input=_marked_responses_input(), tools=[described_tool], mock_response="ok"
    )

    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    assert set(router.prompt_caching_pin_cache.in_memory_cache.cache_dict) == chat_pin_keys


@pytest.mark.asyncio
async def test_responses_marked_tool_set_long_enough_to_cache_pins_on_its_own():
    marked_tool = {**_RESPONSES_TOOL, "description": LONG_PROMPT, "cache_control": {"type": "ephemeral"}}
    request = {"input": "hello", "tools": [marked_tool]}

    pinned = await _pinned_deployment_ids(request, [{"role": "user", "content": "hello"}], request)

    assert pinned == ("dep-1",)


@pytest.mark.asyncio
async def test_responses_continuation_turn_is_left_to_the_deployment_holding_its_previous_response(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    new_thread = {"input": "hello", "instructions": LONG_PROMPT}

    pinned = await _pinned_deployment_ids(
        new_thread, [{"role": "user", "content": "hello"}], {**new_thread, "previous_response_id": "resp_1"}
    )

    assert len(pinned) == 2


@pytest.mark.asyncio
async def test_responses_answer_from_a_fallback_group_keeps_the_primary_groups_pin():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    request = {"input": _marked_responses_input()}
    assert await _serve_turn(check, request["input"]) == 2
    fallback_hop_metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**request, "litellm_metadata": fallback_hop_metadata})

    await _log_responses_success(check, request["input"], fallback_hop_metadata, model_id="fallback-dep")

    pinned = await _read_responses_pin(check, {**request, "litellm_metadata": {}})
    assert pinned == ("dep-1",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answering_deployment,expected_pin",
    [("dep-1", ("dep-1", "dep-2")), ("dep-2", ("dep-2",))],
    ids=["below_its_minimum", "above_its_minimum"],
)
async def test_responses_pin_goes_only_to_a_deployment_whose_model_caches_the_prefix(
    answering_deployment: str, expected_pin: tuple[str, ...]
):
    mixed_group = ("anthropic/claude-opus-4-5", "anthropic/claude-sonnet-4-5")
    check = PromptCachingDeploymentCheck(cache=DualCache())
    request = {"input": [{"role": "user", "content": [_marked_input_text("word " * 1400)]}]}
    metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**request, "litellm_metadata": metadata}, mixed_group)

    await _log_responses_success(check, request["input"], metadata, model_id=answering_deployment)

    pinned = await _read_responses_pin(check, {**request, "litellm_metadata": {}}, mixed_group)
    assert pinned == expected_pin


@pytest.mark.asyncio
async def test_embedding_request_never_takes_a_responses_pin():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    pinned_input = _marked_responses_input()
    assert await _serve_turn(check, pinned_input) == 2
    assert len(await _read_responses_pin(check, {"input": pinned_input, "litellm_metadata": {}})) == 1

    filtered = await _read_responses_pin(check, {"input": pinned_input, "metadata": {}})

    assert len(filtered) == 2


class _InputReadOnlyOnceTheLoopTurns(list[dict[str, object]]):
    def __init__(self, items: list[dict[str, object]], loop_turned: threading.Event) -> None:
        super().__init__(items)
        self.loop_turned: Final = loop_turned
        self.loop_turned_before_first_read: bool | None = None

    def __iter__(self):
        if self.loop_turned_before_first_read is None:
            self.loop_turned_before_first_read = self.loop_turned.wait(timeout=5)
        return super().__iter__()


@pytest.mark.asyncio
async def test_responses_thread_is_converted_off_the_event_loop():
    loop_turned = threading.Event()
    thread = _InputReadOnlyOnceTheLoopTurns(_marked_responses_input(), loop_turned)
    asyncio.get_running_loop().call_soon(loop_turned.set)

    await _read_responses_pin(
        PromptCachingDeploymentCheck(cache=DualCache()), {"input": thread, "litellm_metadata": {}}
    )

    assert thread.loop_turned_before_first_read is True


@pytest.mark.asyncio
async def test_responses_long_instructions_count_toward_the_caching_minimum(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    request = {"input": "hello", "instructions": LONG_PROMPT}

    pinned = await _pinned_deployment_ids(request, [{"role": "user", "content": "hello"}], request)

    assert pinned == ("dep-1",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_map_value,pinned_past_five_minutes",
    [
        pytest.param({"cache_creation_input_token_cost_above_1hr": 6.6e-06}, True, id="one-hour-cache-writes"),
        pytest.param({"cache_creation_input_token_cost_above_1hr": None}, False, id="no-one-hour-cache-writes"),
    ],
)
async def test_responses_one_hour_pin_follows_the_deployments_one_hour_cache_support(
    model_map_value: dict[str, float | None], pinned_past_five_minutes: bool
):
    clock = _ManualClock()
    check = PromptCachingDeploymentCheck(cache=DualCache(in_memory_cache=InMemoryCache(clock=clock)))
    marked = {**_marked_input_text(LONG_PROMPT), "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    request = {"input": [{"role": "user", "content": [marked]}]}
    metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**request, "litellm_metadata": metadata})
    await _log_responses_success(check, request["input"], metadata, model_map_value=model_map_value)

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)

    assert (len(await _read_responses_pin(check, {**request, "litellm_metadata": {}})) == 1) is pinned_past_five_minutes


@pytest.mark.asyncio
async def test_responses_turn_without_input_leaves_routing_to_its_previous_response(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    request = {"previous_response_id": "resp_1", "instructions": LONG_PROMPT}

    pinned = await _pinned_deployment_ids(request, None, request)

    assert len(pinned) == 2


async def _serve_turn(check: PromptCachingDeploymentCheck, request_input: object) -> int:
    metadata: dict[str, object] = {}
    pinned = await _read_responses_pin(check, {"input": request_input, "litellm_metadata": metadata})
    await _log_responses_success(check, request_input, metadata)
    return len(pinned)


@pytest.mark.asyncio
async def test_responses_thread_longer_than_the_lookback_window_keeps_its_deployment(monkeypatch):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    check = PromptCachingDeploymentCheck(cache=DualCache())
    turns = [
        {"role": "system", "content": LONG_PROMPT},
        *(
            {"role": "assistant" if index % 2 else "user", "content": f"turn {index}"}
            for index in range(2 * PROMPT_CACHE_LOOKBACK_POSITIONS)
        ),
    ]

    served = [await _serve_turn(check, turns[:end]) for end in range(2, len(turns) + 1, 2)]

    assert served[1:] == [1] * (len(served) - 1)


@pytest.mark.asyncio
async def test_responses_failover_writes_the_pin_for_the_deployment_that_answered():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    request = {"input": _marked_responses_input()}
    metadata: dict[str, object] = {}
    await _read_responses_pin(check, {**request, "litellm_metadata": metadata})
    await _read_responses_pin(check, {**request, "litellm_metadata": metadata})
    await _log_responses_success(check, _marked_responses_input(), metadata, model_id="dep-2")

    pinned = await _read_responses_pin(check, {**request, "litellm_metadata": {}})

    assert pinned == ("dep-2",)


@pytest.mark.asyncio
async def test_responses_pin_covers_only_what_follows_the_newest_compaction():
    compaction = {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps({"type": "compaction", "content": "summary"}),
    }
    prefix = {"role": "user", "content": [_marked_input_text(LONG_PROMPT)]}

    first_input = [{"role": "user", "content": "dropped turn"}, compaction, prefix]

    pinned = await _pinned_deployment_ids(
        {"input": first_input},
        first_input,
        {"input": [{"role": "user", "content": "another dropped turn"}, compaction, prefix]},
    )

    assert pinned == ("dep-1",)


@pytest.mark.asyncio
async def test_responses_image_in_the_prefix_pins_to_one_deployment():
    request_input = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": ONE_PIXEL_PNG}, _marked_input_text(LONG_PROMPT)],
        },
        {"role": "user", "content": "hello"},
    ]

    assert len(await _deployments_serving_repeats(_responses_router(RESPONSES_BRIDGED_MODEL), request_input)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_input",
    [[{"role": "system", "content": LONG_PROMPT}, {"role": "user", "content": "hello"}], LONG_PROMPT],
    ids=["item_list", "plain_string"],
)
async def test_responses_auto_cached_prefix_pins_to_one_deployment(monkeypatch, request_input: object):
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)

    assert len(await _deployments_serving_repeats(_responses_router(RESPONSES_BRIDGED_MODEL), request_input)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_input",
    [
        [LONG_PROMPT, LONG_PROMPT],
        [{"role": "user", "content": 7}],
        7,
        [{"type": "web_search_call"}],
        [{"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search"}}],
        [{"type": ["message"], "role": "user", "content": LONG_PROMPT}],
        [{"role": "user", "content": [{"type": "input_image"}]}],
    ],
    ids=[
        "embedding_strings",
        "non_text_item_content",
        "integer",
        "web_search_call_without_id",
        "search_action_without_query",
        "unhashable_item_type",
        "image_without_a_source",
    ],
)
async def test_input_that_is_not_a_responses_prompt_never_narrows_or_raises(request_input: object):
    cache = DualCache()
    check = PromptCachingDeploymentCheck(cache=cache)
    metadata: dict[str, object] = {}

    filtered = await _read_responses_pin(check, {"input": request_input, "litellm_metadata": metadata})
    await _log_responses_success(check, request_input, metadata)

    assert filtered == ("dep-1", "dep-2")
    assert not cache.in_memory_cache.cache_dict


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "next_tools,expected_pin",
    [([{"type": "image_generation"}], ("dep-1",)), ([{"type": "web_search"}], ("dep-1", "dep-2"))],
    ids=["same_native_tool", "other_native_tool"],
)
async def test_responses_native_tools_are_part_of_the_prefix(
    next_tools: list[dict[str, object]], expected_pin: tuple[str, ...]
):
    request_input = _marked_responses_input()

    pinned = await _pinned_deployment_ids(
        {"input": request_input, "tools": [{"type": "image_generation"}]},
        request_input,
        {"input": request_input, "tools": next_tools},
    )

    assert pinned == expected_pin


@pytest.mark.asyncio
@pytest.mark.parametrize("native_tool_type", ["shell", "image_generation", "computer_use", "local_shell"])
async def test_responses_native_tool_routed_to_a_native_model_logs_no_dropped_tool_warning(
    caplog: pytest.LogCaptureFixture, native_tool_type: str
):
    router = _responses_router(RESPONSES_NATIVE_MODEL)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        await router.aresponses(
            model=MODEL_GROUP_ALIAS,
            input=_marked_responses_input(),
            tools=[{"type": native_tool_type}],
            mock_response="ok",
        )

    assert not [record for record in caplog.records if "Dropping Responses API tool" in record.getMessage()]


class _MetadataSerializingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.serialized_metadata: list[str] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.serialized_metadata.append(json.dumps(kwargs["litellm_params"]["metadata"]))


@pytest.mark.asyncio
async def test_logger_serializing_the_request_metadata_never_sees_the_prompt_cache_key(monkeypatch):
    serializing_logger = _MetadataSerializingLogger()
    monkeypatch.setattr(litellm, "callbacks", [serializing_logger])
    router = _responses_router(RESPONSES_NATIVE_MODEL)

    assert len(await _deployments_serving_repeats(router, _marked_responses_input())) == 1

    assert await _eventually(lambda: len(serializing_logger.serialized_metadata) == 9)
    logged_metadata = "".join(serializing_logger.serialized_metadata)
    pin_keys = tuple(router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    assert pin_keys
    assert not [pin_key for pin_key in pin_keys if pin_key.split(":")[1] in logged_metadata]


@pytest.mark.asyncio
async def test_responses_call_hands_back_the_callers_metadata_without_the_pin_token():
    router: Final = _responses_router(RESPONSES_BRIDGED_MODEL)
    caller_metadata: Final[dict[str, object]] = {"tags": ["team-a"]}

    await router.aresponses(
        model=MODEL_GROUP_ALIAS,
        input=_marked_responses_input(),
        litellm_metadata=caller_metadata,
        mock_response="ok",
    )

    assert await _eventually(lambda: router.prompt_caching_pin_cache.in_memory_cache.cache_dict)
    assert "responses_prompt_cache_pin" not in caller_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_in_request,expected_deployments",
    [({}, ("dep-2",)), ({"_excluded_deployment_ids": ["dep-2"]}, ("dep-1", "dep-2"))],
    ids=["previous_deployment_healthy", "previous_deployment_failed_this_request"],
)
async def test_responses_continuation_turn_goes_to_the_deployment_that_answered_its_previous_response(
    failed_in_request: dict[str, object], expected_deployments: tuple[str, ...]
):
    previous_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id("anthropic", "dep-2", "resp_x")
    request: dict[str, object] = {
        "input": "hello",
        "previous_response_id": previous_response_id,
        "litellm_metadata": {},
    }

    pinned = await _read_responses_pin(
        PromptCachingDeploymentCheck(cache=DualCache()), {**request, **failed_in_request}
    )

    assert pinned == expected_deployments


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tool", ["get_time", None, 42], ids=["string", "null", "number"])
async def test_responses_request_with_a_malformed_tool_is_left_to_the_provider(bad_tool: object):
    request: dict[str, object] = {"input": _marked_responses_input(), "tools": [bad_tool], "litellm_metadata": {}}

    pinned = await _read_responses_pin(PromptCachingDeploymentCheck(cache=DualCache()), request)

    assert pinned == ("dep-1", "dep-2")

import asyncio
import copy
from typing import List, cast

import pytest


import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
    _get_min_token_count_for_deployments,
)
from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_prompt_cache_min_tokens, is_prompt_caching_valid_prompt, token_counter

MODEL_GROUP_ALIAS = "my-claude-group"
OPUS_4_6_MIN_TOKENS = 4096


@pytest.fixture(autouse=True)
def _local_model_cost_map_autouse(local_model_cost_map):
    """Every test here reads `prompt_cache_min_tokens`, which only the in-repo map
    carries, so the shared local_model_cost_map fixture (conftest.py) is autouse
    for the whole file."""
    yield



def _deployments(*models: str) -> List[dict]:
    return [
        {
            "model_name": MODEL_GROUP_ALIAS,
            "litellm_params": {"model": model},
            "model_info": {"id": f"dep-{index}"},
        }
        for index, model in enumerate(models, start=1)
    ]


def _messages(word_count: int) -> List[AllMessageValues]:
    return cast(
        List[AllMessageValues],
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

    token_count = token_counter(messages=messages, model="anthropic/claude-opus-4-5", use_default_image_token_count=True)
    assert 1024 < token_count < 4096

    assert is_prompt_caching_valid_prompt(model="anthropic/claude-opus-4-5", messages=messages) is False
    assert is_prompt_caching_valid_prompt(model="anthropic/claude-sonnet-4-5", messages=messages) is True


def test_get_min_token_count_for_deployments_falls_back_to_default_for_empty_group():
    """An empty group has no member minimum to read, so it must fall back rather than crash."""
    assert _get_min_token_count_for_deployments([]) == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT


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

    token_count = token_counter(messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True)
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

    token_count = token_counter(messages=messages, model="anthropic/claude-opus-4-6", use_default_image_token_count=True)
    assert token_count > OPUS_4_6_MIN_TOKENS

    await PromptCachingCache(cache=cache).async_add_model_id(model_id="dep-2", messages=messages, tools=None)

    filtered = await check.async_filter_deployments(
        model=MODEL_GROUP_ALIAS,
        healthy_deployments=deployments,
        messages=messages,
    )

    assert filtered == [deployments[1]]


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


def _auto_caching_messages() -> List[AllMessageValues]:
    """A prompt over the model minimum that carries no client cache_control."""
    return cast(
        List[AllMessageValues],
        [
            {"role": "system", "content": "word " * 3000},
            {"role": "user", "content": "hello"},
        ],
    )


def _affinity_messages(messages: List[AllMessageValues]) -> List[AllMessageValues]:
    """The messages the check keys deployment affinity on, for a group of `AUTO_CACHING_MODEL`."""
    return AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages,
        models=(AUTO_CACHING_MODEL,),
    )


class _SentMessagesCapture(CustomLogger):
    def __init__(self):
        self.messages: List[AllMessageValues] | None = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
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
    capture = _SentMessagesCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    messages = _auto_caching_messages()

    await litellm.acompletion(
        model=AUTO_CACHING_MODEL,
        messages=copy.deepcopy(messages),
        mock_response="ok",
        api_key="sk-fake",
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
    assert await _eventually(lambda: router.cache.get_cache(key=affinity_key)) is not None

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
        List[AllMessageValues],
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

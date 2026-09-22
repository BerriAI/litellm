import asyncio
import copy
import functools
from typing import Final, cast

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT, PROMPT_CACHE_LOOKBACK_POSITIONS
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
                "model": model,
                "messages": messages,
                "model_id": "dep-2",
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
    def __init__(self):
        self.messages: list[AllMessageValues] | None = None

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
    from tests.test_litellm.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("anthropic/claude-fable-5")
    check = PromptCachingDeploymentCheck(cache=DualCache())
    deployments = _deployments("anthropic/claude-fable-5")
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
    from tests.test_litellm.litellm_core_utils.event_loop_lag import (
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
        "model": "anthropic/claude-fable-5",
        "messages": messages,
        "model_id": "dep-1",
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
    capture = _SentMessagesCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    image = {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG}}
    turn_one = _turn({"role": "user", "content": [image, _marked(LONG_PROMPT)]})

    await litellm.acompletion(
        model=AUTO_CACHING_MODEL, messages=copy.deepcopy(turn_one), mock_response="ok", api_key="sk-fake"
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
        assert await _eventually(functools.partial(router.cache.get_cache, key=pin_key)) is not None
        history = [*history, {"role": "user", "content": [_text(text)]}, {"role": "assistant", "content": "ok"}]

    assert served == [served[0]] * len(user_turns)

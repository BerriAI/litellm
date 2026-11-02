"""
Characterization of the pre-call checks on a router core: with ``enable_pre_call_checks`` the
context window (token counting over messages, Responses input, instructions, tools and an Anthropic
system block), the allowed region and unsupported params (``response_format``) narrow a group's
deployments before one is picked; image requests skip text-only deployments; budget limits drop
deployments over budget. Covers the errors raised when nothing is left (type and message), which
deployment ``select`` and the endpoints reach, and that no provider call is made then.

Each test asserts what the Python core does today; oddities are named in the test's docstring.
"""

import re
from collections.abc import Callable, Generator, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.core.construction_precall_helpers import CallbackHost, restoring_model_cost, windowed
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, python_core

MakeCore = Callable[..., RouterCore]
PICKS: Final = 60  # three-way shuffles miss a deployment in 20 picks about once in a thousand runs
SHORT: Final = [{"role": "user", "content": "hi"}]
LONG_TEXT: Final = " ".join(["word"] * 100)
LONG: Final = [{"role": "user", "content": LONG_TEXT}]
IMAGE: Final = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ],
    }
]
UNCOUNTABLE: Final = [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"detail": "not-a-real-detail-value"}}]}
]
JSON_MODE: Final = {"response_format": {"type": "json_object"}}
NO_JSON_MODE_MODEL: Final = "openai/gpt-3.5-turbo-16k"
JSON_MODE_MODEL: Final = "openai/gpt-4o"
CONTEXT_WINDOW_PREFIX: Final = (
    "litellm.ContextWindowExceededError: litellm.BadRequestError: litellm._pre_call_checks: Context Window exceeded "
    "for given call. No models have context window large enough for this call.\n"
)
UNSUPPORTED_MESSAGE: Final = (
    "litellm.UnsupportedParamsError: no deployment of model=g supports ['response_format']. To drop these, set "
    "`litellm.drop_params=True` or for proxy `litellm_settings: drop_params: true`, or send "
    "allowed_openai_params=['response_format'] to pass them through"
)


@pytest.fixture
def host() -> ScriptedHost:
    return CallbackHost()


@pytest.fixture(autouse=True)
def isolated_model_cost() -> Generator[None, None, None]:
    """Routers write their deployments' model info into the process cost map; undo it after each test."""
    yield from restoring_model_cost("gpt-3.5-turbo-16k")


async def _picked(
    core: RouterCore, model: str = "g", times: int = PICKS, **select_arguments: object
) -> set[str]:  # kwargs-ok: select's arguments
    picked: Final[set[str]] = set()
    for _ in range(times):
        chosen = await core.select(model, **select_arguments)  # pyright: ignore[reportCallIssue]
        assert isinstance(chosen, Mapping)
        picked.add(str(chosen["model_info"]["id"]))
    return picked


def _small_and_big(
    make_core: MakeCore, small: int = 5, **arguments: object
) -> RouterCore:  # kwargs-ok: constructor arguments
    return make_core(
        model_list=[windowed("g", "small", small), windowed("g", "big", 10000)],
        **{"enable_pre_call_checks": True, **arguments},
    )


def _json_mode_group(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: constructor arguments
    return make_core(
        model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL), deployment("g", "new", model=JSON_MODE_MODEL)],
        **{"enable_pre_call_checks": True, **arguments},
    )


# --- context window ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_enable_pre_call_checks_a_too_small_context_window_is_still_picked(make_core: MakeCore) -> None:
    core: Final = _small_and_big(make_core, enable_pre_call_checks=False)
    assert await _picked(core, messages=LONG) == {"small", "big"}


@pytest.mark.asyncio
async def test_without_enable_pre_call_checks_a_too_long_prompt_reaches_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)])
    await core.acompletion(model="g", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["small"]


@pytest.mark.asyncio
async def test_select_skips_a_deployment_whose_context_window_the_prompt_exceeds(make_core: MakeCore) -> None:
    assert await _picked(_small_and_big(make_core), messages=LONG) == {"big"}


@pytest.mark.asyncio
async def test_a_short_prompt_may_pick_either_deployment(make_core: MakeCore) -> None:
    assert await _picked(_small_and_big(make_core, small=50), messages=SHORT) == {"small", "big"}


@pytest.mark.asyncio
async def test_async_completion_reaches_only_a_deployment_whose_context_window_fits(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _small_and_big(make_core)
    for _ in range(8):
        await core.acompletion(model="g", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) == {"big"}


def test_sync_completion_reaches_only_a_deployment_whose_context_window_fits(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _small_and_big(make_core)
    for _ in range(8):
        core.completion(model="g", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) == {"big"}


@pytest.mark.asyncio
async def test_pass_through_selection_skips_a_deployment_whose_context_window_the_prompt_exceeds(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=[
            windowed("g", "small", 5, use_in_pass_through=True),
            windowed("g", "big", 10000, use_in_pass_through=True),
        ],
        enable_pre_call_checks=True,
    )
    assert await _picked(core, kind="pass_through", messages=LONG) == {"big"}


@pytest.mark.asyncio
async def test_select_without_messages_or_input_runs_no_context_window_check(make_core: MakeCore) -> None:
    assert await _picked(_small_and_big(make_core)) == {"small", "big"}


@pytest.mark.asyncio
async def test_when_no_context_window_fits_select_raises_context_window_exceeded(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)], enable_pre_call_checks=True)
    with pytest.raises(litellm.ContextWindowExceededError) as error:
        await core.select("g", messages=LONG)
    message: Final = str(error.value)
    assert message.startswith(CONTEXT_WINDOW_PREFIX)
    assert re.fullmatch(
        r"Model=openai/cw-small-[0-9a-f]{8}, Max Input Tokens=5, Got=\d+", message.removeprefix(CONTEXT_WINDOW_PREFIX)
    )
    assert error.value.model == "g"
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_when_no_context_window_fits_async_completion_raises_without_calling_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)], enable_pre_call_checks=True)
    with pytest.raises(litellm.ContextWindowExceededError, match="Context Window exceeded for given call"):
        await core.acompletion(model="g", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []
    assert host.sleeps == []


def test_when_no_context_window_fits_sync_completion_raises_without_calling_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[windowed("g", "small", 5)], enable_pre_call_checks=True)
    with pytest.raises(litellm.ContextWindowExceededError):
        core.completion(model="g", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []


@pytest.mark.asyncio
async def test_the_context_window_error_lists_each_deployment_back_to_back(make_core: MakeCore) -> None:
    """Oddity: the per-deployment entries are concatenated with no separator between them."""
    core: Final = make_core(model_list=[windowed("g", "one", 5), windowed("g", "two", 6)], enable_pre_call_checks=True)
    with pytest.raises(litellm.ContextWindowExceededError) as error:
        await core.select("g", messages=LONG)
    assert re.fullmatch(
        r"Model=openai/cw-one-[0-9a-f]{8}, Max Input Tokens=5, Got=(\d+)Model=openai/cw-two-[0-9a-f]{8}, Max Input Tokens=6, Got=\1",
        str(error.value).removeprefix(CONTEXT_WINDOW_PREFIX),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("where", "expected"),
    [("model_info", "azure/gpt-4-mi-base"), ("litellm_params", "azure/gpt-4-lp-base")],
)
async def test_the_context_window_error_names_a_deployment_by_its_base_model(
    make_core: MakeCore, where: str, expected: str
) -> None:
    extra: Final[dict[str, object]] = (
        {"model_info": {"base_model": expected}} if where == "model_info" else {"base_model": expected}
    )
    core: Final = make_core(model_list=[windowed("g", f"base-{where}", 5, **extra)], enable_pre_call_checks=True)
    with pytest.raises(litellm.ContextWindowExceededError) as error:
        await core.select("g", messages=LONG)
    assert f"Model={expected}, Max Input Tokens=5, Got=" in str(error.value)


@pytest.mark.asyncio
async def test_when_no_context_window_fits_the_context_window_fallback_group_answers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[windowed("small-group", "small", 5), deployment("big-group", "big")],
        enable_pre_call_checks=True,
        context_window_fallbacks=[{"small-group": ["big-group"]}],
        num_retries=0,
    )
    await core.acompletion(model="small-group", messages=LONG)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["big"]


@pytest.mark.asyncio
async def test_a_deployment_without_a_known_context_window_is_kept_for_any_prompt(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[windowed("g", "small", 5), deployment("g", "unknown-window-model")], enable_pre_call_checks=True
    )
    assert await _picked(core, messages=LONG) == {"unknown-window-model"}


@pytest.mark.asyncio
async def test_the_context_window_comes_from_the_cost_map_when_the_deployment_declares_none(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "gpt4", model="openai/gpt-4"), deployment("g", "gpt4o", model="openai/gpt-4o")],
        enable_pre_call_checks=True,
    )
    assert await _picked(core, times=10, messages=[{"role": "user", "content": "word " * 10000}]) == {"gpt4o"}


@pytest.mark.asyncio
async def test_an_uncountable_prompt_skips_the_context_window_check(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[windowed("g", "tiny", 1)], enable_pre_call_checks=True)
    assert await _picked(core, times=2, messages=UNCOUNTABLE) == {"tiny"}


@pytest.mark.asyncio
async def test_responses_input_text_is_counted_against_the_context_window(make_core: MakeCore) -> None:
    assert await _picked(_small_and_big(make_core), times=10, input=LONG_TEXT) == {"big"}


@pytest.mark.asyncio
async def test_a_short_responses_input_may_pick_either_deployment(make_core: MakeCore) -> None:
    assert await _picked(_small_and_big(make_core, small=50), input="hi") == {"small", "big"}


@pytest.mark.asyncio
async def test_responses_instructions_are_counted_against_the_context_window(make_core: MakeCore) -> None:
    core: Final = _small_and_big(make_core, small=50)
    assert await _picked(core, request_kwargs={"instructions": LONG_TEXT}, input="hi") == {"big"}


@pytest.mark.asyncio
async def test_tool_definitions_are_counted_against_the_context_window(make_core: MakeCore) -> None:
    tool: Final = {
        "type": "function",
        "function": {"name": "f", "description": LONG_TEXT, "parameters": {"type": "object", "properties": {}}},
    }
    core: Final = _small_and_big(make_core, small=50)
    assert await _picked(core, request_kwargs={"tools": [tool]}, messages=SHORT) == {"big"}


@pytest.mark.asyncio
async def test_an_anthropic_system_block_is_counted_against_the_context_window(make_core: MakeCore) -> None:
    core: Final = _small_and_big(make_core, small=50)
    assert await _picked(core, request_kwargs={"system": LONG_TEXT}, messages=SHORT) == {"big"}


@pytest.mark.asyncio
async def test_a_deployment_context_window_is_written_into_the_process_cost_map(make_core: MakeCore) -> None:
    """Oddity: a deployment's declared ``max_input_tokens`` replaces the process-wide cost-map entry of its
    model, so it outlives the router and narrows another router's deployment of the same model."""
    first: Final = make_core(
        model_list=[deployment("g", "declares-five", model=NO_JSON_MODE_MODEL, model_info={"max_input_tokens": 5})]
    )
    first.retire()
    other: Final = python_core(
        CallbackHost(),
        model_list=[deployment("g", "declares-nothing", model=NO_JSON_MODE_MODEL)],
        enable_pre_call_checks=True,
    )
    with pytest.raises(litellm.ContextWindowExceededError, match="Max Input Tokens=5"):
        await other.select("g", messages=SHORT)
    other.retire()


# --- region ------------------------------------------------------------------------------


def _regional(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: constructor arguments
    return make_core(
        model_list=[
            deployment("g", "eu", region_name="eu"),
            deployment("g", "us", region_name="us"),
            deployment("g", "none"),
        ],
        **{"enable_pre_call_checks": True, **arguments},
    )


@pytest.mark.asyncio
async def test_allowed_model_region_keeps_only_deployments_in_that_region(make_core: MakeCore) -> None:
    core: Final = _regional(make_core)
    assert await _picked(core, request_kwargs={"allowed_model_region": "eu"}, messages=SHORT) == {"eu"}


@pytest.mark.asyncio
async def test_allowed_model_region_is_ignored_without_enable_pre_call_checks(make_core: MakeCore) -> None:
    core: Final = _regional(make_core, enable_pre_call_checks=False)
    assert await _picked(core, request_kwargs={"allowed_model_region": "eu"}, messages=SHORT) == {"eu", "us", "none"}


@pytest.mark.asyncio
async def test_allowed_model_region_is_ignored_without_messages_or_input(make_core: MakeCore) -> None:
    core: Final = _regional(make_core)
    assert await _picked(core, request_kwargs={"allowed_model_region": "eu"}) == {"eu", "us", "none"}


@pytest.mark.asyncio
async def test_no_deployment_in_the_allowed_region_raises_router_rate_limit_error(make_core: MakeCore) -> None:
    core: Final = _regional(make_core)
    with pytest.raises(RouterRateLimitError) as error:
        await core.select("g", {"allowed_model_region": "ap"}, messages=SHORT)
    assert str(error.value) == (
        "No deployments available for selected model, Try again in 5 seconds. Passed model=g. "
        "pre-call-checks=True, cooldown_list=[]"
    )


@pytest.mark.asyncio
async def test_no_deployment_in_the_allowed_region_never_calls_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _regional(make_core)
    with pytest.raises(RouterRateLimitError):
        await core.acompletion(model="g", messages=SHORT, allowed_model_region="ap")  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []


@pytest.mark.asyncio
async def test_async_completion_with_an_allowed_region_reaches_only_that_region(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _regional(make_core)
    for _ in range(6):
        await core.acompletion(model="g", messages=SHORT, allowed_model_region="us")  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) == {"us"}


# --- unsupported params ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_drop_params_off_a_deployment_that_lacks_response_format_is_skipped(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    assert await _picked(_json_mode_group(make_core), request_kwargs=dict(JSON_MODE), messages=SHORT) == {"new"}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [None, True])
async def test_unless_drop_params_is_exactly_false_unsupported_params_are_not_checked(
    make_core: MakeCore, host: ScriptedHost, drop_params: bool | None
) -> None:
    """Oddity: the unset process default (``None``) skips the check as ``drop_params=True`` does."""
    host.set_dynamic(drop_params=drop_params)
    assert await _picked(_json_mode_group(make_core), request_kwargs=dict(JSON_MODE), messages=SHORT) == {"old", "new"}


@pytest.mark.asyncio
async def test_unsupported_params_are_not_checked_without_enable_pre_call_checks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = _json_mode_group(make_core, enable_pre_call_checks=False)
    assert await _picked(core, request_kwargs=dict(JSON_MODE), messages=SHORT) == {"old", "new"}


@pytest.mark.asyncio
async def test_unsupported_params_are_not_checked_without_messages_or_input(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    assert await _picked(_json_mode_group(make_core), request_kwargs=dict(JSON_MODE)) == {"old", "new"}


@pytest.mark.asyncio
async def test_a_string_response_format_is_not_checked(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = _json_mode_group(make_core)
    assert await _picked(core, request_kwargs={"response_format": "b64_json"}, messages=SHORT) == {"old", "new"}


@pytest.mark.asyncio
async def test_when_no_deployment_supports_the_param_select_raises_unsupported_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL)], enable_pre_call_checks=True)
    with pytest.raises(litellm.UnsupportedParamsError) as error:
        await core.select("g", dict(JSON_MODE), messages=SHORT)
    assert str(error.value) == UNSUPPORTED_MESSAGE
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_when_no_deployment_supports_the_param_async_completion_raises_without_calling_or_retrying(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL)], enable_pre_call_checks=True, num_retries=2
    )
    with pytest.raises(litellm.UnsupportedParamsError, match="allowed_openai_params"):
        await core.acompletion(model="g", messages=SHORT, **JSON_MODE)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []
    assert host.sleeps == []


def test_when_no_deployment_supports_the_param_sync_completion_raises_without_calling(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL)], enable_pre_call_checks=True)
    with pytest.raises(litellm.UnsupportedParamsError):
        core.completion(model="g", messages=SHORT, **JSON_MODE)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []


REQUEST_ESCAPES: Final = [
    {"allowed_openai_params": ["response_format"]},
    {"drop_params": True},
    {"additional_drop_params": ["response_format"]},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", REQUEST_ESCAPES, ids=["allowlist", "drop", "additional-drop"])
async def test_a_request_that_allows_or_drops_the_param_keeps_the_only_deployment(
    make_core: MakeCore, host: ScriptedHost, escape: dict[str, object]
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL)], enable_pre_call_checks=True)
    assert await _picked(core, times=2, request_kwargs={**JSON_MODE, **escape}, messages=SHORT) == {"old"}


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", REQUEST_ESCAPES, ids=["allowlist", "drop", "additional-drop"])
async def test_a_deployment_that_takes_the_param_is_preferred_over_dropping_it(
    make_core: MakeCore, host: ScriptedHost, escape: dict[str, object]
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = _json_mode_group(make_core)
    assert await _picked(core, request_kwargs={**JSON_MODE, **escape}, messages=SHORT) == {"new"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment_escape",
    [{"drop_params": True}, {"allowed_openai_params": ["response_format"]}],
    ids=["drop", "allowlist"],
)
async def test_a_deployment_that_allows_or_drops_the_param_is_kept(
    make_core: MakeCore, host: ScriptedHost, deployment_escape: dict[str, object]
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL, **deployment_escape)], enable_pre_call_checks=True
    )
    assert await _picked(core, times=2, request_kwargs=dict(JSON_MODE), messages=SHORT) == {"old"}


@pytest.mark.asyncio
async def test_a_request_drop_params_false_overrides_the_deployment_drop_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[deployment("g", "old", model=NO_JSON_MODE_MODEL, drop_params=True)], enable_pre_call_checks=True
    )
    with pytest.raises(litellm.UnsupportedParamsError):
        await core.select("g", {**JSON_MODE, "drop_params": False}, messages=SHORT)


@pytest.mark.asyncio
async def test_a_context_window_error_wins_over_an_unsupported_param(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[deployment("g", "small-no-json", model=NO_JSON_MODE_MODEL, model_info={"max_input_tokens": 5})],
        enable_pre_call_checks=True,
    )
    with pytest.raises(litellm.ContextWindowExceededError):
        await core.select("g", dict(JSON_MODE), messages=LONG)
    core.retire()


@pytest.mark.asyncio
async def test_a_capable_deployment_outside_the_region_gives_no_deployments_rather_than_a_400(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[
            deployment("g", "old", model=NO_JSON_MODE_MODEL, region_name="eu"),
            deployment("g", "new", model=JSON_MODE_MODEL, region_name="us"),
        ],
        enable_pre_call_checks=True,
    )
    with pytest.raises(RouterRateLimitError, match=r"^No deployments available for selected model"):
        await core.select("g", {**JSON_MODE, "allowed_model_region": "eu"}, messages=SHORT)


@pytest.mark.asyncio
async def test_dropping_the_param_keeps_the_in_region_deployment_that_lacks_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = make_core(
        model_list=[
            deployment("g", "old", model=NO_JSON_MODE_MODEL, region_name="eu"),
            deployment("g", "new", model=JSON_MODE_MODEL, region_name="us"),
        ],
        enable_pre_call_checks=True,
    )
    request: Final = {**JSON_MODE, "allowed_model_region": "eu", "drop_params": True}
    assert await _picked(core, times=5, request_kwargs=request, messages=SHORT) == {"old"}


@pytest.mark.asyncio
async def test_a_capable_deployment_in_cooldown_gives_no_deployments_rather_than_a_400(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(drop_params=False)
    core: Final = _json_mode_group(make_core)
    core.health().cooldowns.add_deployment_to_cooldown(
        "new", original_exception=Exception("down"), exception_status=429, cooldown_time=60
    )
    with pytest.raises(RouterRateLimitError) as error:
        await core.select("g", dict(JSON_MODE), messages=SHORT)
    assert str(error.value).startswith(
        "No deployments available for selected model, Try again in 60 seconds. Passed model=g. pre-call-checks=True"
    )


# --- images --------------------------------------------------------------------------------


def _vision_group(make_core: MakeCore) -> RouterCore:
    return make_core(
        model_list=[
            deployment("g", "text", model_info={"supports_vision": False}),
            deployment("g", "vision", model_info={"supports_vision": True}),
        ]
    )


@pytest.mark.asyncio
async def test_an_image_request_skips_a_declared_text_only_deployment_even_without_pre_call_checks(
    make_core: MakeCore,
) -> None:
    assert await _picked(_vision_group(make_core), messages=IMAGE) == {"vision"}


@pytest.mark.asyncio
async def test_an_image_in_the_request_messages_counts_as_an_image_request(make_core: MakeCore) -> None:
    assert await _picked(_vision_group(make_core), request_kwargs={"messages": IMAGE}) == {"vision"}


@pytest.mark.asyncio
async def test_a_text_request_may_pick_a_text_only_deployment(make_core: MakeCore) -> None:
    assert await _picked(_vision_group(make_core), messages=SHORT) == {"text", "vision"}


@pytest.mark.asyncio
async def test_an_image_request_keeps_every_deployment_when_all_are_text_only(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "text", model_info={"supports_vision": False}),
            deployment("g", "text2", model_info={"supports_vision": False}),
        ]
    )
    assert await _picked(core, messages=IMAGE) == {"text", "text2"}


@pytest.mark.asyncio
async def test_async_completion_with_an_image_reaches_the_vision_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _vision_group(make_core)
    for _ in range(6):
        await core.acompletion(model="g", messages=IMAGE)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) == {"vision"}


# --- budgets -------------------------------------------------------------------------------

PROVIDER_BUDGET: Final = {"openai": {"budget_limit": 1, "time_period": "1d"}}


@pytest.mark.asyncio
async def test_a_provider_over_budget_leaves_no_deployment_and_select_raises_value_error(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], provider_budget_config=PROVIDER_BUDGET)
    await core.health().store.async_set_cache("provider_spend:openai:1d", 5.0)
    with pytest.raises(ValueError, match=r"^No deployments available - crossed budget: ") as error:
        await core.select("g")
    assert (
        str(error.value)
        == "No deployments available - crossed budget: Exceeded budget for provider openai: 5.0 >= 1.0\n"
    )
    core.retire()


@pytest.mark.asyncio
async def test_a_provider_over_budget_raises_from_async_completion_without_calling_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], provider_budget_config=PROVIDER_BUDGET)
    await core.health().store.async_set_cache("provider_spend:openai:1d", 5.0)
    with pytest.raises(ValueError, match=r"^No deployments available - crossed budget"):
        await core.acompletion(model="g", messages=SHORT)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []
    core.retire()


@pytest.mark.asyncio
async def test_a_provider_under_budget_is_picked(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], provider_budget_config=PROVIDER_BUDGET)
    await core.health().store.async_set_cache("provider_spend:openai:1d", 0.5)
    assert await _picked(core, times=2) == {"a"}
    core.retire()


@pytest.mark.asyncio
async def test_a_deployment_over_its_own_budget_is_skipped(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", max_budget=1, budget_duration="1d"), deployment("g", "b")])
    await core.health().store.async_set_cache("deployment_spend:a:1d", 5.0)
    assert await _picked(core, times=10) == {"b"}
    core.retire()

"""Cross-group fallbacks of a router core: where the chain comes from, how it is walked, and what is raised."""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.utils import Usage
from tests.unit.test_router.core.construction_precall_helpers import windowed
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    agentic_followup_failure,
    bad_request,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
    timeout,
)
from tests.unit.test_router.core.streaming_helpers import request_logging

MakeCore = Callable[..., RouterCore]

MESSAGES = [{"role": "user", "content": "hi"}]
GROUPS_ABC = [deployment("a", "a"), deployment("b", "b"), deployment("c", "c")]
GROUPS_ABCD = [*GROUPS_ABC, deployment("d", "d")]
CHAIN_ABCD = [{"a": ["b"]}, {"b": ["c"]}, {"c": ["d"]}]
TIERED_A = [deployment("a", "a1", order=1), deployment("a", "a2", order=2), deployment("b", "b")]


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(**{"model_list": GROUPS_ABC, "num_retries": 0, **arguments})


async def complete(core: RouterCore, model: str = "a", **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.acompletion(model=model, messages=MESSAGES, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def headers(response: object) -> Mapping[str, object]:
    return getattr(response, "_hidden_params")["additional_headers"]


def show_router_debug(host: ScriptedHost) -> None:
    host.set_dynamic(expose_router_debug_in_errors=True)


# --- where the fallback chain comes from -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "process", "request_kwargs", "expected"),
    [
        ({"fallbacks": [{"a": ["b"]}]}, None, {}, ["a", "b"]),
        ({}, [{"a": ["c"]}], {}, ["a", "c"]),
        ({}, None, {"fallbacks": [{"a": ["b"]}]}, ["a", "b"]),
        ({}, None, {"fallbacks": ["c"]}, ["a", "c"]),
        ({"fallbacks": [{"a": ["b"]}]}, [{"a": ["c"]}], {}, ["a", "b"]),
        ({"fallbacks": [{"a": ["b"]}]}, None, {"fallbacks": [{"a": ["c"]}]}, ["a", "c"]),
    ],
    ids=[
        "constructor",
        "process-setting",
        "request-kwarg",
        "request-plain-list",
        "constructor-over-process-setting",
        "request-over-constructor",
    ],
)
async def test_a_failed_group_falls_back_to_the_chain_from_the_winning_source(
    make_core: MakeCore,
    host: ScriptedHost,
    constructor: dict[str, object],
    process: list[object] | None,
    request_kwargs: dict[str, object],
    expected: list[str],
) -> None:
    if process is not None:
        host.set_dynamic(fallbacks=process)
    core = build(make_core, **constructor)
    host.answer("openai/a", rate_limit())

    response = await complete(core, **request_kwargs)

    assert host.deployments_called("acompletion") == expected
    assert headers(response)["x-litellm-model-group"] == expected[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("fallbacks", rate_limit),
        ("context_window_fallbacks", context_window),
        ("content_policy_fallbacks", content_policy),
    ],
    ids=["fallbacks", "context-window", "content-policy"],
)
async def test_request_fallback_string_is_ignored_and_router_chain_is_used(
    make_core: MakeCore, host: ScriptedHost, field: str, error: Callable[[], Exception]
) -> None:
    core = build(make_core, **{field: [{"a": ["b"]}]})
    host.answer("openai/a", error())

    response = await complete(core, **{field: "unconfigured-scalar-model"})

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert headers(response)["x-litellm-model-group"] == "b"


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("fallbacks", rate_limit),
        ("context_window_fallbacks", context_window),
        ("content_policy_fallbacks", content_policy),
    ],
    ids=["fallbacks", "context-window", "content-policy"],
)
def test_sync_request_fallback_string_is_ignored_and_router_chain_is_used(
    make_core: MakeCore, host: ScriptedHost, field: str, error: Callable[[], Exception]
) -> None:
    core = build(make_core, **{field: [{"a": ["b"]}]})
    host.answer("openai/a", error())

    response: Final = core.completion(  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
        model="a", messages=MESSAGES, **{field: "unconfigured-scalar-model"}
    )

    assert host.deployments_called("completion") == ["a", "b"]
    assert headers(response)["x-litellm-model-group"] == "b"


@pytest.mark.asyncio
async def test_an_empty_request_fallback_list_overrides_the_constructor_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, fallbacks=[])

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_null_request_fallback_list_overrides_the_constructor_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, fallbacks=None)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["assign", "settings"])
async def test_fallbacks_applied_by_changeset_take_effect(make_core: MakeCore, host: ScriptedHost, field: str) -> None:
    core = build(make_core)
    core.apply(ChangeSet(**{field: {"fallbacks": [{"a": ["c"]}]}}))  # pyright: ignore[reportArgumentType]  # assign or settings
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


# --- resolving the group's chain ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_generic_star_chain_catches_a_group_with_no_entry(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"b": ["a"]}, {"*": ["c"]}])
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallbacks",
    [[{"*": ["c"]}, {"a": ["b"]}], [{"a": ["b"]}, {"*": ["c"]}]],
    ids=["star-first", "star-last"],
)
async def test_a_group_specific_chain_wins_over_the_generic_star_chain(
    make_core: MakeCore, host: ScriptedHost, fallbacks: list[object]
) -> None:
    core = build(make_core, fallbacks=fallbacks)
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group", "fallback_key"),
    [("openai/gpt-x", "gpt-x"), ("gpt-4o", "openai/gpt-4o")],
    ids=["provider-stripped-key", "provider-prefixed-key"],
)
async def test_a_chain_keyed_with_or_without_the_provider_prefix_matches_the_group(
    make_core: MakeCore, host: ScriptedHost, group: str, fallback_key: str
) -> None:
    core = build(
        make_core, model_list=[deployment(group, "p"), deployment("b", "b")], fallbacks=[{fallback_key: ["b"]}]
    )
    host.answer("openai/p", rate_limit())

    await complete(core, model=group)

    assert host.deployments_called("acompletion") == ["p", "b"]


@pytest.mark.asyncio
async def test_a_prefixed_key_does_not_match_a_group_whose_provider_cannot_be_inferred(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(
        make_core, model_list=[deployment("mygroup", "p"), deployment("b", "b")], fallbacks=[{"openai/mygroup": ["b"]}]
    )
    host.answer("openai/p", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, model="mygroup")

    assert host.deployments_called("acompletion") == ["p"]


@pytest.mark.asyncio
async def test_a_dict_target_names_its_group_under_model(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": [{"model": "c"}]}])
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


# --- walking the chain ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "expected"),
    [(["a"], ["a", "b"]), (["a", "b"], ["a", "b", "c"])],
    ids=["first-target-answers", "second-target-answers"],
)
async def test_targets_are_tried_in_listed_order_until_one_answers(
    make_core: MakeCore, host: ScriptedHost, failing: list[str], expected: list[str]
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}])
    for group in failing:
        host.answer(f"openai/{group}", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == expected
    assert headers(response)["x-litellm-model-group"] == expected[-1]


@pytest.mark.asyncio
async def test_a_failed_fallback_group_follows_its_own_chain(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}])
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallbacks", "expected"),
    [
        ([{"a": ["b"]}, {"b": ["a", "c"]}, {"c": ["a", "b"]}], ["a", "b", "c"]),
        ([{"a": ["a", "b"]}], ["a", "b"]),
        ([{"a": ["b", "b", "c"]}], ["a", "b", "c"]),
    ],
    ids=["cyclic-graph", "group-lists-itself", "target-listed-twice"],
)
async def test_no_group_is_tried_twice_in_one_request(
    make_core: MakeCore, host: ScriptedHost, fallbacks: list[object], expected: list[str]
) -> None:
    core = build(make_core, fallbacks=fallbacks)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constructor", "process", "expected"),
    [
        ({"max_fallbacks": 0}, {}, ["a"]),
        ({"max_fallbacks": 1}, {}, ["a", "b"]),
        ({"max_fallbacks": 2}, {}, ["a", "b", "c"]),
        ({}, {"max_fallbacks": 2}, ["a", "b", "c"]),
        ({}, {"router_max_fallbacks": 2}, ["a", "b", "c"]),
        ({}, {}, ["a", "b", "c", "d"]),
        ({"max_fallbacks": 3}, {"max_fallbacks": 1}, ["a", "b", "c", "d"]),
        ({}, {"max_fallbacks": 1, "router_max_fallbacks": 3}, ["a", "b"]),
    ],
    ids=[
        "constructor-0",
        "constructor-1",
        "constructor-2",
        "process-max_fallbacks-2",
        "process-router_max_fallbacks-2",
        "default",
        "constructor-over-process",
        "process-max_fallbacks-over-router_max_fallbacks",
    ],
)
async def test_max_fallbacks_bounds_the_depth_of_a_chained_walk(
    make_core: MakeCore,
    host: ScriptedHost,
    constructor: dict[str, object],
    process: dict[str, object],
    expected: list[str],
) -> None:
    if process:
        host.set_dynamic(**process)
    core = build(make_core, model_list=GROUPS_ABCD, fallbacks=CHAIN_ABCD, **constructor)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
async def test_max_fallbacks_applied_by_changeset_bounds_the_walk(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=GROUPS_ABCD, fallbacks=CHAIN_ABCD)
    core.apply(ChangeSet(assign={"max_fallbacks": 1}))
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, ["a", "b"]),
        (False, ["a"]),
        (1.0, ["a", "b"]),
        ("1", ["a", "b"]),
        ("1.0", ["a", "b"]),
        (0, ["a"]),
    ],
    ids=["true", "false", "integral-float", "integer-string", "decimal-string", "zero"],
)
async def test_request_max_fallbacks_uses_bounded_integer_coercion(
    make_core: MakeCore, host: ScriptedHost, value: object, expected: list[str]
) -> None:
    core = build(make_core, model_list=GROUPS_ABCD, fallbacks=CHAIN_ABCD)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, max_fallbacks=value)

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallbacks", 3),
        ("fallbacks", ["c", 1]),
        ("context_window_fallbacks", [None]),
        ("content_policy_fallbacks", [True]),
        ("max_fallbacks", "not-an-integer"),
        ("max_fallbacks", -1),
        ("max_fallbacks", 2**32),
        ("max_fallbacks", 1.5),
        ("max_fallbacks", float("nan")),
        ("max_fallbacks", float("inf")),
    ],
    ids=[
        "fallbacks-not-a-list",
        "fallbacks-invalid-entry",
        "context-window-invalid-entry",
        "content-policy-invalid-entry",
        "max-fallbacks-invalid-string",
        "max-fallbacks-negative",
        "max-fallbacks-too-large",
        "max-fallbacks-fractional",
        "max-fallbacks-nan",
        "max-fallbacks-infinite",
    ],
)
async def test_invalid_request_fallback_fields_raise_bad_request(
    make_core: MakeCore, host: ScriptedHost, field: str, value: object
) -> None:
    core = build(make_core)

    with pytest.raises(litellm.BadRequestError) as raised:
        await complete(core, **{field: value})

    assert raised.value.status_code == 400
    assert host.deployments_called("acompletion") == []


@pytest.mark.asyncio
async def test_a_fallback_group_with_no_deployments_is_passed_over(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["missing", "c"]}])
    host.answer("openai/a", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"


@pytest.mark.asyncio
async def test_a_chain_of_only_empty_groups_raises_the_primary_error(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["missing"]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_fallback_group_runs_after_the_primary_group_spends_its_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}], num_retries=2)
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "a", "a", "b"]


# --- error-specific chains ---------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "field"),
    [(context_window, "context_window_fallbacks"), (content_policy, "content_policy_fallbacks")],
    ids=["context-window", "content-policy"],
)
@pytest.mark.parametrize("source", ["constructor", "process-setting", "request-kwarg"])
async def test_an_error_specific_chain_wins_over_the_general_chain(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception], field: str, source: str
) -> None:
    chain = [{"a": ["c"]}]
    if source == "process-setting":
        host.set_dynamic(**{field: chain})
    core = build(make_core, fallbacks=[{"a": ["b"]}], **({field: chain} if source == "constructor" else {}))
    host.answer("openai/a", error())

    response = await complete(core, **({field: chain} if source == "request-kwarg" else {}))

    assert host.deployments_called("acompletion") == ["a", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"


@pytest.mark.asyncio
async def test_an_unmatched_content_policy_chain_does_not_try_the_general_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], content_policy_fallbacks=[{"other": ["c"]}])
    host.answer("openai/a", content_policy())

    with pytest.raises(litellm.ContentPolicyViolationError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [context_window, content_policy], ids=["context-window", "content-policy"])
async def test_without_an_error_specific_chain_the_general_chain_applies(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception]
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", error())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_a_context_window_fallback_that_fails_otherwise_follows_the_general_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, context_window_fallbacks=[{"a": ["b"]}], fallbacks=[{"b": ["c"]}])
    host.answer("openai/a", context_window())
    host.answer("openai/b", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]


# --- default fallbacks -------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["constructor", "process-setting"])
async def test_default_fallbacks_catch_any_group(make_core: MakeCore, host: ScriptedHost, source: str) -> None:
    if source == "process-setting":
        host.set_dynamic(default_fallbacks=["c"])
    core = build(make_core, **({"default_fallbacks": ["c"]} if source == "constructor" else {}))
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


@pytest.mark.asyncio
async def test_default_fallbacks_join_the_configured_chains_as_the_star_entry(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"b": ["a"]}], default_fallbacks=["c"])
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert core.settings().fallbacks == [{"b": ["a"]}, {"*": ["c"]}]
    assert host.deployments_called("acompletion") == ["a", "c"]


def test_default_fallbacks_given_as_a_string_are_refused(make_core: MakeCore) -> None:
    with pytest.raises(TypeError, match=r"default_fallbacks must be a list of model names, got 'c'"):
        build(make_core, default_fallbacks="c")


# --- order tiers inside one group ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "expected", "model_group"),
    [
        (["a1"], ["a1", "a2"], "a"),
        (["a1", "a2"], ["a1", "a2", "b"], "b"),
    ],
    ids=["next-tier-answers", "external-fallback-after-every-tier"],
)
async def test_order_tiers_are_tried_before_the_external_fallbacks(
    make_core: MakeCore, host: ScriptedHost, failing: list[str], expected: list[str], model_group: str
) -> None:
    core = build(make_core, model_list=TIERED_A, fallbacks=[{"a": ["b"]}])
    for deployment_id in failing:
        host.answer(f"openai/{deployment_id}", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == expected
    assert headers(response)["x-litellm-model-group"] == model_group


@pytest.mark.asyncio
async def test_order_tiers_are_walked_without_any_configured_fallbacks(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=[*TIERED_A, deployment("a", "a3", order=3)])
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a1", "a2", "a3"]


@pytest.mark.asyncio
async def test_a_context_window_error_skips_the_order_tiers_for_its_own_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=TIERED_A, context_window_fallbacks=[{"a": ["b"]}])
    host.answer("openai/a1", context_window())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a1", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window_key", "tiered", "client_side", "expected"),
    [
        (None, True, False, "a2"),
        ("other", True, False, "a2"),
        ("other", False, False, "b"),
        ("other", False, True, "b"),
        ("a", True, False, "c"),
        ("*", True, False, "c"),
        ("a", True, True, "c"),
        ("a", False, True, "c"),
    ],
    ids=[
        "no-list-takes-the-order-tier",
        "unmatched-list-takes-the-order-tier",
        "unmatched-list-falls-through-to-the-general-chain",
        "unmatched-list-leaves-the-client-side-list",
        "matched-list-beats-the-order-tier",
        "matched-star-beats-the-order-tier",
        "matched-list-beats-the-order-tier-and-the-client-side-list",
        "matched-list-beats-the-client-side-list",
    ],
)
async def test_a_context_window_error_takes_its_own_chain_only_when_the_group_matches(
    make_core: MakeCore,
    host: ScriptedHost,
    window_key: str | None,
    tiered: bool,
    client_side: bool,
    expected: str,
) -> None:
    core: Final = build(
        make_core,
        model_list=[
            deployment("a", "a1", order=1),
            deployment("a" if tiered else "unused", "a2", order=2),
            deployment("b", "b"),
            deployment("c", "c"),
        ],
        fallbacks=[{"a": ["b"]}],
        context_window_fallbacks=None if window_key is None else [{window_key: ["c"]}],
    )
    host.answer("openai/a1", context_window())

    await complete(core, **({"fallbacks": ["b"]} if client_side else {}))

    assert host.deployments_called("acompletion") == ["a1", expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "raised"),
    [(context_window, litellm.ContextWindowExceededError), (server_error, litellm.Timeout)],
    ids=["context-window-keeps-the-original-error", "any-other-error-raises-the-last-hop"],
)
async def test_a_failed_order_chain_raises_by_the_error_that_started_it(
    make_core: MakeCore, host: ScriptedHost, first: Callable[[], Exception], raised: type[Exception]
) -> None:
    core: Final = build(
        make_core, model_list=TIERED_A, fallbacks=[{"a": ["b"]}], context_window_fallbacks=[{"other": ["c"]}]
    )
    host.answer("openai/a1", first())
    host.answer("openai/a2", rate_limit())
    host.answer("openai/b", timeout())

    with pytest.raises(raised):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a1", "a2", "b"]


@pytest.mark.asyncio
async def test_a_failed_order_chain_appends_no_outcome_message(make_core: MakeCore, host: ScriptedHost) -> None:
    show_router_debug(host)
    core: Final = build(make_core, model_list=TIERED_A)
    host.answer("openai/a1", context_window())
    host.answer("openai/a2", rate_limit())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    assert "LiteLLM: model group" not in str(raised.value)


@pytest.mark.asyncio
async def test_an_unmatched_context_window_list_reports_the_general_chain_it_fell_through_to(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    show_router_debug(host)
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], context_window_fallbacks=[{"other": ["c"]}])
    host.answer("openai/a", context_window())
    host.answer("openai/b", rate_limit())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    assert "No fallback was attempted" not in str(raised.value)
    assert "context_window_fallbacks=None" not in str(raised.value)
    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_a_pre_call_context_window_error_is_kept_when_its_order_chain_fails(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(
        make_core,
        model_list=[windowed("a", "a1", 1, order=1), windowed("a", "a2", 1, order=2), deployment("b", "b")],
        enable_pre_call_checks=True,
        fallbacks=[{"a": ["b"]}],
        context_window_fallbacks=[{"other": ["c"]}],
    )
    host.answer("openai/b", rate_limit())

    with pytest.raises(litellm.ContextWindowExceededError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["b"]


@pytest.mark.asyncio
async def test_weighted_failover_skips_a_context_window_error_with_an_unmatched_list(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("a", "a1"), deployment("a", "a2"), deployment("b", "b")],
        enable_weighted_failover=True,
        fallbacks=[{"a": ["b"]}],
        context_window_fallbacks=[{"other": ["c"]}],
    )
    host.answer("openai/a1", context_window())
    host.answer("openai/a2", context_window())

    await complete(core)

    called: Final = host.deployments_called("acompletion")
    assert len(called) == 2
    assert called[-1] == "b"


# --- a failed agentic follow-up ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_agentic_followup_still_falls_back(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], num_retries=3)
    host.answer("openai/a", lambda **_: agentic_followup_failure(server_error()))

    response: Final = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert headers(response)["x-litellm-model-group"] == "b"


@pytest.mark.asyncio
async def test_a_fallback_hop_carries_the_superseded_attempt_usage(make_core: MakeCore, host: ScriptedHost) -> None:
    logging_obj: Final = request_logging()
    usage: Final = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    def failed_after_usage(**_: object) -> Exception:
        logging_obj.model_call_details["combined_usage_object"] = usage
        return server_error()

    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", failed_after_usage)

    await complete(core, litellm_logging_obj=logging_obj)

    assert logging_obj.model_call_details["failed_attempts_usage"] == ((usage, 0.0),)


# --- when no fallback runs -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disable_fallbacks_on_the_request_raises_the_primary_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, disable_fallbacks=True)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        GuardrailRaisedException(guardrail_name="g", message="no"),
        BlockedPiiEntityError(entity_type="EMAIL", guardrail_name="g"),
    ],
    ids=["guardrail-raised", "blocked-pii"],
)
async def test_a_guardrail_intervention_never_falls_back(
    make_core: MakeCore, host: ScriptedHost, error: Exception
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", error)

    with pytest.raises(type(error)) as raised:
        await complete(core)

    assert raised.value is host.raised[-1]
    assert host.deployments_called("acompletion") == ["a"]


def unsupported_params() -> litellm.UnsupportedParamsError:
    return litellm.UnsupportedParamsError(message="nope", llm_provider="openai", model="a")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [bad_request, unsupported_params], ids=["bad-request", "unsupported-params"])
async def test_a_bad_request_does_not_fall_back_when_fallback_on_unsupported_params_is_off(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception]
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}], fallback_on_unsupported_params=False)
    host.answer("openai/a", error())

    with pytest.raises(type(error())):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "fallback_on_unsupported_params"),
    [
        (bad_request, True),
        (unsupported_params, True),
        (context_window, False),
        (content_policy, False),
        (rate_limit, False),
        (server_error, False),
    ],
    ids=[
        "bad-request-flag-on",
        "unsupported-params-flag-on",
        "context-window-flag-off",
        "content-policy-flag-off",
        "rate-limit-flag-off",
        "server-error-flag-off",
    ],
)
async def test_errors_outside_the_unsupported_params_rule_fall_back(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception], fallback_on_unsupported_params: bool
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}], fallback_on_unsupported_params=fallback_on_unsupported_params)
    host.answer("openai/a", error())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]


# --- the error when every group fails --------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_every_fallback_fails_the_primary_groups_error_is_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}], max_fallbacks=1)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", timeout())
    host.answer("openai/c", server_error())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert str(raised.value) == "litellm.RateLimitError: slow down"
    assert host.deployments_called("acompletion") == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_without_router_debug_the_error_message_is_not_extended(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert str(raised.value) == "litellm.RateLimitError: slow down"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "suffix"),
    [
        ({}, "\n\nLiteLLM: model group 'a' failed with the error above. No fallback was attempted."),
        (
            {"fallbacks": [{"a": ["b", "c"]}]},
            "\n\nLiteLLM: model group 'a' failed with the error above. "
            "Fallback to b, c also failed: litellm.RateLimitError: slow down",
        ),
        (
            {"fallbacks": [{"a": ["b"]}, {"b": ["c"]}]},
            "\n\nLiteLLM: model group 'a' failed with the error above. "
            "Fallback to b also failed: litellm.RateLimitError: slow down",
        ),
        (
            {"fallbacks": [{"b": ["c"]}]},
            "\n\nLiteLLM: model group 'a' failed with the error above and no fallback model group was found for it, "
            "so the request was not retried on another model. Fallbacks are configured for: b. "
            "Add a fallbacks entry for that model group (Router fallbacks or proxy router_settings.fallbacks) "
            "to retry on another model.",
        ),
    ],
    ids=["no-fallbacks", "fallback-list-failed", "chained-fallback-failed", "no-chain-for-group"],
)
async def test_router_debug_explains_the_fallback_outcome_once(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, object], suffix: str
) -> None:
    core = build(make_core, **arguments)
    show_router_debug(host)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert str(raised.value) == f"litellm.RateLimitError: slow down{suffix}"


@pytest.mark.asyncio
async def test_router_debug_names_the_first_fallback_groups_error(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", server_error())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert str(raised.value).endswith(
        "LiteLLM: model group 'a' failed with the error above. Fallback to b also failed: litellm.InternalServerError: boom"
    )


@pytest.mark.asyncio
async def test_router_debug_reports_a_fallback_group_with_no_deployments(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["missing"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert (
        "LiteLLM: model group 'a' failed with the error above. Fallback to missing also failed: "
        "litellm.BadRequestError: You passed in model=missing. There are no healthy deployments for this model"
    ) in str(raised.value)


@pytest.mark.asyncio
async def test_router_debug_on_an_unmatched_context_window_chain_says_no_fallback_was_attempted(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, context_window_fallbacks=[{"other": ["c"]}])
    show_router_debug(host)
    host.answer("openai/a", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    assert str(raised.value).endswith(
        "\n\nLiteLLM: model group 'a' failed with the error above. No fallback was attempted."
    )


@pytest.mark.asyncio
async def test_router_debug_names_only_the_chain_that_ran_not_the_whole_config(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    unrelated: Final = [{f"unrelated-{index}": ["x"]} for index in range(50)]
    core = build(make_core, fallbacks=[{"a": ["b"]}, *unrelated])
    show_router_debug(host)
    host.answer("openai/a", context_window())
    host.answer("openai/b", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    message: Final = str(raised.value)
    assert "fallbacks: 'a' -> ['b']" in message
    assert "fallbacks: 'b' -> " not in message
    for entry in unrelated:
        assert next(iter(entry)) not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "downstream_error",
    (
        litellm.ContextWindowExceededError(
            message="Context window exceeded\n  openai/b: max_input_tokens=50, got 88 input tokens",
            llm_provider="openai",
            model="b",
        ),
        bad_request(message="Request rejected\ninvalid field: tools[0].input_schema"),
        litellm.ContentPolicyViolationError(
            message="Request blocked\npolicy category: unsafe input", llm_provider="openai", model="b"
        ),
    ),
    ids=("context-window", "invalid-field", "content-policy"),
)
async def test_router_debug_preserves_multiline_downstream_failure_without_nested_hints(
    make_core: MakeCore, host: ScriptedHost, downstream_error: litellm.BadRequestError
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", downstream_error)

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    message: Final = str(raised.value)
    detail: Final = message.split("Fallback to b also failed: ", 1)[1]
    assert host.deployments_called("acompletion") == ["a", "b"]
    for line in downstream_error.message.splitlines():
        assert line.strip() in detail
    assert "\n" not in detail
    assert "context_window_fallbacks" not in detail
    assert "content_policy_fallbacks" not in detail
    assert message.count("LiteLLM: model group") == 1


@pytest.mark.asyncio
async def test_router_debug_states_how_many_groups_the_config_covers_when_the_chain_does_not_match(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"b": ["c"]}, {"c": ["a"]}])
    show_router_debug(host)
    host.answer("openai/a", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    assert "fallbacks has no chain for 'a' (2 other model groups are configured)" in str(raised.value)


@pytest.mark.asyncio
async def test_router_debug_reports_a_failed_hop_as_one_line_without_repeating_its_own_hint(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}, {"c": ["a"]}])
    show_router_debug(host)
    host.answer("openai/a", context_window())
    host.answer("openai/b", context_window())
    host.answer("openai/c", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    message: Final = str(raised.value)
    assert message.count("LiteLLM: model group") == 1
    assert message.count("context_window_fallbacks is not set") == 1


@pytest.mark.asyncio
async def test_router_debug_on_content_policy_without_any_chain_hints_at_content_policy_fallbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", content_policy())

    with pytest.raises(litellm.ContentPolicyViolationError) as raised:
        await complete(core)

    message = str(raised.value)
    assert "model=a. content_policy_fallbacks is not set." in message
    assert message.endswith("LiteLLM: model group 'a' failed with the error above. No fallback was attempted.")


# --- response headers ------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "attempted", "model_group"),
    [([], 0, "a"), (["a"], 1, "b")],
    ids=["primary-answers", "one-fallback"],
)
async def test_success_headers_name_the_answering_group_and_the_fallbacks_attempted(
    make_core: MakeCore, host: ScriptedHost, failing: list[str], attempted: int, model_group: str
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    for group in failing:
        host.answer(f"openai/{group}", rate_limit())

    response = await complete(core)

    assert headers(response)["x-litellm-attempted-fallbacks"] == attempted
    assert headers(response)["x-litellm-model-group"] == model_group


@pytest.mark.asyncio
async def test_each_attempt_carries_its_own_model_group_to_the_host(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    await core.call("acompletion", {"model": "a", "messages": MESSAGES})

    assert [call.model_group for call in host.calls_to("acompletion")] == ["a", "b"]
    assert [call.model for call in host.calls_to("acompletion")] == ["openai/a", "openai/b"]


@pytest.mark.asyncio
async def test_embedding_falls_back_like_completion(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=[deployment("e", "e"), deployment("f", "f")], fallbacks=[{"e": ["f"]}])
    host.answer("openai/e", rate_limit())

    response = await core.aembedding(model="e", input=["x"])  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method

    assert host.deployments_called("aembedding") == ["e", "f"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


def test_sync_completion_falls_back_like_async(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    response = core.completion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method

    assert host.deployments_called("completion") == ["a", "b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1

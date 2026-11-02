"""
Characterization of what a router core's fallbacks leave behind (non-streaming): response headers
and hidden params, the metadata each attempt carries, the error text when fallbacks give up,
fallback access and budget checks, fallback events, content-filter responses and weighted failover.
"""

from collections.abc import Mapping

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from tests.unit.test_router.core.fallbacks_helpers import (
    GROUP_A_OF_THREE,
    MakeCore,
    build,
    complete,
    complete_sync,
    fails_once_then_answers,
    headers,
    hidden,
    metadata_of,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    completion_response,
    context_window,
    deployment,
    rate_limit,
    server_error,
    timeout,
)

ERROR_429 = '{"message": "litellm.RateLimitError: slow down", "type": "throttling_error", "param": null, "code": "429"}'
ERROR_500 = (
    '{"message": "litellm.InternalServerError: boom", "type": "internal_server_error", "param": null, "code": "500"}'
)

# --- headers and hidden params ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_two_hop_chain_reports_one_attempted_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}])
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1
    assert headers(response)["x-litellm-model-group"] == "c"


@pytest.mark.asyncio
async def test_the_answering_fallback_deployment_is_the_responses_model_id(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    response = await complete(core)

    assert hidden(response)["model_id"] == "b"


@pytest.mark.asyncio
async def test_success_headers_after_a_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    response = await complete(core)

    assert headers(response) == {
        "x-litellm-model-group": "b",
        "x-litellm-attempted-retries": 0,
        "x-litellm-attempted-fallbacks": 1,
    }


@pytest.mark.asyncio
async def test_include_fallback_errors_lists_the_primary_error_in_a_header(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    response = await complete(core, include_fallback_errors=True)

    assert headers(response)["x-litellm-fallback-errors"] == f"[{ERROR_429}]"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallbacks",
    [[{"a": ["b", "c"]}], [{"a": ["b"]}, {"b": ["c"]}]],
    ids=["one-list", "chained"],
)
async def test_include_fallback_errors_lists_the_inner_hops_errors_first(
    make_core: MakeCore, host: ScriptedHost, fallbacks: list[object]
) -> None:
    core = build(make_core, fallbacks=fallbacks)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", server_error())

    response = await complete(core, include_fallback_errors=True)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]
    assert headers(response)["x-litellm-fallback-errors"] == f"[{ERROR_500}, {ERROR_429}]"


@pytest.mark.asyncio
async def test_without_include_fallback_errors_no_errors_header_is_set(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    response = await complete(core)

    assert "x-litellm-fallback-errors" not in headers(response)


# --- what each attempt carries to the host ------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_attempt_carries_the_fallback_count_and_original_group_in_its_metadata(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}])
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    await complete(core)

    assert [
        (meta["model_group"], meta["original_model_group"], meta["attempted_fallbacks"])
        for meta in (metadata_of(call.kwargs) for call in host.calls_to("acompletion"))
    ] == [("a", "a", 0), ("b", "a", 1), ("c", "a", 2)]


@pytest.mark.asyncio
async def test_a_fallback_attempt_also_receives_the_routers_walk_state_as_kwargs(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    await complete(core)

    primary, fallback = host.calls_to("acompletion")
    assert {"fallback_depth", "max_fallbacks", "attempted_targets"} & set(primary.kwargs) == set()
    assert (fallback.kwargs["fallback_depth"], fallback.kwargs["max_fallbacks"]) == (1, 5)
    assert "attempted_targets" in fallback.kwargs


@pytest.mark.asyncio
async def test_the_fallback_attempt_sees_the_failed_attempt_in_previous_models(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    await complete(core)

    previous = metadata_of(host.calls_to("acompletion")[1].kwargs)["previous_models"]
    assert [
        (entry["model_group"], entry["deployment_id"], entry["exception_type"])
        for entry in previous  # pyright: ignore[reportGeneralTypeIssues]  # a tuple of dicts
    ] == [("a", "a", "RateLimitError")]


# --- the error when fallbacks give up -------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_debug_on_a_context_window_error_without_its_chain_names_the_missing_setting(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    assert str(raised.value) == (
        "litellm.ContextWindowExceededError: litellm.BadRequestError: too long\n"
        "model=a. context_window_fallbacks=None. fallbacks=None.\n\n"
        "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks\n\n"
        "LiteLLM: model group 'a' failed with the error above. No fallback was attempted."
    )


@pytest.mark.asyncio
async def test_router_debug_on_a_context_window_error_that_also_fails_its_fallback_nests_the_hint(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("*", context_window())

    with pytest.raises(litellm.ContextWindowExceededError) as raised:
        await complete(core)

    hint = "context_window_fallbacks=None. fallbacks=[{'a': ['b']}].\n\nSet 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
    assert str(raised.value) == (
        f"litellm.ContextWindowExceededError: litellm.BadRequestError: too long\nmodel=a. {hint}\n\n"
        "LiteLLM: model group 'a' failed with the error above. Fallback to b also failed: "
        f"litellm.ContextWindowExceededError: litellm.BadRequestError: too long\nmodel=b. {hint}"
    )


@pytest.mark.asyncio
async def test_router_debug_names_a_plain_request_list_that_failed(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core)
    show_router_debug(host)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core, fallbacks=["b", "c"])

    assert str(raised.value) == (
        "litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. "
        "Fallback to b, c also failed: litellm.RateLimitError: slow down"
    )


@pytest.mark.asyncio
async def test_router_debug_with_max_fallbacks_one_names_only_the_first_level(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}], max_fallbacks=1)
    show_router_debug(host)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert str(raised.value).endswith("Fallback to b also failed: litellm.RateLimitError: slow down")


@pytest.mark.asyncio
async def test_router_debug_reports_the_first_fallback_errors_message_not_the_last(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", server_error())
    host.answer("openai/c", timeout())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert str(raised.value) == (
        "litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. "
        "Fallback to b also failed: litellm.InternalServerError: boom"
    )


@pytest.mark.asyncio
async def test_router_debug_truncates_a_long_fallback_error(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit(message="x" * 3000))

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    detail = "litellm.RateLimitError: " + "x" * 3000
    assert str(raised.value) == (
        "litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. "
        f"Fallback to b also failed: {detail[:2000]}... [truncated {len(detail) - 2000} characters]"
    )


@pytest.mark.asyncio
async def test_router_debug_on_an_unmatched_content_policy_chain_says_no_fallback_was_attempted(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    from tests.unit.test_router.core.scripted_host import content_policy

    core = build(make_core, content_policy_fallbacks=[{"other": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", content_policy())

    with pytest.raises(litellm.ContentPolicyViolationError) as raised:
        await complete(core)

    assert str(raised.value) == (
        "litellm.BadRequestError: litellm.ContentPolicyViolationError: blocked\n\n"
        "LiteLLM: model group 'a' failed with the error above. No fallback was attempted."
    )


@pytest.mark.asyncio
async def test_router_debug_no_chain_message_without_any_configured_keys(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core, fallbacks=[{"x": ["b"]}, {"y": ["c"]}])

    assert str(raised.value).endswith(
        "LiteLLM: model group 'a' failed with the error above and no fallback model group was found for it, "
        "so the request was not retried on another model. Fallbacks are configured for: x, y. "
        "Add a fallbacks entry for that model group (Router fallbacks or proxy router_settings.fallbacks) "
        "to retry on another model."
    )


# --- fallback access and budget checks -------------------------------------------------------------


def denying(denied: str, seen: list[str]):  # noqa: ANN201  # a FallbackAccessCheck / FallbackBudgetCheck
    async def check(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
        seen.append(model)
        return model != denied

    return check


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["fallback_access_check", "fallback_budget_check"])
async def test_a_fallback_target_the_check_rejects_is_skipped(
    make_core: MakeCore, host: ScriptedHost, field: str
) -> None:
    seen: list[str] = []
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}], **{field: denying("b", seen)})
    host.answer("openai/a", rate_limit())

    response = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"
    assert seen == ["b", "c"]


def test_sync_completion_runs_the_fallback_access_check(make_core: MakeCore, host: ScriptedHost) -> None:
    seen: list[str] = []
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}], fallback_access_check=denying("b", seen))
    host.answer("openai/a", rate_limit())

    complete_sync(core)

    assert host.deployments_called("completion") == ["a", "c"]
    assert seen == ["b", "c"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "sentence"),
    [
        ("fallback_access_check", "Fallback to b was skipped because the caller is not authorized to call it."),
        ("fallback_budget_check", "Fallback to b was skipped because the caller is over budget for it."),
    ],
)
async def test_router_debug_explains_a_skipped_fallback_target(
    make_core: MakeCore, host: ScriptedHost, field: str, sentence: str
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}], **{field: denying("b", [])})
    show_router_debug(host)
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]
    assert str(raised.value) == (
        f"litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. {sentence}"
    )


@pytest.mark.asyncio
async def test_router_debug_names_both_the_failed_and_the_skipped_targets(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    seen: list[str] = []
    core = build(
        make_core,
        fallbacks=[{"a": ["b", "c"]}],
        fallback_access_check=denying("b", seen),
        fallback_budget_check=denying("b", seen),
    )
    show_router_debug(host)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]
    assert str(raised.value) == (
        "litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. "
        "Fallback to c also failed: litellm.RateLimitError: slow down "
        "Fallback to b was skipped because the caller is not authorized to call it."
    )
    # The access check runs before the budget check; a skipped target is asked again in c's hop.
    assert seen == ["b", "c", "c", "b"]


# --- fallback events ---------------------------------------------------------------------------------


class FallbackEvents(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, object, str]] = []

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ) -> None:
        self.events.append(("success", original_model_group, kwargs.get("model"), type(original_exception).__name__))

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ) -> None:
        self.events.append(("failure", original_model_group, kwargs.get("model"), type(original_exception).__name__))


@pytest.mark.asyncio
async def test_fallback_events_reach_the_process_custom_loggers_innermost_hop_first(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = FallbackEvents()
    monkeypatch.setattr(litellm, "callbacks", [events])
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}])
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", server_error())

    await complete(core)

    assert events.events == [
        ("success", "b", "c", "InternalServerError"),
        ("success", "a", "b", "RateLimitError"),
    ]


@pytest.mark.asyncio
async def test_a_fallback_that_fails_everywhere_logs_failure_events(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = FallbackEvents()
    monkeypatch.setattr(litellm, "callbacks", [events])
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert events.events == [("failure", "a", "b", "RateLimitError")]


# --- content-filter responses ------------------------------------------------------------------------


def content_filtered() -> object:
    response = completion_response()
    response.choices[0].finish_reason = "content_filter"  # pyright: ignore[reportAttributeAccessIssue]  # Choices
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"content_policy_fallbacks": [{"a": ["b"]}]}, ["a", "b"]),
        ({"default_fallbacks": ["c"]}, ["a", "c"]),
        ({"fallbacks": [{"a": ["b"]}]}, ["a"]),
        ({"content_policy_fallbacks": [{"other": ["b"]}], "default_fallbacks": ["c"]}, ["a"]),
    ],
    ids=["content-policy-chain", "default-fallbacks", "general-chain-only", "unmatched-content-policy-chain"],
)
async def test_a_content_filtered_response_falls_back_only_when_a_content_policy_fallback_resolves(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, object], expected: list[str]
) -> None:
    core = build(make_core, **arguments)
    host.answer("openai/a", content_filtered())

    response = await complete(core)

    assert host.deployments_called("acompletion") == expected
    finish_reason = response.choices[0].finish_reason  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponse
    assert finish_reason == ("content_filter" if expected == ["a"] else "stop")


def test_sync_completion_falls_back_on_a_content_filtered_response(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, content_policy_fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", content_filtered())

    complete_sync(core)

    assert host.deployments_called("completion") == ["a", "b"]


@pytest.mark.asyncio
async def test_a_content_filtered_response_with_fallbacks_disabled_raises_content_policy_violation(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, content_policy_fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", content_filtered())

    with pytest.raises(litellm.ContentPolicyViolationError) as raised:
        await complete(core, disable_fallbacks=True)

    assert (
        str(raised.value)
        == "litellm.BadRequestError: litellm.ContentPolicyViolationError: Response output was blocked."
    )
    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_the_last_content_filtered_response_is_returned_when_every_fallback_is_filtered(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, content_policy_fallbacks=[{"a": ["b"]}])
    host.answer("*", content_filtered())

    response = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert response.choices[0].finish_reason == "content_filter"  # pyright: ignore[reportAttributeAccessIssue]  # ModelResponse
    assert headers(response)["x-litellm-model-group"] == "b"


# --- weighted failover ---------------------------------------------------------------------------


def a_deployments_fail(host: ScriptedHost, error: object) -> None:
    for deployment_id in ("a1", "a2", "a3"):
        host.answer(f"openai/{deployment_id}", error)


@pytest.mark.asyncio
async def test_weighted_failover_tries_every_deployment_of_the_group_before_the_external_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=GROUP_A_OF_THREE, enable_weighted_failover=True, fallbacks=[{"a": ["b"]}])
    a_deployments_fail(host, rate_limit())

    response = await complete(core)

    called = host.deployments_called("acompletion")
    assert sorted(called[:3]) == ["a1", "a2", "a3"]
    assert called[3:] == ["b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


@pytest.mark.asyncio
async def test_without_weighted_failover_one_deployment_fails_over_to_the_external_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=GROUP_A_OF_THREE, fallbacks=[{"a": ["b"]}])
    a_deployments_fail(host, rate_limit())

    await complete(core)

    called = host.deployments_called("acompletion")
    assert len(called) == 2
    assert called[0] in {"a1", "a2", "a3"}
    assert called[1] == "b"


@pytest.mark.asyncio
async def test_a_weighted_failover_that_answers_counts_as_a_fallback_inside_the_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=GROUP_A_OF_THREE, enable_weighted_failover=True)
    host.answer("*", fails_once_then_answers(rate_limit()))

    response = await complete(core)

    calls = host.calls_to("acompletion")
    assert len(calls) == 2
    assert calls[0].deployment_id != calls[1].deployment_id
    assert [call.model_group for call in calls] == ["a", "a"]
    assert [metadata_of(call.kwargs)["attempted_fallbacks"] for call in calls] == [0, 1]
    assert headers(response)["x-litellm-model-group"] == "a"
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("max_fallbacks", "group_attempts"), [(0, 1), (1, 2), (2, 3)])
async def test_weighted_failover_is_bounded_by_max_fallbacks(
    make_core: MakeCore, host: ScriptedHost, max_fallbacks: int, group_attempts: int
) -> None:
    core = build(
        make_core,
        model_list=GROUP_A_OF_THREE,
        enable_weighted_failover=True,
        fallbacks=[{"a": ["b"]}],
        max_fallbacks=max_fallbacks,
    )
    a_deployments_fail(host, rate_limit())

    if max_fallbacks == 0:
        with pytest.raises(litellm.RateLimitError):
            await complete(core)
    else:
        await complete(core)

    called = host.deployments_called("acompletion")
    assert len(set(called[:group_attempts])) == group_attempts
    assert called[group_attempts:] == ([] if max_fallbacks == 0 else ["b"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"routing_strategy": "latency-based-routing"}, rate_limit),
        ({}, context_window),
    ],
    ids=["not-simple-shuffle", "context-window-error"],
)
async def test_weighted_failover_does_not_apply(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, object], error: object
) -> None:
    core = build(
        make_core, model_list=GROUP_A_OF_THREE, enable_weighted_failover=True, fallbacks=[{"a": ["b"]}], **arguments
    )
    a_deployments_fail(host, error())  # pyright: ignore[reportCallIssue]  # an error factory

    await complete(core)

    called = host.deployments_called("acompletion")
    assert len(called) == 2
    assert called[1] == "b"


@pytest.mark.asyncio
async def test_weighted_failover_with_fallbacks_disabled_tries_one_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=GROUP_A_OF_THREE, enable_weighted_failover=True, fallbacks=[{"a": ["b"]}])
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, disable_fallbacks=True)

    assert len(host.deployments_called("acompletion")) == 1


def test_sync_completion_also_runs_weighted_failover(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=GROUP_A_OF_THREE, enable_weighted_failover=True, fallbacks=[{"a": ["b"]}])
    a_deployments_fail(host, rate_limit())

    complete_sync(core)

    called = host.deployments_called("completion")
    assert sorted(called[:3]) == ["a1", "a2", "a3"]
    assert called[3:] == ["b"]


@pytest.mark.asyncio
async def test_weighted_failover_on_a_single_deployment_group_goes_straight_to_the_external_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(
        make_core,
        model_list=[deployment("a", "a1"), deployment("b", "b")],
        enable_weighted_failover=True,
        fallbacks=[{"a": ["b"]}],
    )
    host.answer("openai/a1", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a1", "b"]

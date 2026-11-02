"""
Characterization of a router core's fallbacks (non-streaming): chain sources and formats,
client-side fallback lists, per-fallback params, validation, mock-testing flags, depth bounds,
default fallbacks and wildcard lookup. Complements ``test_fallbacks.py``.
"""

import pytest

import litellm
from tests.unit.test_router.core.fallbacks_helpers import (
    TIERED_A,
    MakeCore,
    build,
    complete,
    complete_sync,
    headers,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    content_policy,
    context_window,
    deployment,
    rate_limit,
)

# --- validation of the configured chains ---------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("fallbacks", ["b"], ValueError, "Item 'b' is not a dictionary."),
        (
            "fallbacks",
            [{"a": ["b"], "c": ["b"]}],
            ValueError,
            "Dictionary '{'a': ['b'], 'c': ['b']}' must have exactly one key, but has 2 keys.",
        ),
        (
            "fallbacks",
            [{"a": "b"}],
            TypeError,
            "Fallbacks for 'a' must be a list of model names, got 'b'. Use {'a': ['b']}",
        ),
        (
            "content_policy_fallbacks",
            [{"a": "b"}],
            TypeError,
            "Fallbacks for 'a' must be a list of model names, got 'b'. Use {'a': ['b']}",
        ),
    ],
    ids=["plain-string-item", "two-keys", "string-target", "content-policy-string-target"],
)
def test_a_malformed_fallbacks_configuration_is_refused_at_construction(
    make_core: MakeCore, field: str, value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error) as raised:
        build(make_core, **{field: value})

    assert str(raised.value) == message


@pytest.mark.asyncio
async def test_context_window_fallbacks_are_not_validated_at_construction(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, context_window_fallbacks=[{"a": "b"}])

    await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_constructor_chain_of_parameter_dicts_passes_validation_and_applies_to_any_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"model": "b"}])
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_an_empty_constructor_chain_defers_to_the_process_setting(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(fallbacks=[{"a": ["c"]}])
    core = build(make_core, fallbacks=[])
    host.answer("openai/a", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


# --- client-side fallback lists ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_plain_request_list_is_walked_in_order(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    response = await complete(core, fallbacks=["b", "c"])

    assert host.deployments_called("acompletion") == ["a", "b", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"


@pytest.mark.asyncio
async def test_a_plain_request_list_overrides_the_context_window_chain(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, context_window_fallbacks=[{"a": ["c"]}])
    host.answer("openai/a", context_window())

    await complete(core, fallbacks=["b"])

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_a_dict_target_with_params_overrides_the_request_for_that_attempt(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    host.answer("openai/a", rate_limit())
    override = [{"role": "user", "content": "shorter"}]

    await complete(core, fallbacks=[{"model": "b", "temperature": 0.3, "messages": override}])

    primary, fallback = host.calls_to("acompletion")
    assert (primary.deployment_id, primary.kwargs.get("temperature")) == ("a", None)
    assert primary.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert (fallback.deployment_id, fallback.kwargs["temperature"], fallback.kwargs["messages"]) == ("b", 0.3, override)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"request": [{"model": "b", "temperature": 0.3}, {"model": "c"}]},
        {"constructor": [{"a": [{"model": "b", "temperature": 0.3}, "c"]}]},
    ],
    ids=["request-list", "constructor-chain"],
)
async def test_a_dict_targets_params_carry_over_to_the_next_target(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, list[object]]
) -> None:
    core = build(make_core, **({"fallbacks": arguments["constructor"]} if "constructor" in arguments else {}))
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    await complete(core, **({"fallbacks": arguments["request"]} if "request" in arguments else {}))

    assert [(c.deployment_id, c.kwargs.get("temperature")) for c in host.calls_to("acompletion")] == [
        ("a", None),
        ("b", 0.3),
        ("c", 0.3),
    ]


@pytest.mark.asyncio
async def test_a_dict_target_naming_the_failed_group_with_params_retries_that_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    host.answer("openai/a", rate_limit(), None)

    response = await complete(core, fallbacks=[{"model": "a", "temperature": 0}])

    assert [(c.deployment_id, c.kwargs.get("temperature")) for c in host.calls_to("acompletion")] == [
        ("a", None),
        ("a", 0),
    ]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


@pytest.mark.asyncio
async def test_a_dict_target_with_its_own_api_key_runs_on_a_client_credential_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    host.answer("openai/a", rate_limit())

    response = await complete(core, fallbacks=[{"model": "b", "api_key": "sk-other"}])

    primary, fallback = host.calls_to("acompletion")
    assert (fallback.model, fallback.kwargs["api_key"]) == ("openai/b", "sk-other")
    assert fallback.deployment_id not in {None, "b"}
    assert headers(response)["x-litellm-model-group"] == "b"
    assert primary.deployment_id == "a"


@pytest.mark.asyncio
async def test_a_dict_target_carrying_an_oidc_api_key_is_refused_and_the_primary_error_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core, fallbacks=[{"model": "b", "api_key": "oidc/google/x"}])

    assert host.deployments_called("acompletion") == ["a"]
    assert str(raised.value) == (
        "litellm.RateLimitError: slow down\n\nLiteLLM: model group 'a' failed with the error above. "
        "Fallback to {'model': 'b', 'api_key': 'oidc*********'} also failed: Rejected Request: api_key holds an "
        "oidc/ secret reference, which only a deployment or credential a proxy admin configures can carry."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallbacks", "expected"),
    [(["c", {"a": ["b"]}], ["a", "b", "c"]), (["b", {"x": ["c"]}], ["a", "b"])],
    ids=["keyed-entry-wins-then-string-catches-the-next-hop", "string-entry-is-a-chain-for-every-group"],
)
async def test_a_mixed_list_of_names_and_keyed_entries_is_resolved_as_a_chain_map(
    make_core: MakeCore, host: ScriptedHost, fallbacks: list[object], expected: list[str]
) -> None:
    core = build(make_core)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, fallbacks=fallbacks)

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
async def test_the_request_fallbacks_kwarg_is_not_forwarded_to_the_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core)
    host.answer("openai/a", rate_limit())

    await complete(core, fallbacks=["b"])

    assert [("fallbacks" in c.kwargs) for c in host.calls_to("acompletion")] == [False, False]


# --- disable_fallbacks ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disable_fallbacks_given_as_a_string_does_not_disable_them(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit())

    await complete(core, disable_fallbacks="true")

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_disable_fallbacks_also_stops_the_order_tier_walk(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=TIERED_A)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, disable_fallbacks=True)

    assert host.deployments_called("acompletion") == ["a1"]


@pytest.mark.asyncio
async def test_disable_fallbacks_is_not_forwarded_to_the_provider(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core)

    await complete(core, disable_fallbacks=True)

    assert "disable_fallbacks" not in host.calls_to("acompletion")[0].kwargs


# --- depth bounds -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_request_max_fallbacks_overrides_the_routers(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}, {"b": ["c"]}])
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, max_fallbacks=1)

    assert host.deployments_called("acompletion") == ["a", "b"]


@pytest.mark.asyncio
async def test_max_fallbacks_zero_also_stops_the_order_tier_walk(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, model_list=TIERED_A, max_fallbacks=0)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a1"]


@pytest.mark.asyncio
async def test_max_fallbacks_one_still_tries_every_target_of_the_first_level(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}, {"b": ["c"]}], max_fallbacks=1)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_a_fallback_group_spends_its_own_retries(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}], num_retries=1)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert host.deployments_called("acompletion") == ["a", "a", "b", "b"]
    assert str(raised.value) == "litellm.RateLimitError: slow down LiteLLM Retried: 1 times, LiteLLM Max Retries: 1"


@pytest.mark.asyncio
async def test_a_fallback_hop_resumes_the_original_groups_chain(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b", "c"]}], max_fallbacks=5)
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    response = await complete(core, include_fallback_errors=True)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]
    # c answered inside b's hop (b resumed a's chain), so b's error is listed before a's.
    assert headers(response)["x-litellm-fallback-errors"] == (
        '[{"message": "litellm.RateLimitError: slow down", "type": "throttling_error", "param": null, "code": "429"}, '
        '{"message": "litellm.RateLimitError: slow down", "type": "throttling_error", "param": null, "code": "429"}]'
    )


# --- order tiers with other chains ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [{"request": {"fallbacks": ["c"]}}, {"constructor": {"default_fallbacks": ["c"]}}],
    ids=["plain-request-list", "default-fallbacks"],
)
async def test_order_tiers_run_before_a_plain_list_or_the_default_chain(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, dict[str, object]]
) -> None:
    core = build(make_core, model_list=TIERED_A, **arguments.get("constructor", {}))
    host.answer("*", rate_limit())
    host.answer("openai/c", None)

    response = await complete(core, **arguments.get("request", {}))

    assert host.deployments_called("acompletion") == ["a1", "a2", "c"]
    assert headers(response)["x-litellm-model-group"] == "c"


# --- default fallbacks ---------------------------------------------------------------------------


def test_default_fallbacks_extend_the_process_fallbacks_without_changing_the_process_list(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    process_fallbacks: list[object] = [{"a": ["b"]}]
    host.set_dynamic(fallbacks=process_fallbacks)

    core = build(make_core, default_fallbacks=["c"])
    other = build(make_core, default_fallbacks=["d"])

    assert core.settings().fallbacks == [{"a": ["b"]}, {"*": ["c"]}]
    assert other.settings().fallbacks == [{"a": ["b"]}, {"*": ["d"]}]  # pyright: ignore[reportUnknownMemberType]  # untyped in the checker's view
    assert process_fallbacks == [{"a": ["b"]}]


def test_default_fallbacks_leave_the_constructor_fallbacks_list_unchanged(make_core: MakeCore) -> None:
    fallbacks: list[object] = [{"a": ["b"]}]

    core = build(make_core, fallbacks=fallbacks, default_fallbacks=["c"])

    assert core.settings().fallbacks == [{"a": ["b"]}, {"*": ["c"]}]
    assert fallbacks == [{"a": ["b"]}]


@pytest.mark.asyncio
async def test_an_unknown_group_is_served_by_the_first_default_fallback_without_a_fallback_hop(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, default_fallbacks=["b", "c"])

    response = await complete(core, model="unknown")

    assert host.deployments_called("acompletion") == ["b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 0
    assert headers(response)["x-litellm-model-group"] == "unknown"
    assert host.calls_to("acompletion")[0].model_group == "unknown"


@pytest.mark.asyncio
async def test_an_unknown_group_with_its_own_chain_falls_back_through_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"unknown": ["b"]}])

    response = await complete(core, model="unknown")

    assert host.deployments_called("acompletion") == ["b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1
    assert headers(response)["x-litellm-model-group"] == "b"


@pytest.mark.asyncio
async def test_default_fallbacks_catch_a_content_policy_error_without_a_content_policy_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, default_fallbacks=["c"])
    host.answer("openai/a", content_policy())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "c"]


@pytest.mark.asyncio
async def test_a_request_context_window_fallbacks_of_none_falls_through_to_the_general_chain(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, context_window_fallbacks=[{"a": ["c"]}], fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", context_window())

    await complete(core, context_window_fallbacks=None)

    assert host.deployments_called("acompletion") == ["a", "b"]


# --- wildcard groups -----------------------------------------------------------------------------

WILDCARD = [deployment("openai/*", "w", model="openai/*"), deployment("b", "b")]


@pytest.mark.asyncio
async def test_a_wildcard_routed_request_uses_the_chain_keyed_by_the_bare_model_name(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=WILDCARD, fallbacks=[{"gpt-x": ["b"]}])
    host.answer("openai/gpt-x", rate_limit())

    response = await complete(core, model="openai/gpt-x")

    assert host.deployments_called("acompletion") == ["w", "b"]
    assert [c.model for c in host.calls_to("acompletion")] == ["openai/gpt-x", "openai/b"]
    assert headers(response)["x-litellm-model-group"] == "b"


@pytest.mark.asyncio
async def test_a_chain_keyed_by_the_wildcard_pattern_does_not_match_a_routed_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, model_list=WILDCARD, fallbacks=[{"openai/*": ["b"]}])
    host.answer("openai/gpt-x", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core, model="openai/gpt-x")

    assert host.deployments_called("acompletion") == ["w"]


# --- mock testing flags ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag", "field", "expected"),
    [
        ("mock_testing_fallbacks", "fallbacks", "b"),
        ("mock_testing_context_fallbacks", "context_window_fallbacks", "c"),
        ("mock_testing_content_policy_fallbacks", "content_policy_fallbacks", "c"),
    ],
    ids=["general", "context-window", "content-policy"],
)
async def test_a_mock_testing_flag_skips_the_primary_and_walks_the_matching_chain(
    make_core: MakeCore, host: ScriptedHost, flag: str, field: str, expected: str
) -> None:
    core = build(make_core, num_retries=2, **{"fallbacks": [{"a": ["b"]}], field: [{"a": [expected]}]})

    response = await complete(core, **{flag: True})

    assert host.deployments_called("acompletion") == [expected]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1
    assert headers(response)["x-litellm-attempted-retries"] == 0
    assert host.sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag", "error", "message"),
    [
        (
            "mock_testing_fallbacks",
            litellm.InternalServerError,
            "litellm.InternalServerError: This is a mock exception for model=a, to trigger a fallback. Fallbacks=None",
        ),
        (
            "mock_testing_context_fallbacks",
            litellm.ContextWindowExceededError,
            "litellm.ContextWindowExceededError: litellm.BadRequestError: This is a mock exception for model=a, "
            "to trigger a fallback.                     Context_Window_Fallbacks=None",
        ),
        (
            "mock_testing_content_policy_fallbacks",
            litellm.ContentPolicyViolationError,
            "litellm.BadRequestError: litellm.ContentPolicyViolationError: This is a mock exception for model=a, "
            "to trigger a fallback.                     Context_Policy_Fallbacks=None",
        ),
    ],
    ids=["general", "context-window", "content-policy"],
)
async def test_a_mock_testing_flag_without_a_chain_raises_the_mock_error_without_calling_the_provider(
    make_core: MakeCore, host: ScriptedHost, flag: str, error: type[Exception], message: str
) -> None:
    core = build(make_core)

    with pytest.raises(error) as raised:
        await complete(core, **{flag: True})

    assert str(raised.value) == message
    assert host.calls == []


@pytest.mark.asyncio
async def test_the_mock_error_names_the_configured_chain_and_the_fallback_outcome(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("*", rate_limit())

    with pytest.raises(litellm.InternalServerError) as raised:
        await complete(core, mock_testing_fallbacks=True)

    assert str(raised.value) == (
        "litellm.InternalServerError: This is a mock exception for model=a, to trigger a fallback. "
        "Fallbacks=[{'a': ['b']}]\n\nLiteLLM: model group 'a' failed with the error above. "
        "Fallback to b also failed: litellm.RateLimitError: slow down"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "expected"), [("true", ["b"]), ("false", ["a"]), (False, ["a"])])
async def test_a_mock_testing_flag_given_as_a_string_is_parsed_as_a_bool(
    make_core: MakeCore, host: ScriptedHost, value: object, expected: list[str]
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])

    await complete(core, mock_testing_fallbacks=value)

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
async def test_the_mock_testing_flag_is_not_forwarded_to_the_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])

    await complete(core, mock_testing_fallbacks=True)

    assert "mock_testing_fallbacks" not in host.calls_to("acompletion")[0].kwargs


def test_sync_completion_honours_the_mock_testing_flag(make_core: MakeCore, host: ScriptedHost) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])

    response = complete_sync(core, mock_testing_fallbacks=True)

    assert host.deployments_called("completion") == ["b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1

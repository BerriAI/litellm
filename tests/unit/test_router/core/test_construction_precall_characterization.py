"""
Characterization of a router core's construction and lifecycle: what each constructor argument sets
(and the process settings it falls back to), the validation errors, the caches and callbacks it
creates, alerting, optional pre-call check registration, ``observe``, ``register_plugin``,
``retire``, ``flush_cache``/``reset``, attribute writes, ``get_valid_args`` and ``print_deployment``.

Each test asserts what the Python core does today; oddities are named in the test's docstring.
"""

import logging
from collections.abc import Callable, Generator, Mapping
from typing import Final

import pytest

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.router_api import ChangeSet, router_init_args
from litellm.router_core import RouterCore
from litellm.router_python_core import PythonCore
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import PromptCachingDeploymentCheck
from litellm.types.router import (
    AlertingConfig,
    AllowedFailsPolicy,
    CustomRoutingStrategyBase,
    RetryPolicy,
    RoutingStrategy,
)
from tests.unit.test_router.core.construction_precall_helpers import (
    CallbackHost,
    collect_other_routers,
    host_callback_names,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, records_outcomes_through_callbacks

MakeCore = Callable[..., RouterCore]
MESSAGES: Final = [{"role": "user", "content": "hi"}]


@pytest.fixture
def host() -> ScriptedHost:
    return CallbackHost()


def _one(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: constructor arguments
    return make_core(model_list=[deployment("g", "a")], **arguments)


# --- defaults --------------------------------------------------------------------------------

SETTING_DEFAULTS: Final = {
    "routing_strategy": "simple-shuffle",
    "fallbacks": None,
    "context_window_fallbacks": None,
    "content_policy_fallbacks": None,
    "default_deployment": None,
    "default_max_parallel_requests": None,
    "cooldown_time": 5,
    "allowed_fails": 3,
    "allowed_fails_policy": None,
    "disable_cooldowns": None,
    "enable_pre_call_checks": False,
    "enable_health_check_routing": False,
    "health_check_ignore_transient_errors": False,
    "background_health_check_model_groups": None,
    "enable_tag_filtering": False,
    "tag_filtering_match_any": True,
    "tag_routing_prefix": "",
    "provider_budget_config": None,
    "provider_default_deployment_ids": [],
    "fallback_access_check": None,
    "fallback_budget_check": None,
    "search_tools": [],
    "guardrail_list": [],
    "cache_responses": False,
}


@pytest.mark.parametrize(("field", "expected"), sorted(SETTING_DEFAULTS.items()))
def test_settings_default_of_a_router_built_with_only_a_model_list(
    make_core: MakeCore, field: str, expected: object
) -> None:
    """Note: ``fallbacks`` defaults to ``[]`` in the signature, yet reads ``None``: an empty list falls
    through ``fallbacks or <process fallbacks>``."""
    assert getattr(_one(make_core).settings(), field) == expected


def test_default_litellm_params_default_to_no_timeout_no_client_retries_and_no_caching_groups(
    make_core: MakeCore,
) -> None:
    assert _one(make_core).settings().default_litellm_params == {
        "timeout": None,
        "max_retries": 0,
        "metadata": {"caching_groups": None},
    }


def test_router_general_settings_default_to_async_only_and_pass_through_off(make_core: MakeCore) -> None:
    general: Final = _one(make_core).settings().router_general_settings
    assert (general.async_only_mode, general.pass_through_all_models) == (False, False)


def test_settings_as_dict_of_a_default_router(make_core: MakeCore) -> None:
    assert dict(_one(make_core).settings().as_dict()) == {
        "routing_strategy_args": {},
        "routing_strategy": "simple-shuffle",
        "allowed_fails": 3,
        "cooldown_time": 5,
        "num_retries": 2,
        "timeout": 6000.0,
        "retry_after": 0,
        "fallbacks": None,
        "context_window_fallbacks": None,
        "model_group_retry_policy": {},
        "retry_policy": None,
        "model_group_alias": {},
        "enable_weighted_failover": False,
        "enable_mid_stream_fallback_continuation": False,
        "enable_tag_filtering": False,
        "tag_routing_prefix": "",
        "fallback_on_unsupported_params": True,
        "routing_groups": [],
    }


ATTRIBUTE_DEFAULTS: Final = {
    "num_retries": 2,
    "max_fallbacks": 5,
    "timeout": 6000.0,
    "stream_timeout": None,
    "request_timeout": None,
    "retry_after": 0,
    "client_ttl": 3600,
    "set_verbose": False,
    "debug_level": "INFO",
    "retry_policy": None,
    "model_group_retry_policy": {},
    "alerting_config": None,
    "routing_strategy_args": {},
    "deployment_affinity_ttl_seconds": 3600,
    "default_priority": None,
    "optional_callbacks": None,
    "router_budget_logger": None,
    "fallback_on_unsupported_params": True,
    "enable_weighted_failover": False,
    "enable_mid_stream_fallback_continuation": False,
    "explicit_allowed_fails": None,
    "ignore_invalid_deployments": False,
    "auto_router_capability_limit": None,
    "assistants_config": None,
    "session_affinity_repin_on_failover": False,
    "model_group_affinity_config": None,
    "access_groups": None,
    "leastbusy_logger": None,
    "lowesttpm_logger": None,
}


@pytest.mark.parametrize(("name", "expected"), sorted(ATTRIBUTE_DEFAULTS.items()))
def test_attribute_default_of_a_router_built_with_only_a_model_list(
    make_core: MakeCore, name: str, expected: object
) -> None:
    assert getattr(_one(make_core), name) == expected


def test_a_router_without_a_model_list_has_no_deployments(make_core: MakeCore) -> None:
    core: Final = make_core()
    assert list(core.registry().all()) == []


def test_construction_turns_off_the_host_debug_info(make_core: MakeCore, host: ScriptedHost) -> None:
    _one(make_core)
    assert host.suppressed is True


def test_model_group_alias_none_reads_as_an_empty_alias_map(make_core: MakeCore) -> None:
    core: Final = _one(make_core, model_group_alias=None)
    assert dict(core.settings().as_dict()["model_group_alias"]) == {}  # pyright: ignore[reportArgumentType]


# --- process settings each argument falls back to -----------------------------------------


@pytest.mark.parametrize(("argument", "host_value", "expected"), [(5, 4, 5), (0, 4, 0), (None, 4, 4), (None, None, 2)])
def test_num_retries_comes_from_the_argument_then_the_process_then_the_openai_default(
    make_core: MakeCore, host: ScriptedHost, argument: int | None, host_value: int | None, expected: int
) -> None:
    host.set_dynamic(num_retries=host_value)
    core: Final = _one(make_core, **({"num_retries": argument} if argument is not None else {}))
    assert core.settings().as_dict()["num_retries"] == expected


@pytest.mark.asyncio
async def test_num_retries_from_the_process_settings_governs_the_attempts(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(num_retries=1)
    host.answer("openai/a", litellm.InternalServerError(message="boom", llm_provider="openai", model="a"))
    core: Final = _one(make_core)
    with pytest.raises(litellm.InternalServerError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert len(host.calls) == 2


@pytest.mark.parametrize(
    ("argument", "host_max", "host_router_max", "expected"),
    [(3, 9, 11, 3), (0, 9, 11, 0), (None, 9, 11, 9), (None, None, 11, 11)],
)
def test_max_fallbacks_comes_from_the_argument_then_litellm_max_fallbacks_then_router_max_fallbacks(
    make_core: MakeCore,
    host: ScriptedHost,
    argument: int | None,
    host_max: int | None,
    host_router_max: int,
    expected: int,
) -> None:
    host.set_dynamic(max_fallbacks=host_max, router_max_fallbacks=host_router_max)
    core: Final = _one(make_core, **({"max_fallbacks": argument} if argument is not None else {}))
    assert core.max_fallbacks == expected  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("argument", "expected", "explicit"), [(7, 7, 7), (0, 0, 0), (None, 8, None)])
def test_allowed_fails_comes_from_the_argument_then_the_process_and_only_an_argument_counts_as_explicit(
    make_core: MakeCore, host: ScriptedHost, argument: int | None, expected: int, explicit: int | None
) -> None:
    host.set_dynamic(allowed_fails=8)
    core: Final = _one(make_core, **({"allowed_fails": argument} if argument is not None else {}))
    assert core.settings().allowed_fails == expected
    assert core.explicit_allowed_fails == explicit  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("argument", "expected"), [(12.5, 12.5), (None, 33.0), (0, 33.0)])
def test_timeout_comes_from_the_argument_else_the_process_request_timeout(
    make_core: MakeCore, host: ScriptedHost, argument: float | None, expected: float
) -> None:
    """Oddity: ``timeout=0`` is falsy and reads as the process ``request_timeout``."""
    host.set_dynamic(request_timeout=33.0)
    core: Final = _one(make_core, **({"timeout": argument} if argument is not None else {}))
    assert core.settings().as_dict()["timeout"] == expected


def test_timeout_zero_still_reaches_default_litellm_params_as_zero(make_core: MakeCore) -> None:
    core: Final = _one(make_core, timeout=0)
    assert core.settings().default_litellm_params["timeout"] == 0


@pytest.mark.parametrize(("argument", "expected"), [(30, 30), (0.5, 0.5), (0, 5), (None, 5)])
def test_cooldown_time_defaults_to_five_seconds_and_zero_reads_as_the_default(
    make_core: MakeCore, argument: float | None, expected: float
) -> None:
    """Oddity: ``cooldown_time=0`` is replaced by the 5 second default (``cooldown_time or DEFAULT``)."""
    core: Final = _one(make_core, **({"cooldown_time": argument} if argument is not None else {}))
    assert core.settings().cooldown_time == expected


def test_request_timeout_is_unset_without_a_router_timeout(
    make_core: MakeCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "request_timeout", 77.0)
    assert _one(make_core).request_timeout is None  # pyright: ignore[reportAttributeAccessIssue]


def test_request_timeout_reads_the_host_request_timeout_when_a_router_timeout_is_set(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read from the host's process settings (33 here), not from ``litellm.request_timeout`` itself."""
    host.set_dynamic(request_timeout=33.0)
    monkeypatch.setattr(litellm, "request_timeout", 77.0)
    assert _one(make_core, timeout=10).request_timeout == 33.0  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


def test_request_timeout_is_unset_when_the_host_request_timeout_is_its_untouched_default(
    make_core: MakeCore,
    host: ScriptedHost,
    monkeypatch: pytest.MonkeyPatch,  # pyright: ignore[reportUnknownMemberType, reportUnknownParameterType]  # pytest and the cores' untyped members
) -> None:
    host.set_dynamic(request_timeout=6000.0, request_timeout_explicitly_set=False)
    monkeypatch.setattr(litellm, "request_timeout", 77.0)  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
    assert _one(make_core, timeout=10).request_timeout is None  # pyright: ignore[reportAttributeAccessIssue]


def test_request_timeout_keeps_the_default_value_when_the_host_marks_it_explicitly_set(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    host.set_dynamic(request_timeout=6000.0, request_timeout_explicitly_set=True)
    assert _one(make_core, timeout=10).request_timeout == 6000.0  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


def test_stream_timeout_is_kept_as_given(make_core: MakeCore) -> None:
    assert _one(make_core, stream_timeout=3).stream_timeout == 3  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("field", ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"])
def test_fallback_lists_come_from_the_process_when_the_argument_is_empty(
    make_core: MakeCore, host: ScriptedHost, field: str
) -> None:
    host.set_dynamic(**{field: [{"g": ["process"]}]})
    assert getattr(_one(make_core).settings(), field) == [{"g": ["process"]}]


@pytest.mark.parametrize("field", ["fallbacks", "context_window_fallbacks", "content_policy_fallbacks"])
def test_fallback_list_arguments_win_over_the_process_lists(
    make_core: MakeCore, host: ScriptedHost, field: str
) -> None:
    host.set_dynamic(**{field: [{"g": ["process"]}]})
    assert getattr(_one(make_core, **{field: [{"g": ["argument"]}]}).settings(), field) == [{"g": ["argument"]}]


def test_default_fallbacks_are_appended_to_the_fallbacks_as_a_star_entry(make_core: MakeCore) -> None:
    core: Final = _one(make_core, fallbacks=[{"g": ["b"]}], default_fallbacks=["c"])
    assert core.settings().fallbacks == [{"g": ["b"]}, {"*": ["c"]}]


def test_default_fallbacks_alone_become_the_only_star_entry(make_core: MakeCore) -> None:
    assert _one(make_core, default_fallbacks=["c"]).settings().fallbacks == [{"*": ["c"]}]


def test_default_fallbacks_come_from_the_process_when_not_given(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(default_fallbacks=["d"])
    assert _one(make_core).settings().fallbacks == [{"*": ["d"]}]


def test_default_fallbacks_given_as_a_string_raise_type_error(make_core: MakeCore) -> None:
    with pytest.raises(TypeError, match=r"^default_fallbacks must be a list of model names, got 'b'\. Use \['b'\]$"):
        _one(make_core, default_fallbacks="b")


def test_default_fallbacks_extend_a_copy_of_the_process_fallbacks_list(make_core: MakeCore, host: ScriptedHost) -> None:
    """With no ``fallbacks`` argument the router starts from the process list, but appends the star
    entry to its own copy, so the process setting is left alone."""
    process_fallbacks: Final[list[object]] = [{"g": ["e"]}]
    host.set_dynamic(fallbacks=process_fallbacks)
    core: Final = _one(make_core, default_fallbacks=["c"])
    assert core.settings().fallbacks == [{"g": ["e"]}, {"*": ["c"]}]
    assert process_fallbacks == [{"g": ["e"]}]


# --- validation --------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["fallbacks", "content_policy_fallbacks"])
def test_a_fallback_entry_that_is_not_a_dict_raises_value_error(make_core: MakeCore, field: str) -> None:
    with pytest.raises(ValueError, match=r"^Item 'x' is not a dictionary\.$"):
        _one(make_core, **{field: ["x"]})


@pytest.mark.parametrize("field", ["fallbacks", "content_policy_fallbacks"])
def test_a_fallback_entry_with_two_keys_raises_value_error(make_core: MakeCore, field: str) -> None:
    with pytest.raises(ValueError, match=r"must have exactly one key, but has 2 keys\.$"):
        _one(make_core, **{field: [{"a": ["b"], "c": ["d"]}]})


@pytest.mark.parametrize("field", ["fallbacks", "content_policy_fallbacks"])
def test_a_fallback_target_given_as_a_string_raises_type_error(make_core: MakeCore, field: str) -> None:
    with pytest.raises(TypeError) as error:
        _one(make_core, **{field: [{"a": "b"}]})
    assert str(error.value) == "Fallbacks for 'a' must be a list of model names, got 'b'. Use {'a': ['b']}"


@pytest.mark.parametrize("targets", [[], [{"model": "b"}], ["b", "c"]])
def test_fallback_targets_as_a_list_of_names_or_dicts_are_accepted(make_core: MakeCore, targets: list[object]) -> None:
    assert _one(make_core, fallbacks=[{"g": targets}]).settings().fallbacks == [{"g": targets}]


@pytest.mark.parametrize("value", [["x"], [{"a": "b"}]])
def test_context_window_fallbacks_are_not_validated(make_core: MakeCore, value: list[object]) -> None:
    """Oddity: unlike ``fallbacks`` and ``content_policy_fallbacks``, malformed context window
    fallbacks are accepted as given."""
    assert _one(make_core, context_window_fallbacks=value).settings().context_window_fallbacks == value


def test_malformed_process_fallbacks_raise_at_construction(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(fallbacks=[{"a": "e"}])
    with pytest.raises(TypeError, match=r"^Fallbacks for 'a' must be a list of model names"):
        _one(make_core)


def test_validate_fallbacks_accepts_none(make_core: MakeCore) -> None:
    assert _one(make_core).validate_fallbacks(fallback_param=None) is None  # pyright: ignore[reportAttributeAccessIssue]


def test_an_unknown_routing_strategy_raises_value_error(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match=r"^Invalid routing_strategy: 'bogus'\.") as error:
        _one(make_core, routing_strategy="bogus")
    assert str(error.value).startswith("Invalid routing_strategy: 'bogus'. Valid options: [")
    assert str(error.value).endswith(
        "Check 'router_settings.routing_strategy' in your config.yaml "
        "or the 'routing_strategy' parameter if using the Router SDK directly."
    )


def test_a_routing_strategy_enum_is_stored_as_its_string_value(make_core: MakeCore) -> None:
    assert _one(make_core, routing_strategy=RoutingStrategy.LEAST_BUSY).settings().routing_strategy == "least-busy"


@pytest.mark.parametrize(
    "strategy",
    [
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "usage-based-routing-v2",
        "latency-based-routing",
        "cost-based-routing",
        "lar1",
    ],
)
def test_each_routing_strategy_is_reported_in_settings(make_core: MakeCore, strategy: str) -> None:
    assert _one(make_core, routing_strategy=strategy).settings().routing_strategy == strategy


# --- default_litellm_params ----------------------------------------------------------------


def test_default_litellm_params_keep_given_values_and_gain_max_retries_and_caching_groups(make_core: MakeCore) -> None:
    core: Final = _one(
        make_core,
        timeout=12.5,
        default_litellm_params={"timeout": 99, "temperature": 0.1, "metadata": {"x": 1}},
        caching_groups=[("a", "b")],
    )
    assert core.settings().default_litellm_params == {
        "timeout": 99,
        "temperature": 0.1,
        "metadata": {"x": 1, "caching_groups": [("a", "b")]},
        "max_retries": 0,
    }


def test_default_litellm_params_timeout_defaults_to_the_router_timeout_argument(make_core: MakeCore) -> None:
    assert _one(make_core, timeout=12.5).settings().default_litellm_params["timeout"] == 12.5


def test_default_litellm_params_argument_is_not_changed_by_construction(make_core: MakeCore) -> None:
    given: Final = {"temperature": 0.1}
    _one(make_core, default_litellm_params=given)
    assert given == {"temperature": 0.1}


# --- policies, priority, scheduler -------------------------------------------------------


def test_retry_policy_given_as_a_dict_becomes_a_retry_policy(make_core: MakeCore) -> None:
    policy: Final = _one(make_core, retry_policy={"RateLimitErrorRetries": 3}).retry_policy  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(policy, RetryPolicy)
    assert policy.RateLimitErrorRetries == 3


def test_retry_policy_given_as_a_retry_policy_is_kept(make_core: MakeCore) -> None:
    policy: Final = RetryPolicy(TimeoutErrorRetries=1)
    assert _one(make_core, retry_policy=policy).retry_policy == policy  # pyright: ignore[reportAttributeAccessIssue]


def test_retry_policy_of_another_type_is_ignored(make_core: MakeCore) -> None:
    assert _one(make_core, retry_policy="nope").retry_policy is None  # pyright: ignore[reportAttributeAccessIssue]


def test_allowed_fails_policy_given_as_a_dict_becomes_an_allowed_fails_policy(make_core: MakeCore) -> None:
    policy: Final = (
        _one(make_core, allowed_fails_policy={"RateLimitErrorAllowedFails": 7}).settings().allowed_fails_policy
    )
    assert isinstance(policy, AllowedFailsPolicy)
    assert policy.RateLimitErrorAllowedFails == 7


def test_allowed_fails_policy_of_another_type_is_ignored(make_core: MakeCore) -> None:
    assert _one(make_core, allowed_fails_policy="nope").settings().allowed_fails_policy is None


def test_model_group_retry_policy_is_kept_as_given(make_core: MakeCore) -> None:
    policy: Final = {"g": RetryPolicy(TimeoutErrorRetries=4)}
    assert _one(make_core, model_group_retry_policy=policy).model_group_retry_policy == policy  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("given", "expected"), [(1, 1), ("2", 2), (-1, -1), (None, None), ("x", None), (1.5, None), (True, None)]
)
def test_default_priority_keeps_an_integer_and_ignores_anything_else(
    make_core: MakeCore, given: object, expected: int | None
) -> None:
    assert _one(make_core, default_priority=given).default_priority == expected  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(("given", "expected"), [(None, 0.03), (0.5, 0.5)])
def test_scheduler_polling_interval_defaults_to_three_hundredths_of_a_second(
    make_core: MakeCore, given: float | None, expected: float
) -> None:
    core: Final = _one(make_core, **({"polling_interval": given} if given is not None else {}))
    assert core.scheduler.polling_interval == expected  # pyright: ignore[reportAttributeAccessIssue]


def test_client_ttl_is_kept_as_given(make_core: MakeCore) -> None:
    assert _one(make_core, client_ttl=10).client_ttl == 10  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("routing_strategy_args", {"ttl": 5}),
        ("deployment_affinity_ttl_seconds", 60),
        ("session_affinity_repin_on_failover", True),
        ("enable_weighted_failover", True),
        ("fallback_on_unsupported_params", False),
        ("ignore_invalid_deployments", True),
        ("retry_after", 4),
    ],
)
def test_plain_arguments_are_kept_as_attributes(make_core: MakeCore, name: str, value: object) -> None:
    assert getattr(_one(make_core, **{name: value}), name) == value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("enable_tag_filtering", True),
        ("tag_filtering_match_any", False),
        ("tag_routing_prefix", "team-"),
        ("enable_pre_call_checks", True),
        ("disable_cooldowns", True),
        ("default_max_parallel_requests", 4),
        ("enable_health_check_routing", True),
        ("health_check_ignore_transient_errors", True),
    ],
)
def test_setting_arguments_are_reported_in_settings(make_core: MakeCore, name: str, value: object) -> None:
    assert getattr(_one(make_core, **{name: value}).settings(), name) == value


def test_background_health_check_model_groups_become_a_frozenset(make_core: MakeCore) -> None:
    core: Final = _one(make_core, background_health_check_model_groups=["g", "g", "h"])
    assert core.settings().background_health_check_model_groups == frozenset({"g", "h"})


@pytest.mark.parametrize(("given", "expected"), [(True, True), ("true", True), ("false", False), (False, False)])
def test_mid_stream_fallback_continuation_accepts_a_bool_or_a_bool_string(
    make_core: MakeCore, given: object, expected: bool
) -> None:
    assert (
        _one(make_core, enable_mid_stream_fallback_continuation=given).enable_mid_stream_fallback_continuation
        is expected
    )  # pyright: ignore[reportAttributeAccessIssue]


# --- logging level -------------------------------------------------------------------------


@pytest.fixture
def router_log_level() -> Generator[int, None, None]:
    level: Final = verbose_router_logger.level
    yield level
    verbose_router_logger.setLevel(level)


@pytest.mark.parametrize(
    ("set_verbose", "debug_level", "expected"),
    [(True, "DEBUG", logging.DEBUG), (True, "INFO", logging.INFO)],
)
def test_set_verbose_sets_the_router_logger_to_the_debug_level(
    make_core: MakeCore, router_log_level: object, set_verbose: bool, debug_level: str, expected: int
) -> None:
    verbose_router_logger.setLevel(logging.WARNING)
    _one(make_core, set_verbose=set_verbose, debug_level=debug_level)
    assert verbose_router_logger.level == expected


def test_debug_level_without_set_verbose_leaves_the_router_logger_alone(
    make_core: MakeCore, router_log_level: object
) -> None:
    verbose_router_logger.setLevel(logging.WARNING)
    _one(make_core, debug_level="DEBUG")
    assert verbose_router_logger.level == logging.WARNING


# --- redis and response caching ----------------------------------------------------------


def test_redis_host_and_port_give_the_routing_state_a_redis_cache(make_core: MakeCore) -> None:
    """The port is passed on as a string."""
    core: Final = _one(
        make_core, redis_host="localhost", redis_port=6399, redis_password="pw", cache_kwargs={"ssl": False}
    )
    redis_cache: Final = core.health().store.redis_cache
    assert isinstance(redis_cache, RedisCache)
    assert {k: v for k, v in redis_cache.redis_kwargs.items() if k != "socket_timeout"} == {
        "host": "localhost",
        "port": "6399",
        "password": "pw",
        "ssl": False,
    }


def test_redis_url_gives_the_routing_state_a_redis_cache(make_core: MakeCore) -> None:
    redis_cache: Final = _one(make_core, redis_url="redis://localhost:6399").health().store.redis_cache
    assert isinstance(redis_cache, RedisCache)
    assert redis_cache.redis_kwargs["url"] == "redis://localhost:6399"


def test_redis_db_is_passed_on_as_a_string(make_core: MakeCore) -> None:
    redis_cache: Final = _one(make_core, redis_host="localhost", redis_port=6399, redis_db=3).health().store.redis_cache
    assert isinstance(redis_cache, RedisCache)
    assert redis_cache.redis_kwargs["db"] == "3"


def test_redis_host_without_a_port_creates_no_redis_cache(make_core: MakeCore) -> None:
    assert _one(make_core, redis_host="localhost").health().store.redis_cache is None


def test_without_redis_arguments_the_routing_state_is_in_memory_only(make_core: MakeCore) -> None:
    assert _one(make_core).health().store.redis_cache is None


def test_the_routing_state_and_prompt_caching_pins_share_one_redis_cache(make_core: MakeCore) -> None:
    core: Final = _one(make_core, redis_host="localhost", redis_port=6399)
    assert core.registry().plugins("prompt_caching_pins").redis_cache is core.health().store.redis_cache


def test_applying_a_redis_cache_attaches_it_to_the_routing_state_and_the_pins(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    redis_cache: Final = RedisCache(host="localhost", port=6399)
    core.apply(ChangeSet(redis_cache=redis_cache))
    assert core.health().store.redis_cache is redis_cache
    assert core.registry().plugins("prompt_caching_pins").redis_cache is redis_cache


def test_cache_responses_installs_a_local_response_cache_on_the_host(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one(make_core, cache_responses=True)
    assert isinstance(host.cache, litellm.Cache)
    assert host.cache.type == "local"
    assert core.settings().cache_responses is True


def test_cache_responses_with_redis_installs_a_redis_response_cache(make_core: MakeCore, host: ScriptedHost) -> None:
    _one(make_core, cache_responses=True, redis_host="localhost", redis_port=6399)
    assert isinstance(host.cache, litellm.Cache)
    assert host.cache.type == "redis"


def test_cache_responses_keeps_a_response_cache_the_host_already_has(make_core: MakeCore, host: ScriptedHost) -> None:
    existing: Final = object()
    host.cache = existing
    _one(make_core, cache_responses=True)
    assert host.cache is existing


def test_without_cache_responses_the_host_response_cache_is_untouched(make_core: MakeCore, host: ScriptedHost) -> None:
    _one(make_core, redis_host="localhost", redis_port=6399)
    assert host.cache is None


# --- callback registration ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "callbacks", "input_callbacks"),
    [
        ("simple-shuffle", [], []),
        ("lar1", [], []),
        ("least-busy", ["LeastBusyLoggingHandler"], ["LeastBusyLoggingHandler"]),
        ("usage-based-routing", ["LowestTPMLoggingHandler"], []),
        ("usage-based-routing-v2", ["LowestTPMLoggingHandler_v2"], []),
        ("latency-based-routing", ["LowestLatencyLoggingHandler"], []),
        ("cost-based-routing", ["LowestCostLoggingHandler"], []),
    ],
)
def test_each_routing_strategy_registers_its_selector_on_the_host(
    make_core: MakeCore, host: ScriptedHost, strategy: str, callbacks: list[str], input_callbacks: list[str]
) -> None:
    _one(make_core, routing_strategy=strategy)
    names: Final = host_callback_names(host)
    assert (names["callbacks"], names["input_callback"]) == (callbacks, input_callbacks)


def test_the_least_busy_selector_is_the_router_leastbusy_logger(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one(make_core, routing_strategy="least-busy")
    assert host.lists["callbacks"] == [core.leastbusy_logger]  # pyright: ignore[reportAttributeAccessIssue]


# --- alerting ----------------------------------------------------------------------------------


def test_alerting_config_registers_slack_alerting_and_its_slow_response_callback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(
        make_core, alerting_config=AlertingConfig(webhook_url="https://example.invalid/hook", alerting_threshold=10)
    )
    names: Final = host_callback_names(host)
    assert names["callbacks"] == ["SlackAlerting"]
    assert names["success_callback"] == ["sync_deployment_callback_on_success", "response_taking_too_long_callback"]
    logger: Final = core.observe().slack_alerting_logger
    assert host.lists["callbacks"] == [logger]
    assert logger is not None
    assert (logger.alerting_threshold, logger.alerting, logger.default_webhook_url) == (
        10.0,
        ["slack"],
        "https://example.invalid/hook",
    )


def test_observe_reports_no_slack_alerting_without_an_alerting_config(make_core: MakeCore) -> None:
    assert _one(make_core).observe().slack_alerting_logger is None


# --- optional pre-call checks: registration -----------------------------------------------


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        (["prompt_caching"], ["PromptCachingDeploymentCheck"]),
        (["enforce_model_rate_limits"], ["ModelRateLimitingCheck"]),
        (["deployment_affinity"], ["DeploymentAffinityCheck"]),
        (["responses_api_deployment_check"], ["DeploymentAffinityCheck"]),
        (["session_affinity"], ["DeploymentAffinityCheck"]),
        (["deployment_affinity", "responses_api_deployment_check", "session_affinity"], ["DeploymentAffinityCheck"]),
        (["encrypted_content_affinity"], ["EncryptedContentAffinityCheck"]),
        (["prompt_caching", "prompt_caching"], ["PromptCachingDeploymentCheck"]),
        (["prompt_caching", "enforce_model_rate_limits"], ["PromptCachingDeploymentCheck", "ModelRateLimitingCheck"]),
    ],
)
def test_each_optional_pre_call_check_registers_one_callback_on_the_router_and_the_host(
    make_core: MakeCore, host: ScriptedHost, checks: list[str], expected: list[str]
) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=checks)
    assert [type(callback).__name__ for callback in core.optional_callbacks] == expected  # pyright: ignore[reportAttributeAccessIssue]
    assert host.lists["callbacks"] == core.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("checks", "flags"),
    [
        (["deployment_affinity"], (True, False, False)),
        (["responses_api_deployment_check"], (False, True, False)),
        (["session_affinity"], (False, False, True)),
        (["deployment_affinity", "session_affinity"], (True, False, True)),
    ],
)
def test_affinity_checks_share_one_callback_with_one_flag_each(
    make_core: MakeCore, checks: list[str], flags: tuple[bool, bool, bool]
) -> None:
    (callback,) = _one(
        make_core, optional_pre_call_checks=checks, deployment_affinity_ttl_seconds=60
    ).optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    assert (
        callback.enable_user_key_affinity,
        callback.enable_responses_api_affinity,
        callback.enable_session_id_affinity,
    ) == flags
    assert callback.ttl_seconds == 60


def test_encrypted_content_affinity_is_placed_before_deployment_affinity(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=["deployment_affinity", "encrypted_content_affinity"])
    expected: Final = ["EncryptedContentAffinityCheck", "DeploymentAffinityCheck"]
    assert [type(callback).__name__ for callback in core.optional_callbacks] == expected  # pyright: ignore[reportAttributeAccessIssue]
    assert host_callback_names(host)["callbacks"] == expected


def test_forward_client_headers_by_model_group_registers_nothing(make_core: MakeCore, host: ScriptedHost) -> None:
    """Oddity: a listed optional pre-call check that the router ignores."""
    core: Final = _one(make_core, optional_pre_call_checks=["forward_client_headers_by_model_group"])
    assert core.optional_callbacks is None  # pyright: ignore[reportAttributeAccessIssue]
    assert host.lists["callbacks"] == []


def test_an_empty_optional_pre_call_check_list_registers_nothing(make_core: MakeCore, host: ScriptedHost) -> None:
    assert _one(make_core, optional_pre_call_checks=[]).optional_callbacks is None  # pyright: ignore[reportAttributeAccessIssue]
    assert host.lists["callbacks"] == []


def test_router_budget_limiting_can_be_built_outside_an_event_loop(make_core: MakeCore, host: ScriptedHost) -> None:
    """Built from synchronous code, the budget limiter defers its background Redis sync until a loop runs."""
    core: Final = _one(make_core, optional_pre_call_checks=["router_budget_limiting"])
    limiter: Final = core.registry().plugins("budget_limiter")
    assert isinstance(limiter, RouterBudgetLimiting)
    assert host.lists["callbacks"] == [limiter]
    core.retire()


def test_a_provider_budget_config_can_be_built_outside_an_event_loop(make_core: MakeCore) -> None:
    core: Final = _one(make_core, provider_budget_config={"openai": {"budget_limit": 1, "time_period": "1d"}})
    assert isinstance(core.registry().plugins("budget_limiter"), RouterBudgetLimiting)
    core.retire()


def test_a_deployment_budget_can_be_built_outside_an_event_loop(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", max_budget=1, budget_duration="1d")])
    assert isinstance(core.registry().plugins("budget_limiter"), RouterBudgetLimiting)
    core.retire()


@pytest.mark.asyncio
async def test_router_budget_limiting_registers_the_budget_limiter(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=["router_budget_limiting"])
    limiter: Final = core.router_budget_logger  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(limiter, RouterBudgetLimiting)
    assert core.registry().plugins("budget_limiter") is limiter
    assert host.lists["callbacks"] == [limiter]
    core.retire()


@pytest.mark.asyncio
async def test_a_provider_budget_config_adds_the_budget_limiter_by_itself(make_core: MakeCore) -> None:
    core: Final = _one(make_core, provider_budget_config={"openai": {"budget_limit": 1, "time_period": "1d"}})
    assert [type(callback).__name__ for callback in core.optional_callbacks] == ["RouterBudgetLimiting"]  # pyright: ignore[reportAttributeAccessIssue]
    core.retire()


@pytest.mark.asyncio
async def test_a_deployment_budget_adds_the_budget_limiter_by_itself(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", max_budget=1, budget_duration="1d")])
    assert isinstance(core.registry().plugins("budget_limiter"), RouterBudgetLimiting)
    core.retire()


def test_model_group_affinity_config_adds_an_affinity_callback_with_every_global_flag_off(make_core: MakeCore) -> None:
    (callback,) = _one(make_core, model_group_affinity_config={"g": ["deployment_affinity"]}).optional_callbacks  # pyright: ignore[reportAttributeAccessIssue]
    assert type(callback).__name__ == "DeploymentAffinityCheck"
    assert (
        callback.enable_user_key_affinity,
        callback.enable_responses_api_affinity,
        callback.enable_session_id_affinity,
    ) == (False, False, False)
    assert callback.model_group_affinity_config == {"g": ["deployment_affinity"]}


def test_model_group_encrypted_content_affinity_adds_both_affinity_callbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    _one(make_core, model_group_affinity_config={"g": ["encrypted_content_affinity"]})
    assert host_callback_names(host)["callbacks"] == ["EncryptedContentAffinityCheck", "DeploymentAffinityCheck"]


def test_mid_stream_fallback_continuation_registers_a_global_prefill_check(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core, enable_mid_stream_fallback_continuation=True)
    assert host_callback_names(host)["callbacks"] == ["ContinuationPrefillDeploymentCheck"]
    assert core.optional_callbacks is None  # pyright: ignore[reportAttributeAccessIssue]


def test_add_optional_pre_call_checks_after_construction_registers_the_new_check(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core.add_optional_pre_call_checks(["enforce_model_rate_limits", "prompt_caching"])  # pyright: ignore[reportAttributeAccessIssue]
    assert host_callback_names(host)["callbacks"] == ["PromptCachingDeploymentCheck", "ModelRateLimitingCheck"]


def test_add_optional_pre_call_checks_with_none_and_apply_default_settings_change_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    before: Final = list(host.lists["callbacks"])
    core.add_optional_pre_call_checks(None)  # pyright: ignore[reportAttributeAccessIssue]
    core.apply_default_settings()  # pyright: ignore[reportAttributeAccessIssue]
    assert host.lists["callbacks"] == before


def test_runtime_optional_pre_call_checks_drop_unlisted_toggleable_checks_but_keep_affinity(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching", "deployment_affinity"])
    core.apply(ChangeSet(settings={"optional_pre_call_checks": ["enforce_model_rate_limits"]}))
    expected: Final = ["DeploymentAffinityCheck", "ModelRateLimitingCheck"]
    assert [type(callback).__name__ for callback in core.optional_callbacks] == expected  # pyright: ignore[reportAttributeAccessIssue]
    assert host_callback_names(host)["callbacks"] == expected


def test_runtime_optional_pre_call_checks_none_changes_nothing(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core.apply(ChangeSet(settings={"optional_pre_call_checks": None}))
    assert host_callback_names(host)["callbacks"] == ["PromptCachingDeploymentCheck"]


def test_runtime_removal_keeps_the_host_callback_while_another_live_router_on_the_host_uses_that_check(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    other: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core.apply(ChangeSet(settings={"optional_pre_call_checks": []}))
    assert core.optional_callbacks == []  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert all(check in host.lists["callbacks"] for check in other.optional_callbacks)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    other.retire()


def test_runtime_removal_drops_the_host_callback_though_a_router_on_another_host_uses_that_check(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Another host's router keeps its own check on its own host; it does not hold this host's."""
    collect_other_routers()
    other_host: Final = CallbackHost()
    from tests.unit.test_router.core.scripted_host import python_core

    other: Final = python_core(
        other_host, model_list=[deployment("g", "o")], optional_pre_call_checks=["prompt_caching"]
    )
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core.apply(ChangeSet(settings={"optional_pre_call_checks": []}))
    assert core.optional_callbacks == []  # pyright: ignore[reportAttributeAccessIssue]
    assert host_callback_names(host)["callbacks"] == []
    assert host_callback_names(other_host)["callbacks"] == ["PromptCachingDeploymentCheck"]
    other.retire()


def test_enable_pre_call_checks_is_not_a_runtime_updatable_setting(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    core.apply(ChangeSet(settings={"enable_pre_call_checks": True}))
    assert core.settings().enable_pre_call_checks is False


def test_enable_pre_call_checks_can_be_assigned(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    core.apply(ChangeSet(assign={"enable_pre_call_checks": True}))
    assert core.settings().enable_pre_call_checks is True


# --- retire ------------------------------------------------------------------------------


def test_retire_removes_selectors_optional_checks_and_alerting_but_keeps_foreign_callbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    host.lists["callbacks"].append("foreign")
    host.lists["success_callback"].append("foreign-success")
    core: Final = _one(
        make_core,
        routing_strategy="least-busy",
        optional_pre_call_checks=["enforce_model_rate_limits"],
        alerting_config=AlertingConfig(webhook_url="https://example.invalid/hook"),
    )
    core.retire()
    assert {name: callbacks for name, callbacks in host.lists.items() if callbacks} == {
        "callbacks": ["foreign"],
        "success_callback": ["foreign-success"],
    }


def test_retire_twice_is_harmless(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one(make_core)
    core.retire()
    core.retire()
    assert {name: callbacks for name, callbacks in host.lists.items() if callbacks} == {}


def test_retiring_one_router_keeps_the_callbacks_of_another_on_the_same_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    first: Final = _one(make_core, routing_strategy="least-busy")
    second: Final = _one(make_core, routing_strategy="least-busy")
    first.retire()
    assert host.lists["callbacks"] == [second.leastbusy_logger]  # pyright: ignore[reportAttributeAccessIssue]
    assert host.lists["input_callback"] == [second.leastbusy_logger]  # pyright: ignore[reportAttributeAccessIssue]
    assert host_callback_names(host)["_async_success_callback"] == ["deployment_callback_on_success"]
    assert all(callback.__self__ is second for callback in host.lists["_async_success_callback"])  # pyright: ignore[reportAttributeAccessIssue]


def test_retire_never_hands_a_vacated_check_to_a_live_router_on_a_different_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Another host's router keeps its check on its own host; this host's list is left empty."""
    collect_other_routers()
    from tests.unit.test_router.core.scripted_host import python_core

    other_host: Final = CallbackHost()
    other: Final = python_core(
        other_host, model_list=[deployment("g", "o")], optional_pre_call_checks=["prompt_caching"]
    )
    core: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    core.retire()
    assert host.lists["callbacks"] == []
    assert other_host.lists["callbacks"] == other.optional_callbacks  # pyright: ignore[reportUnknownMemberType]  # untyped in the checker's view
    other.retire()


def test_retire_hands_a_vacated_check_to_another_live_router_on_the_same_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    first: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    second: Final = _one(make_core, optional_pre_call_checks=["prompt_caching"])
    first.retire()
    assert host.lists["callbacks"] == second.optional_callbacks  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    second.retire()


# --- observe -----------------------------------------------------------------------------


class RecordingLogging:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def add_dynamic_function_callbacks(self, **callbacks: object) -> None:  # kwargs-ok: the logging object's API
        self.calls.append(("functions", sorted(callbacks)))

    def replace_global_callback(self, callback: object) -> None:
        self.calls.append(("replace", type(callback).__name__))

    def add_dynamic_callback(self, callback: object) -> None:
        self.calls.append(("add", type(callback).__name__))


def _bound_outcome_callbacks(core: RouterCore) -> list[tuple[str, object]]:  # mutable-ok: a fresh list
    return (
        [("functions", ["async_failure", "async_success", "failure"])]
        if records_outcomes_through_callbacks(core)
        else []
    )


def test_observe_request_scoped_removes_every_callback_the_router_registered_globally(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    core: Final = _one(make_core, routing_strategy="least-busy", optional_pre_call_checks=["prompt_caching"])
    observers: Final = core.observe(request_scoped=True)
    assert observers.slack_alerting_logger is None
    assert {name: callbacks for name, callbacks in host.lists.items() if callbacks} == {}


def test_observe_with_a_logging_object_binds_the_router_outcome_callbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The Python core binds its three outcome callbacks; the Rust core records every attempt's
    outcome itself and binds none, as it registers none on the host."""
    core: Final = _one(make_core)
    before: Final = host_callback_names(host)
    logging_obj: Final = RecordingLogging()
    core.observe(logging_obj)  # pyright: ignore[reportArgumentType]
    assert logging_obj.calls == _bound_outcome_callbacks(core)
    assert host_callback_names(host) == before


def test_observe_request_scoped_then_binding_hands_selectors_and_policies_to_the_request(make_core: MakeCore) -> None:
    collect_other_routers()
    core: Final = _one(make_core, routing_strategy="least-busy", optional_pre_call_checks=["prompt_caching"])
    core.observe(request_scoped=True)
    logging_obj: Final = RecordingLogging()
    core.observe(logging_obj)  # pyright: ignore[reportArgumentType]
    assert logging_obj.calls == [
        *_bound_outcome_callbacks(core),
        ("replace", "LeastBusyLoggingHandler"),
        ("add", "PromptCachingDeploymentCheck"),
    ]


def test_binding_without_request_scoping_hands_the_request_no_selector_or_policy(make_core: MakeCore) -> None:
    """Without ``observe(request_scoped=True)`` the selectors and checks stay on the host lists, so a
    bound request gets none of them."""
    collect_other_routers()
    core: Final = _one(make_core, routing_strategy="least-busy", optional_pre_call_checks=["prompt_caching"])
    logging_obj: Final = RecordingLogging()
    core.observe(logging_obj)  # pyright: ignore[reportCallIssue]  # observe takes a logging object on every core
    assert logging_obj.calls == _bound_outcome_callbacks(core)


def test_binding_skips_a_policy_the_host_already_runs(make_core: MakeCore, host: ScriptedHost) -> None:
    """A request-scoped policy whose key is back on the host's ``callbacks`` list (here re-added by
    hand) is not added to the request again; the selector still replaces the host's."""
    collect_other_routers()
    core: Final = _one(make_core, routing_strategy="least-busy", optional_pre_call_checks=["prompt_caching"])
    core.observe(request_scoped=True)  # pyright: ignore[reportCallIssue]  # see above
    host.lists["callbacks"].append(PromptCachingDeploymentCheck(cache=DualCache()))
    logging_obj: Final = RecordingLogging()
    core.observe(logging_obj)  # pyright: ignore[reportCallIssue]  # observe takes a logging object on every core
    assert logging_obj.calls == [*_bound_outcome_callbacks(core), ("replace", "LeastBusyLoggingHandler")]


# --- register_plugin ---------------------------------------------------------------------


class PickB(CustomRoutingStrategyBase):
    def __init__(self) -> None:
        self.asked: list[str] = []

    @staticmethod
    def _b() -> dict[str, object]:
        return deployment("g", "b")

    async def async_get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool = False,
        request_kwargs: object = None,
    ) -> dict[str, object]:
        self.asked.append(model)
        return self._b()

    def get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: str,
        messages: object = None,
        input: object = None,
        specific_deployment: bool = False,
        request_kwargs: object = None,
    ) -> dict[str, object]:
        self.asked.append(model)
        return self._b()


@pytest.mark.asyncio
async def test_a_registered_routing_plugin_picks_for_select(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    plugin: Final = PickB()
    core.register_plugin(plugin)
    picked: Final = {(await core.select("g"))["model_info"]["id"] for _ in range(5)}  # pyright: ignore[reportIndexIssue]
    assert picked == {"b"}
    assert plugin.asked == ["g"] * 5


@pytest.mark.asyncio
async def test_a_registered_routing_plugin_picks_for_async_and_sync_endpoints(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    core.register_plugin(PickB())
    for _ in range(3):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) == {"b"}


@pytest.mark.asyncio
async def test_changing_the_routing_strategy_drops_a_registered_routing_plugin(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    plugin: Final = PickB()
    core.register_plugin(plugin)
    core.apply(ChangeSet(settings={"routing_strategy": "least-busy"}))
    for _ in range(5):
        await core.select("g")
    assert plugin.asked == []


# --- flush_cache / reset ------------------------------------------------------------------


def test_flush_cache_drops_the_response_cache_routing_state_and_prompt_caching_pins(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core)
    host.cache = object()
    core.health().store.set_cache("k", 1)
    core.registry().plugins("prompt_caching_pins").set_cache("p", 1)
    before: Final = host_callback_names(host)
    core.flush_cache()  # pyright: ignore[reportAttributeAccessIssue]
    assert host.cache is None
    assert core.health().store.get_cache("k") is None
    assert core.registry().plugins("prompt_caching_pins").get_cache("p") is None
    assert host_callback_names(host) == before


def test_reset_drops_the_routers_own_success_and_failure_callbacks_and_the_retry_policy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """reset drops only this router's entries from the host's success/failure lists (other code's
    callbacks stay) and leaves ``callbacks`` and ``input_callback`` alone."""
    core: Final = _one(make_core, retry_policy={"RateLimitErrorRetries": 3}, routing_strategy="least-busy")
    host.lists["success_callback"].append("foreign")
    host.lists["_async_failure_callback"].append("foreign-failure")
    core.health().store.set_cache("k", 1)
    core.reset()  # pyright: ignore[reportAttributeAccessIssue]
    assert {name: callbacks for name, callbacks in host_callback_names(host).items() if callbacks} == {
        "callbacks": ["LeastBusyLoggingHandler"],
        "input_callback": ["LeastBusyLoggingHandler"],
        "success_callback": ["str"],
        "_async_failure_callback": ["str"],
    }
    assert (host.lists["success_callback"], host.lists["_async_failure_callback"]) == (["foreign"], ["foreign-failure"])
    assert core.retry_policy is None  # pyright: ignore[reportAttributeAccessIssue]
    assert core.health().store.get_cache("k") is None


def test_resetting_one_router_keeps_another_routers_success_and_failure_callbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    collect_other_routers()
    first: Final = _one(make_core)
    before_first: Final = {name: list(callbacks) for name, callbacks in host.lists.items()}
    second: Final = _one(make_core)
    first.reset()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    for name in ("success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback"):
        remaining = host.lists.get(name, [])
        assert all(getattr(callback, "__self__", None) is not first for callback in remaining)
        theirs = [callback for callback in before_first.get(name, []) if getattr(callback, "__self__", None) is first]
        assert len(remaining) == len(theirs) and all(callback.__self__ is second for callback in remaining)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


# --- attribute writes ----------------------------------------------------------------------


def test_assigning_num_retries_after_construction_is_reported_by_settings(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    core.num_retries = 7  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings().as_dict()["num_retries"] == 7


def test_assigning_a_setting_attribute_replaces_the_settings_snapshot(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    before: Final = core.settings()
    core.enable_pre_call_checks = True  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings() is not before
    assert core.settings().enable_pre_call_checks is True
    assert before.enable_pre_call_checks is False


def test_assigning_a_plain_setting_keeps_the_settings_snapshot(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    before: Final = core.settings()
    core.num_retries = 7  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings() is before


def test_assigning_an_unrelated_attribute_keeps_the_settings_snapshot(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    before: Final = core.settings()
    core.some_custom_attribute = 1  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings() is before
    assert core.some_custom_attribute == 1  # pyright: ignore[reportAttributeAccessIssue]


def test_assigning_a_new_model_list_advances_the_registry_version_once(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    version: Final = core.registry().version
    core.model_list = [deployment("g", "z")]  # pyright: ignore[reportAttributeAccessIssue]
    assert core.registry().version == version + 1
    assert [d["model_info"]["id"] for d in core.registry().all()] == ["z"]


def test_deleting_a_setting_attribute_makes_it_unset_in_settings(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    del core.routing_strategy  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings().routing_strategy is None
    assert core.settings().is_set("routing_strategy") is False


def test_registry_settings_and_health_views_are_built_once(make_core: MakeCore) -> None:
    core: Final = _one(make_core)
    assert core.registry() is core.registry()
    assert core.settings() is core.settings()
    assert core.health() is core.health()


# --- get_valid_args -----------------------------------------------------------------------


def test_get_valid_args_lists_every_constructor_argument_but_self_and_the_host() -> None:
    names: Final = PythonCore.get_valid_args()
    assert names[0] == "model_list"
    assert "self" not in names
    assert "_host" not in names
    assert {"routing_strategy", "optional_pre_call_checks", "enable_pre_call_checks", "redis_url"} <= set(names)
    assert len(names) == len(set(names))


def test_get_valid_args_matches_the_router_init_args() -> None:
    assert tuple(PythonCore.get_valid_args()) == tuple(router_init_args())


def test_get_valid_args_on_a_built_router(make_core: MakeCore) -> None:
    assert _one(make_core).get_valid_args() == PythonCore.get_valid_args()  # pyright: ignore[reportAttributeAccessIssue]


# --- print_deployment ----------------------------------------------------------------------

SECRET_DEPLOYMENT: Final = {
    "model_name": "g",
    "litellm_params": {
        "model": "openai/x",
        "api_key": "sk-abcdef",
        "api_base": "https://h",
        "aws_secret_access_key": "secretvalue",
    },
    "model_info": {"id": "x"},
}


def test_print_deployment_shows_two_characters_of_the_api_key_then_ten_stars(make_core: MakeCore) -> None:
    printed: Final = _one(make_core).print_deployment(SECRET_DEPLOYMENT)  # pyright: ignore[reportAttributeAccessIssue]
    assert printed["litellm_params"] == {
        "model": "openai/x",
        "api_key": "sk**********",
        "api_base": "https://h",
        "aws_secret_access_key": "secretvalue",
    }
    assert printed["model_info"] == {"id": "x"}


def test_print_deployment_leaves_the_given_deployment_unchanged(make_core: MakeCore) -> None:
    _one(make_core).print_deployment(SECRET_DEPLOYMENT)  # pyright: ignore[reportAttributeAccessIssue]
    assert SECRET_DEPLOYMENT["litellm_params"]["api_key"] == "sk-abcdef"  # pyright: ignore[reportIndexIssue]


def test_print_deployment_without_an_api_key_returns_an_equal_copy(make_core: MakeCore) -> None:
    given: Final = {"litellm_params": {"model": "m"}}
    printed: Final = _one(make_core).print_deployment(given)  # pyright: ignore[reportAttributeAccessIssue]
    assert printed == given
    assert printed is not given


def test_print_deployment_without_litellm_params_raises_key_error(make_core: MakeCore) -> None:
    with pytest.raises(KeyError, match="litellm_params"):
        _one(make_core).print_deployment({"model_name": "x"})  # pyright: ignore[reportAttributeAccessIssue]


def test_print_deployment_with_redaction_masks_every_secret_keeping_its_length(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one(make_core)
    host.set_dynamic(redact_user_api_key_info=True)
    printed: Final = core.print_deployment(SECRET_DEPLOYMENT)  # pyright: ignore[reportAttributeAccessIssue]
    assert printed["litellm_params"] == {
        "model": "openai/x",
        "api_key": "sk*******",
        "api_base": "https://h",
        "aws_secret_access_key": "se*********",
    }


def test_settings_snapshot_reads_constructor_mappings_as_given(make_core: MakeCore) -> None:
    alias: Final[Mapping[str, object]] = {"alias": "g"}
    core: Final = _one(make_core, model_group_alias=dict(alias))
    assert core.registry().aliases() == alias

"""
The Rust core's settings semantics on the host side: how ``ChangeSet.settings`` is read (the rules
of ``update_settings``), the ``settings().as_dict()`` view (``get_settings``), the router attributes
the Python core keeps as plain attributes, deleted settings, and the construction-time side effects
of the logging arguments.

The rules are the Python router core's, ported (never imported). Settings change only through a
changeset; a direct attribute write is turned into one by ``RustCore.__setattr__``.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.constants import RUNTIME_UPDATABLE_ROUTER_SETTINGS
from litellm.types.router import RetryPolicy

if TYPE_CHECKING:
    from litellm.router_config import RouterConfig
    from litellm.router_python_host import PythonHost

MISSING: Final = object()

# `update_settings` casts these with `int(value)`.
_INT_SETTINGS: Final = frozenset({"timeout", "num_retries", "retry_after", "allowed_fails", "cooldown_time"})
CONTINUATION_SETTING: Final = "enable_mid_stream_fallback_continuation"

# Settings the Python core keeps as plain router attributes and `update_settings` sets as given,
# with their constructor defaults. ``max_retries`` has none: it exists once a changeset sets it.
PLAIN_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "retry_policy": None,
        "model_group_retry_policy": {},
        "enable_weighted_failover": False,
        "fallback_on_unsupported_params": True,
    }
)
# `get_settings`: the settings it lists, in its order (`routing_groups` comes last).
_VIEW_ORDER: Final = (
    "routing_strategy_args",
    "routing_strategy",
    "allowed_fails",
    "cooldown_time",
    "num_retries",
    "timeout",
    "max_retries",
    "retry_after",
    "fallbacks",
    "context_window_fallbacks",
    "model_group_retry_policy",
    "retry_policy",
    "model_group_alias",
    "enable_weighted_failover",
    "enable_mid_stream_fallback_continuation",
    "enable_tag_filtering",
    "tag_routing_prefix",
    "fallback_on_unsupported_params",
)

# Router attributes that are their constructor argument as given, else this default. Arguments
# the Rust core declines are only ever at their default here.
_ARGUMENT_ATTRIBUTES: Final[Mapping[str, object]] = MappingProxyType(
    {
        "set_verbose": False,
        "debug_level": "INFO",
        "client_ttl": 3600,
        "stream_timeout": None,
        "assistants_config": None,
        "ignore_invalid_deployments": False,
        "alerting_config": None,
        "auto_router_capability_limit": None,
        "deployment_affinity_ttl_seconds": 3600,
        "session_affinity_repin_on_failover": False,
        "model_group_affinity_config": None,
    }
)
# Class-level and constant attributes of the Python core.
_CONSTANT_ATTRIBUTES: Final[Mapping[str, object]] = MappingProxyType(
    {"default_cache_time_seconds": 1 * 60 * 60, "tenacity": None, "access_groups": None}
)


@dataclass(frozen=True)
class SettingsUpdate:
    """What a ``ChangeSet.settings`` mapping changes, read with ``update_settings``' rules."""

    values: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))  # fields and plain settings
    plain: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))  # plain attributes, as stored
    continuation: object = MISSING  # the continuation flag, when given
    routing_groups: object = MISSING  # the routing groups input, when given
    optional_pre_call_checks: object = MISSING  # the optional pre-call checks, when given


def read_settings_update(settings: Mapping[str, object]) -> SettingsUpdate:
    """``_core_update_settings``' reading of each setting: names that are not runtime-updatable are
    ignored, the integer settings are cast with ``int``, a retry policy dict becomes a
    ``RetryPolicy`` (any other type but ``None`` is dropped)."""
    values: Final[dict[str, object]] = {}  # mutable-ok: built here, then frozen into the update
    plain: Final[dict[str, object]] = {}  # mutable-ok: same
    continuation: object = MISSING  # rebind-ok: set when the flag is given
    routing_groups: object = MISSING  # rebind-ok: set when routing groups are given
    optional_checks: object = MISSING  # rebind-ok: set when the optional pre-call checks are given
    for name, value in settings.items():
        if name not in RUNTIME_UPDATABLE_ROUTER_SETTINGS:
            verbose_router_logger.debug("Setting %s is not allowed", name)
        elif name in _INT_SETTINGS:
            values[name] = int(value)  # pyright: ignore[reportArgumentType]  # `int(kwargs[var])`, as given: a bad value raises int's own error
        elif name == "routing_groups":
            routing_groups = value
        elif name == "optional_pre_call_checks":
            optional_checks = value
        elif name == CONTINUATION_SETTING:
            continuation = value
        elif name == "retry_policy":
            policy = RetryPolicy.model_validate(value) if isinstance(value, dict) else value
            if policy is None or isinstance(policy, RetryPolicy):
                plain[name] = policy
        elif name in PLAIN_DEFAULTS or name == "max_retries":
            plain[name] = value
        else:
            values[name] = value
    return SettingsUpdate(
        values=MappingProxyType(values),
        plain=MappingProxyType(plain),
        continuation=continuation,
        routing_groups=routing_groups,
        optional_pre_call_checks=optional_checks,
    )


def initial_plain(config: "RouterConfig") -> dict[str, object]:  # mutable-ok: the core's own plain attributes
    """The plain attributes as the constructor sets them: a ``retry_policy`` dict is read into a
    ``RetryPolicy`` and any other type but a policy is no policy."""
    from litellm.router_rust_retries import resolved_retry_policy

    plain: Final = {name: config.get(name, default) for name, default in PLAIN_DEFAULTS.items()}
    plain["retry_policy"] = resolved_retry_policy(plain["retry_policy"])
    return plain


def settings_view(
    values: Mapping[str, object], plain: Mapping[str, object], deleted: frozenset[str]
) -> dict[str, object]:  # mutable-ok: the settings view is built fresh per call, as `get_settings` builds it
    """``get_settings``: the listed settings the router has, then the routing groups. ``values``
    holds the settings fields and plain attributes the Rust core resolves."""
    view: Final[dict[str, object]] = {}  # mutable-ok: see above
    for name in _VIEW_ORDER:
        if name in deleted:
            continue
        if name in plain:
            view[name] = plain[name]
        elif name in values:
            view[name] = values[name]
    view["routing_groups"] = values.get("routing_groups", [])
    return view


def argument_attribute(config: "RouterConfig", name: str) -> object:
    """A router attribute that is its constructor argument (or a constant), else ``MISSING``."""
    if name in _ARGUMENT_ATTRIBUTES:
        return config.get(name, _ARGUMENT_ATTRIBUTES[name])
    if name in _CONSTANT_ATTRIBUTES:
        return _CONSTANT_ATTRIBUTES[name]
    return MISSING


def default_litellm_params(config: "RouterConfig") -> dict[str, object]:  # mutable-ok: the router's own dict
    """``default_litellm_params`` as the constructor completes it: the router timeout, no client
    retries and the caching groups."""
    given: Final = _as_dict(config.get("default_litellm_params")) or {}
    params: Final[dict[str, object]] = dict(given)  # mutable-ok: see above
    params.setdefault("timeout", config.get("timeout"))
    params.setdefault("max_retries", 0)
    metadata: Final = _as_dict(params.setdefault("metadata", {}))
    if metadata is not None:
        params["metadata"] = {**metadata, "caching_groups": config.get("caching_groups")}
    return params


def _as_dict(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]  # the argument dict, as passed


def request_timeout(config: "RouterConfig", host: "PythonHost") -> float | None:
    """The per-attempt ``request_timeout``: the host's configured process ``request_timeout``, only
    stored when a router timeout is also set."""
    from litellm.litellm_core_utils.request_timeout_resolver import configured_request_timeout

    if config.get("timeout") is None:
        return None
    dynamic: Final = host.dynamic()
    return configured_request_timeout(dynamic.request_timeout, dynamic.request_timeout_explicitly_set)


def apply_construction_effects(config: "RouterConfig", host: "PythonHost") -> None:
    """What the constructor's logging arguments do: the host stops its debug info, and
    ``set_verbose`` sets the router logger to ``debug_level``."""
    host.suppress_debug_info()
    if config.get("set_verbose", False) is True:
        level: Final = config.get("debug_level", "INFO")
        if level == "INFO":
            verbose_router_logger.setLevel(logging.INFO)
        elif level == "DEBUG":
            verbose_router_logger.setLevel(logging.DEBUG)


__all__ = [
    "CONTINUATION_SETTING",
    "MISSING",
    "PLAIN_DEFAULTS",
    "SettingsUpdate",
    "apply_construction_effects",
    "argument_attribute",
    "default_litellm_params",
    "initial_plain",
    "read_settings_update",
    "request_timeout",
    "settings_view",
]

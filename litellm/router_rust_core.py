"""
``RustCore``: the router core served by litellm-rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns for the operations in
``RUST_SERVED_OPS``. It never awaits Python: ``RouterCall.next`` returns the step the call loop is
blocked on (run an attempt on a deployment, sleep, check a fallback target, run a plugin) and this
driver answers it. Each attempt runs through the Python core with that one deployment, so request
preparation, inference, logging and cost tracking are unchanged.

Everything else (the views, runtime writes, every other endpoint) is served by the ``PythonCore``
the Rust core wraps, built from the same config. A config the Rust core cannot serve fails at
construction and ``make_core`` builds a ``PythonCore`` instead.
"""

import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import (
    format_content_policy_fallback_hint,
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    truncate_fallback_error_detail,
)
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_config import RouterConfig
from litellm.router_python_core import PythonCore
from litellm.router_python_host import PythonHost
from litellm.router_utils.fallback_event_handlers import (
    AttemptedFallbackTargets,
    _provider_prefixed_model_group,  # pyright: ignore[reportPrivateUsage]  # the Python fallback walk's own spelling rule
    fallback_lookup_groups,
    get_fallback_model_group_for_lookup_groups,
    is_bad_request_without_fallback,
)
from litellm.types.router import FallbackSkipReason
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.rust_bridge._native import RouterCall, RouterCoreHandle

RustServedOp: TypeAlias = Literal["acompletion", "aanthropic_messages", "aembedding", "aresponses"]
RUST_SERVED_OPS: Final[frozenset[str]] = frozenset({"acompletion", "aanthropic_messages", "aembedding", "aresponses"})

_FAILURE_KINDS: Final = (
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GuardrailIntervention",
    "CallerCredentialError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_STEP: Final = TypeAdapter(Mapping[str, object])
_REQUEST_FIELDS: Final = ("specific_deployment", "num_retries", "disable_fallbacks")
# Request keys router-core reads under another name.
_RENAMED_REQUEST_FIELDS: Final = MappingProxyType({"_router_weights": "router_weights"})
_PROVIDERS: Final = tuple(provider.value for provider in LlmProviders)
_PYTHON_ONLY_REQUEST_KEYS: Final = frozenset(
    {
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "max_fallbacks",
        "model_group_retry_policy",
        "retry_policy",
    }
)
_STATIC_FIELDS: Final = (
    "routing_strategy",
    "cooldown_time",
    "disable_cooldowns",
    "allowed_fails_policy",
    "retry_after",
    "timeout",
)
_HANDLED_FIELDS: Final = frozenset(
    {
        *_STATIC_FIELDS,
        "model_list",
        "model_group_alias",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "set_verbose",
        "debug_level",
    }
)
_PYTHON_DEFAULTS: Final = MappingProxyType(
    {
        name: parameter.default
        for name, parameter in inspect.signature(PythonCore).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
)


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    if is_guardrail_intervention(error):  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
        return "GuardrailIntervention"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def _retry_after_seconds(error: BaseException) -> float | None:
    response: Final = getattr(error, "response", None)
    headers: Final = getattr(error, "litellm_response_headers", None) or getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    milliseconds: Final = headers.get("retry-after-ms")
    seconds: Final = headers.get("retry-after")
    try:
        value: Final = float(milliseconds) / 1000 if milliseconds is not None else float(seconds)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "retry_after": _retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "detail": str(error)[:500],
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


def _json_default(value: object) -> object:
    return str(value)


def _plain(value: object) -> object:
    """``value`` as plain JSON values, the shape the Rust core reads."""
    return json.loads(json.dumps(value, default=_json_default))


def _plain_document(value: Mapping[str, object]) -> Mapping[str, object]:
    """A mapping as plain JSON values: still a mapping, with string keys."""
    return as_str_mapping(_plain(value)) or MappingProxyType({})


def _as_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()  # pyright: ignore[reportUnknownVariableType]  # a list's items are read as objects


def _applied_static(python: PythonCore, name: str) -> object:
    """Where an applied static setting lands on the Python core: the value the Rust core is rebuilt with."""
    match name:
        case "routing_strategy":
            return python.settings().routing_strategy
        case "cooldown_time":
            return python.settings().cooldown_time
        case "disable_cooldowns":
            return python.settings().disable_cooldowns
        case "allowed_fails_policy":
            return python.settings().allowed_fails_policy
        case "retry_after":
            return python.retry_after
        case "timeout":
            return python.timeout
        case _:
            raise KeyError(name)


def _static_value(value: object) -> object:
    dump: Final = getattr(value, "model_dump", None)
    return _plain(dump(exclude_none=True) if callable(dump) else value)


def static_settings(config: RouterConfig, python: PythonCore) -> Mapping[str, object]:
    """The static settings: the constructor arguments, overlaid with every setting a changeset
    (``apply``) has changed since, as the Python core resolved it."""
    applied: Final = python._core_applied  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
    values: Final[dict[str, object]] = {  # mutable-ok: the arguments, then the applied settings over them
        name: config.arguments[name] for name in _STATIC_FIELDS if name in config.arguments
    }
    values.update({name: _applied_static(python, name) for name in _STATIC_FIELDS if name in applied})
    return {name: _static_value(value) for name, value in values.items() if value is not None}


def static_document(config: RouterConfig, python: PythonCore) -> Mapping[str, object]:
    """The Python core's own normalized deployments (with their ids) and the static settings."""
    return {
        "model_list": _plain(python._core_model_list),  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownArgumentType]  # RustCore drives its own PythonCore, whose list is untyped
        **static_settings(config, python),
    }


def dynamic_document(python: PythonCore) -> Mapping[str, object]:
    """The values the Python core resolved from its arguments and the process settings."""
    settings: Final = python.settings()
    applied: Final = python._core_applied  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
    resolved: Final[Mapping[str, object]] = {  # pyright: ignore[reportUnknownVariableType]  # the fallback settings are untyped lists
        "version": python._registry_version,  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        "model_group_alias": python._core_model_group_alias,  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        "fallbacks": settings.fallbacks,  # pyright: ignore[reportUnknownMemberType]  # the setting's own untyped list
        "context_window_fallbacks": settings.context_window_fallbacks,  # pyright: ignore[reportUnknownMemberType]  # the setting's own untyped list
        "content_policy_fallbacks": settings.content_policy_fallbacks,  # pyright: ignore[reportUnknownMemberType]  # the setting's own untyped list
        "num_retries": python.num_retries,
        "max_fallbacks": python.max_fallbacks,
        "allowed_fails": python._core_allowed_fails  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        if "allowed_fails" in applied
        else python.explicit_allowed_fails,
        "fallback_keys": fallback_keys(python, settings),
    }
    return {name: _plain(value) for name, value in resolved.items() if value is not None}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # the fallback settings are untyped lists


def fallback_keys(python: PythonCore, settings: object) -> Mapping[str, object]:
    """The spellings a fallback key matches besides the group name (`get_fallback_model_group`)."""
    lists: Final = (
        getattr(settings, "fallbacks", None),
        getattr(settings, "context_window_fallbacks", None),
        getattr(settings, "content_policy_fallbacks", None),
    )
    keys: Final = tuple(  # comprehension-ok: the first group of each fallback entry, across the three lists
        str(next(iter(mapping)))
        for entries in lists
        for entry in _as_list(entries)
        if (mapping := as_str_mapping(entry)) and isinstance(mapping, dict)
    )
    names: Final = {
        *(python._core_model_names or ()),  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        *(python._core_model_group_alias or {}),  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
    }
    prefixed: Final = {
        name: spelled for name in names if (spelled := _provider_prefixed_model_group(name, keys)) is not None
    }
    return {"providers": _PROVIDERS, "provider_prefixed": prefixed}


def unsupported_arguments(config: RouterConfig) -> frozenset[str]:
    """Arguments the Rust core does not serve; any of them keeps the router on Python."""
    changed: Final = frozenset(
        name
        for name, value in config.arguments.items()
        if name not in _HANDLED_FIELDS and value != _PYTHON_DEFAULTS.get(name, None)
    )
    strategy: Final = config.arguments.get("routing_strategy", "simple-shuffle")
    model_list: Final = config.arguments.get("model_list")
    # `simple_shuffle` raises on a non-numeric weight when it picks; the Python core keeps that.
    odd_weight: Final = any(
        (params := as_str_mapping((as_str_mapping(deployment) or {}).get("litellm_params"))) is not None
        and params.get("weight") is not None
        and (isinstance(params.get("weight"), bool) or not isinstance(params.get("weight"), int | float))
        for deployment in _as_list(model_list)
    )
    return (
        changed
        | ({"routing_strategy"} if strategy != "simple-shuffle" else frozenset[str]())
        | ({"model_list"} if odd_weight else frozenset[str]())
    )


def runs_on_python(request: Mapping[str, object]) -> bool:
    """Per-request routing overrides the Rust core does not read yet."""
    return any(key in _PYTHON_ONLY_REQUEST_KEYS or key.startswith("mock_testing_") for key in request)


def request_team_id(request: Mapping[str, object]) -> str | None:
    metadata: Final = as_str_mapping(request.get("metadata") or request.get("litellm_metadata"))
    team_id: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
    return team_id if isinstance(team_id, str) else None


def _bound_request(bound: inspect.BoundArguments) -> Mapping[str, object]:
    """The keyword payload of a call bound to the Python core's endpoint signature."""
    named: Final = {name: value for name, value in bound.arguments.items() if name not in ("self", "kwargs")}
    extra: Final = bound.arguments.get("kwargs")
    return {**named, **(extra if isinstance(extra, Mapping) else {})}


_ACOMPLETION: Final = inspect.signature(PythonCore.acompletion)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # only the parameter names are read
_AEMBEDDING: Final = inspect.signature(PythonCore.aembedding)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # only the parameter names are read
_OWN_ATTRIBUTES: Final = frozenset({"_handle", "_config", "_python", "_host", "_built_for"})


class RustCoreDeclined(Exception):
    """The Rust core cannot serve a router whose Python core is already built: the router keeps that one."""

    def __init__(self, reason: str, python: PythonCore) -> None:
        super().__init__(reason)
        self.python = python


class RustCore:
    """A ``RouterCore`` whose routing for ``RUST_SERVED_OPS`` runs in Rust, over a ``PythonCore``."""

    def __init__(self, handle: "RouterCoreHandle", config: RouterConfig, python: PythonCore, host: PythonHost) -> None:
        self._handle: RouterCoreHandle | None = handle
        self._config: Final = config
        self._python: Final = python
        self._host: Final = host
        self._built_for: tuple[int, int] = self._state()

    @classmethod
    def build(cls, config: RouterConfig, host: PythonHost, python_class: type[PythonCore] = PythonCore) -> "RustCore":
        from litellm.rust_bridge._native import RouterCoreHandle, RustBridgeDeclined

        extra: Final = unsupported_arguments(config)
        if extra:
            raise RustBridgeDeclined(f"router arguments not served by the Rust core: {sorted(extra)}")
        python: Final = python_class(**config.arguments, _host=host)  # pyright: ignore[reportArgumentType]  # the arguments were bound against PythonCore's own signature
        try:
            handle: Final = RouterCoreHandle(static_document(config, python), dynamic_document(python))
        except Exception as declined:
            # The Python core is built already (its callbacks registered): keep it, don't build another.
            raise RustCoreDeclined(str(declined), python) from declined
        return cls(handle, config, python, host)

    def __getattr__(self, name: str) -> object:
        return getattr(self._python, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in _OWN_ATTRIBUTES:
            object.__setattr__(self, name, value)
            return
        setattr(self._python, name, value)

    def __delattr__(self, name: str) -> None:
        if name in _OWN_ATTRIBUTES:
            object.__delattr__(self, name)
            return
        delattr(self._python, name)

    def _state(self) -> tuple[int, int]:
        """What the handle was built from: the deployments, and the settings changesets applied."""
        return (self._python._registry_version, self._python._settings_version)  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore

    def _current_handle(self) -> "RouterCoreHandle | None":
        """The Rust handle, rebuilt after a deployment change or an applied changeset; ``None`` while
        the Rust core cannot serve the router."""
        if "_core_async_get_available_deployment" in vars(self._python):
            # `register_plugin` installed a custom strategy, which only the Python core runs.
            return None
        if self._state() == self._built_for:
            return self._handle
        from litellm.rust_bridge._native import RouterCoreHandle

        if self._handle is not None:
            self._handle.retire()
        strategy: Final = self._python.settings().routing_strategy
        try:
            if strategy not in (None, "simple-shuffle"):
                raise ValueError(f"routing_strategy {strategy!r} is not served by the Rust core")
            self._handle = RouterCoreHandle(static_document(self._config, self._python), dynamic_document(self._python))
        except Exception as declined:  # noqa: BLE001  # a change the Rust core cannot take keeps the router on Python until the next one
            verbose_router_logger.info("Router served by the Python core after a change: %s", declined)
            self._handle = None
        self._built_for = self._state()
        return self._handle

    async def call(self, op: str, request: Mapping[str, object]) -> object:
        if op not in RUST_SERVED_OPS:
            return await self._python.call(op, request)
        return await self._drive(op, request)

    async def acompletion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        try:
            bound: Final = _ACOMPLETION.bind(self._python, *args, **kwargs)
        except TypeError:
            # The Python core raises the caller's mistake in its own words.
            return await self._python.acompletion(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownMemberType, reportUnknownVariableType]  # the call is expected to raise
        return await self._drive("acompletion", _bound_request(bound))

    async def aanthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("aanthropic_messages", kwargs)

    async def aembedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        try:
            bound: Final = _AEMBEDDING.bind(self._python, *args, **kwargs)
        except TypeError:
            return await self._python.aembedding(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownMemberType]  # the call is expected to raise
        return await self._drive("aembedding", _bound_request(bound))

    async def aresponses(self, **kwargs: object) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("aresponses", kwargs)

    def retire(self) -> None:
        if self._handle is not None:
            self._handle.retire()
        self._python.retire()

    async def _drive(self, op: str, request: Mapping[str, object]) -> object:
        handle: Final = self._current_handle()
        if handle is None or runs_on_python(request) or not isinstance(request.get("model"), str):
            return await self._python.call(op, request)
        if not hasattr(request.get(_METADATA_FIELDS.get(op, "metadata"), {}), "update"):
            # `_update_kwargs_before_fallbacks` raises on metadata it cannot update, before any attempt.
            return await self._python.call(op, request)
        fields: Final = {
            **{key: request[key] for key in _REQUEST_FIELDS if request.get(key) is not None},
            **{name: request[key] for key, name in _RENAMED_REQUEST_FIELDS.items() if request.get(key) is not None},
        }
        team_id: Final = request_team_id(request)
        try:
            call: Final = handle.start(
                op, _plain_document({"model": request["model"], **fields, **({"team_id": team_id} if team_id else {})})
            )
        except ValueError:
            return await self._python.call(op, request)
        return await _Driver(call, self._python, op, request).run()


# A fallback chain list in the shape the Python core's fallback walk reads it.
_Chains: TypeAlias = Sequence[Mapping[str, list[str]]] | None

# The request field `_update_kwargs_before_fallbacks` writes the model group into, per op.
_METADATA_FIELDS: Final = MappingProxyType(
    {"aanthropic_messages": "litellm_metadata", "aresponses": "litellm_metadata"}
)


def _ttl(step: Mapping[str, object]) -> float | None:
    ttl: Final = step.get("ttl")
    return ttl if isinstance(ttl, int | float) else None


def _store_for(python: PythonCore, key: str) -> "DualCache":
    """Cooldowns live in the router's cooldown store; every other routing key in its main DualCache."""
    health: Final = python.health()
    return health.cooldowns.cooldown_store if key.endswith(":cooldown") else health.store


async def answer_cache_step(python: PythonCore, step: Mapping[str, object]) -> object:
    """Run one cache op the Rust core asked for against the router's own DualCaches (memory plus Redis)."""
    key: Final = str(step.get("key"))
    ttl: Final = _ttl(step)
    match step.get("op"):
        case "get":
            return _plain(await _store_for(python, key).async_get_cache(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
        case "batch_get":
            keys: Final = [str(k) for k in _as_list(step.get("keys"))]
            values: Final[list[object]] = [  # mutable-ok: built here, then made plain
                await _store_for(python, k).async_get_cache(k)  # pyright: ignore[reportUnknownMemberType]  # DualCache's reads are untyped
                for k in keys
            ]
            return _plain(values)
        case "set":
            store: Final = _store_for(python, key)
            if ttl is None:
                await store.async_set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
            else:
                await store.async_set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
            return None
        case "increment":
            amount: Final = step.get("value")
            by: Final = amount if isinstance(amount, int | float) and not isinstance(amount, bool) else 1
            counter: Final = _store_for(python, key)
            total: Final = (
                await counter.async_increment_cache(key, by)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                if ttl is None
                else await counter.async_increment_cache(key, by, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
            )
            return total if isinstance(total, int | float) else by
        case "delete":
            await _store_for(python, key).async_delete_cache(key)
            return None
        case other:
            raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")


@dataclass(slots=True)
class _Hop:
    """One target of a fallback walk and what its level ended on."""

    group: str
    pending: Mapping[str, object] | None
    error: BaseException | None = None
    failed: Mapping[str, object] | None = None
    children: "list[_Hop]" = field(default_factory=list)  # mutable-ok: the fallbacks tried from this target


class _Driver:
    """Answers the steps one Rust call loop blocks on until it is done.

    Each attempt runs on the Python core with retries and fallbacks off, so the driver puts back
    what Python's own retry and fallback layers add to the outcome: the retry stamp on an error
    whose retry loop ran out, the retry and fallback headers on a response, and the explanation
    the fallback layer appends to the error it finally raises.
    """

    def __init__(self, call: "RouterCall", python: PythonCore, op: str, request: Mapping[str, object]) -> None:
        self._call: Final = call
        self._python: Final = python
        self._op: Final = op
        self._request: Final = request
        self._last_response: object = None
        self._last_error: BaseException | None = None
        # The last failure at each fallback level: a level raises its own last error.
        self._level_errors: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call loop runs
        self._attempted: bool = False
        self._walk: Final = AttemptedFallbackTargets()
        # Fallback targets in the order Rust walked them, and the groups an attempt was made in.
        self._hop_targets: Final[list[str]] = []  # mutable-ok: filled as the call loop runs
        self._attempted_groups: Final[set[str]] = set()  # mutable-ok: filled as the call loop runs
        # The walk in order: ("hop", (level, group, order)), ("attempt", level), ("error", (level,
        # error, request)), ("sleep", None) before each retry, and ("exhausted", group) when a retry
        # loop ran out.
        self._timeline: Final[list[tuple[str, object]]] = []  # mutable-ok: filled as the call loop runs

    async def run(self) -> object:
        outcome: object = None  # rebind-ok: each step's outcome answers it
        while True:
            step = _STEP.validate_python(self._call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome against the step it answers
            self._take_events(step.get("events"))
            match step["kind"]:
                case "attempt":
                    outcome = await self._attempt(step)
                case "sleep":
                    self._timeline.append(("sleep", None))
                    await self._python._host.sleep(float(str(step["seconds"])))  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
                    outcome = None
                case "allow_fallback":
                    outcome = await self._allow_fallback(str(step["target"]))
                case "cache":
                    outcome = await answer_cache_step(self._python, step)
                case "select":
                    outcome = step["candidates"]
                case "done":
                    return self._with_headers(step)
                case "failed":
                    primary = self._level_errors.get(0)
                    if primary is not None:
                        if self._walks_order_tiers(primary, self._request.get("model")):
                            # Python's order-tier walk raises what its last fallback raised, as is.
                            raise await self._walk_error(self._walk_tree(), primary)
                        await self._fill_unattempted_level()
                        self._explain(primary)
                        raise primary
                    if not self._attempted:
                        # No deployment was tried (unknown model, nothing healthy): Python raises its own error.
                        return await self._python.call(self._op, self._request)
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _take_events(self, events: object) -> None:
        for raw_event in _as_list(events):
            event = as_str_mapping(raw_event)
            if event is not None and event.get("kind") == "fallback":
                self._timeline.append(("hop", (event.get("level"), event.get("to"), event.get("order"))))
            if event is None or event.get("kind") != "retries_exhausted":
                continue
            self._timeline.append(("exhausted", event.get("model_group")))
            error: BaseException | None = self._last_error
            # `async_function_with_retries` stamps the error its exhausted retry loop raises.
            if error is not None and type(error) in self._python._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
                error.max_retries = event.get("max")  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
                error.num_retries = event.get("attempted")  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes

    def _walks_order_tiers(self, error: BaseException, group: object) -> bool:
        """`async_function_with_fallbacks_common_utils` hands a group with more than one order level
        to `run_async_fallback` outside its try, so the walk's own error is raised with no suffix."""
        import litellm.exceptions as litellm_exceptions

        python: Final = self._python
        request: Final = self._request
        if (
            request.get("disable_fallbacks") is True
            or not isinstance(group, str)
            or is_guardrail_intervention(error)  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
            or is_bad_request_without_fallback(error, python.fallback_on_unsupported_params)  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
            or isinstance(
                error,
                litellm_exceptions.ContextWindowExceededError | litellm_exceptions.ContentPolicyViolationError,
            )
        ):
            return False
        metadata: Final = as_str_mapping(request.get("metadata"))
        team: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
        deployments: Final = (
            python._core_get_model_list(  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
                model_name=group, team_id=team if isinstance(team, str) else None
            )
            or ()
        )
        orders: Final = {
            order
            for deployment in deployments
            if (order := (deployment.get("litellm_params") or {}).get("order")) is not None  # pyright: ignore[reportUnknownMemberType]  # a deployment's untyped litellm_params
        }
        return len(orders) > 1

    def _walk_tree(self) -> _Hop:
        """The fallback walk as Rust ran it: each hop under the level it was tried from."""
        root: Final = _Hop(str(self._request.get("model")), None)
        current: Final[dict[int, _Hop]] = {0: root}  # mutable-ok: the newest hop at each level
        recent: _Hop | None = None  # rebind-ok: the hop whose attempt failed last
        for kind, value in self._timeline:
            if kind == "hop" and isinstance(value, tuple):
                level, group, order = value  # pyright: ignore[reportUnknownVariableType]  # the timeline's tuples are untyped
                hop = _Hop(str(group), {**self._request, "model": str(group)})  # pyright: ignore[reportUnknownArgumentType]  # the timeline's tuples are untyped
                if isinstance(order, int):
                    hop.pending = {**self._request, "model": str(group), "_target_order": order}  # pyright: ignore[reportUnknownArgumentType]  # the timeline's tuples are untyped
                parent = current.get(level - 1 if isinstance(level, int) else -1)
                if parent is not None and isinstance(level, int):
                    parent.children.append(hop)
                    current[level] = hop
                recent = None
            elif kind == "attempt" and isinstance(value, int) and value in current:
                current[value].pending = None
            elif kind == "error" and isinstance(value, tuple) and value[0] in current:
                recent = current[value[0]]
                recent.error, recent.failed = value[1], value[2]
            elif kind == "sleep" and recent is not None:
                recent.pending = recent.failed  # pyright: ignore[reportUnknownMemberType]  # the timeline's tuples are untyped
            elif kind == "exhausted" and recent is not None and recent.pending is recent.failed:  # pyright: ignore[reportUnknownMemberType]  # the timeline's tuples are untyped
                recent.pending = None
        return root

    async def _walk_error(self, hop: _Hop, primary: BaseException) -> BaseException:
        """The error a hop's level raises. Python raises its own error where Rust made no attempt:
        for a target with no deployment, and for a retry that found no deployment left to try. A
        group with more than one order level raises its walk's last error instead
        (`run_async_fallback`), whatever that walk tried."""
        own: Final = await self._own_error(hop, primary)
        if not hop.children or not self._walks_order_tiers(own, hop.group):
            return own
        last: BaseException = own  # rebind-ok: each fallback tried replaces it
        for child in hop.children:
            last = await self._walk_error(child, primary)
        return last

    async def _own_error(self, hop: _Hop, primary: BaseException) -> BaseException:
        """The error the hop's own target raised: Python's, replayed, when Rust made no attempt."""
        if hop.pending is not None:
            try:
                await self._python.call(self._op, {**hop.pending, "num_retries": 0, "disable_fallbacks": True})
            except Exception as error:  # noqa: BLE001  # the error Python's walk would end on
                return error
        return hop.error or primary

    async def _fill_unattempted_level(self) -> None:
        """A first fallback that resolved no deployment made no attempt, so Rust holds no error for
        it: ask Python, which raises its own no-deployment error without calling a provider."""
        if 1 in self._level_errors:
            return
        target: Final = next((t for t in self._hop_targets if t not in self._attempted_groups), None)
        if target is None:
            return
        try:
            await self._python.call(
                self._op, {**self._request, "model": target, "num_retries": 0, "disable_fallbacks": True}
            )
        except Exception as error:  # noqa: BLE001  # the error is only quoted in the primary error's message
            self._level_errors[1] = error

    def _with_headers(self, step: Mapping[str, object]) -> object:
        response: object = self._last_response  # rebind-ok: the response gains its headers in turn
        retries: Final = step.get("attempted_retries")
        if isinstance(retries, int) and retries > 0:
            max_retries: Final = step.get("max_retries")
            response = add_retry_headers_to_response(  # rebind-ok: the response gains its headers in turn
                response=response,
                attempted_retries=retries,
                max_retries=max_retries if isinstance(max_retries, int) else None,
            )
        fallbacks: Final = step.get("attempted_fallbacks")
        return add_fallback_headers_to_response(
            response=response, attempted_fallbacks=fallbacks if isinstance(fallbacks, int) else 0
        )

    def _explain(self, error: BaseException) -> None:
        """Append what `async_function_with_fallbacks_common_utils` appends to the error it raises."""
        python: Final = self._python
        request: Final = self._request
        if (
            request.get("disable_fallbacks") is True
            or is_guardrail_intervention(error)  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
            or is_bad_request_without_fallback(error, python.fallback_on_unsupported_params)  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
            or not hasattr(error, "message")
            or not python._host.dynamic().expose_router_debug_in_errors  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        ):
            return
        model_group: Final = request.get("model")
        model_name: Final = model_group if isinstance(model_group, str) else None
        try:
            self._explain_fallbacks(error, model_name)
        except Exception:  # noqa: BLE001  # Python looks the chains up inside its fallback try: a bad chain means none was attempted
            error.message += format_fallback_outcome_message(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the error Python raises carries the same suffix
                model_name, None, "", skipped=self._walk.skipped
            )

    def _explain_fallbacks(self, error: BaseException, model_name: str | None) -> None:
        import litellm.exceptions as litellm_exceptions

        python: Final = self._python
        request: Final = self._request
        fallbacks: Final[_Chains] = python._core_fallbacks  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType, reportAssignmentType]  # RustCore drives its own PythonCore, whose chains are untyped lists
        context_window_fallbacks: Final[_Chains] = python._core_context_window_fallbacks  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType, reportAssignmentType]  # RustCore drives its own PythonCore, whose chains are untyped lists
        content_policy_fallbacks: Final[_Chains] = python._core_content_policy_fallbacks  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType, reportAssignmentType]  # RustCore drives its own PythonCore, whose chains are untyped lists
        lookup_groups: Final = fallback_lookup_groups(request, model_name)
        context_window: Final = isinstance(error, litellm_exceptions.ContextWindowExceededError)
        content_policy: Final = isinstance(error, litellm_exceptions.ContentPolicyViolationError)
        attempted_group: Sequence[object] | None = None  # rebind-ok: set below
        orders: Final = sorted(
            {
                order
                for deployment in python._core_get_model_list(model_name=model_name) or ()  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
                if (order := (deployment.get("litellm_params") or {}).get("order")) is not None  # pyright: ignore[reportUnknownMemberType]  # a deployment's untyped litellm_params
            }
        )
        if len(orders) > 1 and not (context_window or content_policy):
            external, generic = (
                get_fallback_model_group_for_lookup_groups(fallbacks=fallbacks, lookup_groups=lookup_groups)
                if fallbacks is not None and lookup_groups
                else (None, None)
            )
            if external is None and generic is not None and fallbacks is not None:
                external = fallbacks[generic]["*"]  # rebind-ok: the generic chain when no specific one matched
            combined: Final = [{"model": model_name, "_target_order": o} for o in orders if o > orders[0]] + list(
                external or ()
            )
            if combined:
                attempted_group = combined  # rebind-ok: settled in the order Python's walk does
        if attempted_group is None:
            specific: Final = (
                context_window_fallbacks if context_window else content_policy_fallbacks if content_policy else None
            )
            if (context_window or content_policy) and specific is not None:
                attempted_group, _ = (  # rebind-ok: settled in the order Python's walk does
                    get_fallback_model_group_for_lookup_groups(fallbacks=specific, lookup_groups=lookup_groups)
                )
                if attempted_group is None:
                    # Python raises before its generic fallbacks: "No fallback was attempted".
                    error.message += format_fallback_outcome_message(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the error Python raises carries the same suffix
                        model_name, None, "", skipped=self._walk.skipped
                    )
                    return
            elif context_window:
                error.message += (  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the error Python raises carries the same suffix
                    f"\nmodel={model_name}. context_window_fallbacks={mask_sensitive_structure(context_window_fallbacks)}."
                    f" fallbacks={mask_sensitive_structure(fallbacks)}.\n\nSet 'context_window_fallback' -"
                    " https://docs.litellm.ai/docs/routing#fallbacks"
                )
            elif content_policy:
                error.message += (  # rebind-ok: the error Python raises carries the same suffix
                    f"\n{format_content_policy_fallback_hint(model_name, content_policy_fallbacks, fallbacks)}"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
                )
            if attempted_group is None and fallbacks is not None and lookup_groups:
                attempted_group, generic_index = (  # rebind-ok: settled in the order Python's walk does
                    get_fallback_model_group_for_lookup_groups(fallbacks=fallbacks, lookup_groups=lookup_groups)
                )
                if attempted_group is None and generic_index is not None:
                    attempted_group = fallbacks[generic_index][  # rebind-ok: settled in the order Python's walk does
                        "*"
                    ]
                if attempted_group is None:
                    error.message += format_no_fallback_group_message(lookup_groups, fallbacks)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the error Python raises carries the same suffix
                    return
        detail: Final = (
            truncate_fallback_error_detail(redact_string(str(self._level_errors.get(1) or error)))
            if attempted_group
            else ""
        )
        error.message += format_fallback_outcome_message(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above  # rebind-ok: the error Python raises carries the same suffix
            model_name, attempted_group, detail, skipped=self._walk.skipped
        )

    def _pinned_request(self, step: Mapping[str, object]) -> Mapping[str, object]:
        """The request as the Python router's own fallback walk would send it, narrowed to the one
        deployment Rust picked: the name Rust resolved, with every other deployment it resolved to
        skipped."""
        deployment_id: Final = str(step["deployment_id"])
        name: Final = str(step.get("model") or step["model_group"])
        candidates: Final = step.get("candidate_ids")
        others: Final = tuple(str(i) for i in _as_list(candidates) if str(i) != deployment_id)
        pinned: Final[dict[str, object]] = {  # mutable-ok: built here, then handed to the Python core as its request
            **self._request,
            "model": name,
        }
        if others:
            pinned["_retry_skipped_deployment_ids"] = others
        target_order: Final = step.get("target_order")
        if isinstance(target_order, int):
            pinned["_target_order"] = target_order
        return pinned

    async def _attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        self._attempted = True
        self._attempted_groups.add(str(step["model_group"]))
        self._timeline.append(("attempt", step.get("fallback_level")))
        request: Final = self._pinned_request(step)
        try:
            response: Final = await self._python.call(
                self._op, {**request, "num_retries": 0, "disable_fallbacks": True}
            )
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            self._last_error = error
            level: Final = step.get("fallback_level")
            self._level_errors[level if isinstance(level, int) else 0] = error
            self._timeline.append(("error", (level if isinstance(level, int) else 0, error, request)))
            return attempt_failure(error)
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response)}

    async def _allow_fallback(self, target: str) -> bool:
        """`_is_fallback_target_authorized` then `_is_fallback_target_within_budget`."""
        ctx: Final = self._python._core_ctx  # pyright: ignore[reportPrivateUsage]  # RustCore drives its own PythonCore
        settings: Final = ctx.settings()
        request: Final = dict(self._request)
        original: Final = self._request.get("model")
        if target == original:
            return True
        if settings.fallback_access_check is not None and not await ctx.fallback_access_allowed(
            model=target, request_kwargs=request
        ):
            self._walk.record_skip(target, FallbackSkipReason.NOT_AUTHORIZED)
            return False
        if settings.fallback_budget_check is not None and not await ctx.fallback_budget_allowed(
            model=target, request_kwargs=request
        ):
            self._walk.record_skip(target, FallbackSkipReason.OVER_BUDGET)
            return False
        self._hop_targets.append(target)
        return True

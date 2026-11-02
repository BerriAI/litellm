"""
The request metadata the Python router stamps as it walks retries and fallbacks, kept for a
``RustCore`` call. router-core decides the walk and reports it (each attempt's ``scope``,
``max_retries`` and ``model_group_size``, and the ``fallback`` and ``retry_logged`` events); this
module only holds what each level's metadata dict holds in the Python router and lays it over the
attempt's kwargs.

In the Python router one metadata dict serves a level's retries, and every fallback hop copies
its parent's (``run_async_fallback``'s ``_hop_metadata``), so a hop's breadcrumbs never reach
its parent. Each scope here is one such dict.
"""

from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping

if TYPE_CHECKING:
    from litellm.router_utils.fallback_event_handlers import AttemptedFallbackTargets

# `RETRY_BREADCRUMB_LIMIT`: how many failures `previous_models` keeps.
RETRY_BREADCRUMB_LIMIT: Final = 4

# The keys the walk stamps on each attempt's metadata.
_WALK_KEYS: Final = (
    "original_model_group",
    "attempted_fallbacks",
    "attempted_retries",
    "max_retries",
    "model_group_size",
    "previous_models",
    "request_retry_count",
)
# What a breadcrumb reads from the last attempt a scope made.
_SERVED_KEYS: Final = ("model_info", "deployment")
# The walk state `run_async_fallback` hands every hop as kwargs.
_HOP_KWARGS: Final = ("fallback_depth", "max_fallbacks", "attempted_targets", "include_fallback_errors")


class _Scope:
    __slots__ = ("kwargs", "metadata")

    def __init__(
        self,
        metadata: dict[str, object],  # mutable-ok: the level's metadata, stamped as the walk runs
        kwargs: Mapping[str, object],
    ) -> None:
        self.metadata: Final = metadata
        self.kwargs: Final = kwargs


def _breadcrumb(
    metadata: Mapping[str, object], model: str, error: BaseException, params_base_model: Callable[[str], str | None]
) -> dict[str, object]:  # mutable-ok: a fresh RetryAttemptRecord
    """``_core_log_retry``'s record of the failure."""
    from litellm.litellm_core_utils.get_litellm_params import get_base_model_from_litellm_call_metadata

    model_info: Final = as_str_mapping(metadata.get("model_info")) or {}
    deployment_id: Final = model_info.get("id")
    known_id: Final = deployment_id if isinstance(deployment_id, str) else None
    base_model: Final = get_base_model_from_litellm_call_metadata(metadata=metadata) or (
        params_base_model(known_id) if known_id is not None else None
    )
    deployment: Final = metadata.get("deployment")
    attempted: Final = metadata.get("attempted_retries")
    return {
        "model_group": model,
        "deployment_id": known_id,
        "model": deployment if isinstance(deployment, str) else None,
        "base_model": base_model,
        "exception_type": type(error).__name__,
        "exception_string": str(error),
        "attempted_retries": attempted if type(attempted) is int else None,
    }


def _kept_breadcrumbs(earlier: object) -> tuple[object, ...]:
    """The breadcrumbs a new one joins: the latest ``RETRY_BREADCRUMB_LIMIT - 1``."""
    if not isinstance(earlier, list | tuple):
        return ()
    stamped: Final[tuple[object, ...]] = tuple(earlier)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # the breadcrumbs as stamped
    return stamped[-(RETRY_BREADCRUMB_LIMIT - 1) :]


def log_retry_into(
    kwargs: dict[str, object],  # mutable-ok: the request kwargs whose metadata the record joins
    error: Exception,
    params_base_model: Callable[[str], str | None],
) -> dict[str, object]:  # mutable-ok: the same kwargs, as ``_core_log_retry`` returns them
    """
    ``_core_log_retry`` (``RoutingContext.log_retry``): the failure joins the request metadata's
    ``previous_models`` and counts toward ``request_retry_count``, with a ``litellm.routing.retry``
    phase event when a pre-routing strategy picked the group.
    """
    from litellm.integrations.otel.runtime import phase_event

    field: Final = "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"
    metadata: Final = kwargs[field]
    view: Final = as_str_mapping(metadata)
    if view is None:
        # Python reads it with ``.get``.
        raise AttributeError(f"{type(metadata).__name__!r} object has no attribute 'get'")
    model_group: Final = kwargs.get("model")
    record: Final = _breadcrumb(view, "", error, params_base_model)
    record["model_group"] = model_group if isinstance(model_group, str) else None
    deployment_id: Final = record["deployment_id"]
    breadcrumbs: Final = (*_kept_breadcrumbs(view.get("previous_models")), record)
    earlier: Final = view.get("request_retry_count")
    request_retry_count: Final = (earlier if type(earlier) is int and 0 <= earlier else 0) + 1
    decision: Final = view.get("routing_decision")
    router_name: Final = (as_str_mapping(decision) or {}).get("router_model_name")
    if isinstance(router_name, str):
        phase_event(
            "litellm.routing.retry",
            {
                "litellm.routing.router_model_name": router_name,
                "litellm.retry.count": request_retry_count,
                "error.type": type(error).__name__,
                **({"litellm.deployment.model_group": model_group} if isinstance(model_group, str) else {}),
                **({"litellm.deployment.id": deployment_id} if isinstance(deployment_id, str) else {}),
            },
        )
    metadata["previous_models"] = breadcrumbs  # pyright: ignore[reportIndexIssue]  # the logging object already holds this dict
    metadata["request_retry_count"] = request_retry_count  # pyright: ignore[reportIndexIssue]  # same dict, read by the cap
    return kwargs


class WalkMetadata:
    """The metadata each level of one call's walk holds, by scope."""

    __slots__ = ("_attempted", "_field", "_params_base_model", "_request", "_scopes")

    def __init__(
        self, request: Mapping[str, object], field: str, params_base_model: Callable[[str], str | None]
    ) -> None:
        model: Final = request.get("model")
        root: Final[dict[str, object]] = {"attempted_fallbacks": 0}  # mutable-ok: the root level's metadata
        if model is not None:
            root["original_model_group"] = model
        self._request: Final = request
        self._field: Final = field
        self._params_base_model: Final = params_base_model
        self._scopes: Final[dict[int, _Scope]] = {0: _Scope(root, {})}  # mutable-ok: filled as the walk hops
        self._attempted: AttemptedFallbackTargets | None = None

    def _scope(self, value: object) -> _Scope:
        found: Final = self._scopes.get(value) if isinstance(value, int) else None
        return found if found is not None else self._scopes[0]

    def targets(self) -> "AttemptedFallbackTargets":
        """The walk's ``attempted_targets``: the targets tried and the ones skipped, and why."""
        from litellm.router_utils.fallback_event_handlers import AttemptedFallbackTargets

        if self._attempted is None:
            self._attempted = AttemptedFallbackTargets()
        return self._attempted

    def hop(self, event: Mapping[str, object]) -> None:
        """A ``fallback`` event: the new level copies its parent's metadata and kwargs."""
        parent: Final = self._scope(event.get("parent"))
        attempted: Final = self.targets()
        for key in (event.get("from"), event.get("to")):
            if isinstance(key, str):
                attempted.record(key)
        metadata: Final = dict(parent.metadata)
        original: Final = metadata.pop("original_model_group", self._request.get("model"))
        metadata.pop("model_group", None)
        metadata.pop("attempted_fallbacks", None)
        metadata["original_model_group"] = original
        metadata["model_group"] = event.get("to")
        metadata["attempted_fallbacks"] = event.get("depth")
        kwargs: Final = {
            **parent.kwargs,
            "fallback_depth": event.get("depth"),
            "max_fallbacks": event.get("max_fallbacks"),
            "attempted_targets": attempted,
        }
        if self._request.get("include_fallback_errors") is True:
            kwargs["include_fallback_errors"] = True
        scope: Final = event.get("scope")
        if isinstance(scope, int):
            self._scopes[scope] = _Scope(metadata, kwargs)

    def hop_kwargs(self, event: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: a fresh dict for the loggers
        """The kwargs the level held after a hop (``kwargs["model"] = target``), as the
        fallback loggers see them."""
        scope: Final = self._scope(event.get("scope"))
        return {**self._request, **scope.kwargs, "model": event.get("to"), self._field: dict(scope.metadata)}

    def log_retry(self, event: Mapping[str, object], error: BaseException | None) -> None:
        """A ``retry_logged`` event: the failure joins the scope's ``previous_models``."""
        scope: Final = self._scope(event.get("scope"))
        metadata: Final = scope.metadata
        attempted: Final = event.get("attempted_retries")
        if isinstance(attempted, int):
            metadata["attempted_retries"] = attempted
        if error is None:
            return
        record: Final = _breadcrumb(metadata, str(event.get("model")), error, self._params_base_model)
        earlier: Final = metadata.get("previous_models")
        kept: Final = _kept_breadcrumbs(earlier)
        count: Final = metadata.get("request_retry_count")
        metadata["previous_models"] = (*kept, record)
        metadata["request_retry_count"] = (count if type(count) is int and count >= 0 else 0) + 1

    def request_retry_count(self, step: Mapping[str, object]) -> object:
        """The level's ``request_retry_count``, as the retry loop has counted it so far."""
        return self._scope(step.get("scope")).metadata.get("request_retry_count")

    def stamp(self, step: Mapping[str, object], kwargs: Mapping[str, object]) -> Mapping[str, object]:
        """The attempt's kwargs with its level's walk metadata and hop kwargs laid over them."""
        scope: Final = self._scope(step.get("scope"))
        metadata: Final = scope.metadata
        metadata["attempted_retries"] = step.get("retry")
        metadata["max_retries"] = step.get("max_retries")
        metadata["model_group_size"] = step.get("model_group_size")
        served: Final = dict(as_str_mapping(kwargs.get(self._field)) or {})
        served.update({key: metadata[key] for key in _WALK_KEYS if key in metadata})
        metadata.update({key: served.get(key) for key in _SERVED_KEYS})
        return {**kwargs, **{key: scope.kwargs[key] for key in _HOP_KWARGS if key in scope.kwargs}, self._field: served}


class FallbackOutcomes:
    """What ``run_async_fallback`` does when a hop returns: it logs the hop to the process's
    custom loggers and, under ``include_fallback_errors``, lists the level's errors (its own
    first, then each hop that failed) in the response's ``x-litellm-fallback-errors``."""

    __slots__ = ("_failed", "_header_errors", "_include", "_pending")

    def __init__(self, request: Mapping[str, object]) -> None:
        self._include: Final = request.get("include_fallback_errors", False) is True
        self._failed: Final[dict[object, list[object]]] = {}  # mutable-ok: each level's failed hops, in order
        self._header_errors: Final[list[list[object]]] = []  # mutable-ok: each successful hop's errors, innermost first
        self._pending: Final[list[Coroutine[object, object, None]]] = []  # mutable-ok: logger calls not yet awaited

    def done(
        self,
        event: Mapping[str, object],
        level_error: BaseException | None,
        hop_error: BaseException | None,
        kwargs: Mapping[str, object],
    ) -> None:
        from litellm.litellm_core_utils.add_retry_fallback_headers import get_fallback_error_info
        from litellm.router_utils import fallback_event_handlers as handlers

        if level_error is None or not isinstance(level_error, Exception):
            return
        parent: Final = event.get("parent")
        failed: Final = self._failed.setdefault(parent, [])
        original: Final = str(event.get("from"))
        if event.get("succeeded") is True:
            if self._include:
                self._header_errors.append([get_fallback_error_info(level_error), *failed])
            self._pending.append(
                handlers.log_success_fallback_event(  # pyright: ignore[reportUnknownMemberType]  # takes the kwargs as an untyped dict
                    original_model_group=original, kwargs=dict(kwargs), original_exception=level_error
                )
            )
            return
        if isinstance(hop_error, Exception):
            failed.append(get_fallback_error_info(hop_error))
        self._pending.append(
            handlers.log_failure_fallback_event(  # pyright: ignore[reportUnknownMemberType]  # see above
                original_model_group=original, kwargs=dict(kwargs), original_exception=level_error
            )
        )

    def take_pending(self) -> tuple[Coroutine[object, object, None], ...]:
        pending: Final = tuple(self._pending)
        self._pending.clear()
        return pending

    def add_headers(self, response: object, attempted_fallbacks: int) -> object:
        """``add_fallback_headers_to_response`` for each successful hop, innermost first."""
        from litellm.litellm_core_utils.add_retry_fallback_headers import add_fallback_headers_to_response

        headed: object = response  # rebind-ok: gains each level's errors
        for errors in self._header_errors:
            headed = add_fallback_headers_to_response(
                response=headed,
                attempted_fallbacks=attempted_fallbacks,
                fallback_errors=errors,  # pyright: ignore[reportArgumentType]  # FallbackErrorInfo dicts
            )
        return headed


async def await_all(pending: Sequence[Coroutine[object, object, None]]) -> None:
    for call in pending:
        await call


__all__ = ["RETRY_BREADCRUMB_LIMIT", "FallbackOutcomes", "WalkMetadata", "await_all"]

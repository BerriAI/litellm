"""
The retry members ``RustCore`` serves: ``should_retry_this_error`` and
``get_num_retries_from_retry_policy``. router-core decides both; this module only describes the
Python exception the way an attempt failure is described and reads the answer back.
"""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.types.router import RetryPolicy

if TYPE_CHECKING:
    from litellm.router_rust_core import Resolved
    from litellm.rust_bridge._native import (  # pyright: ignore[reportPrivateUsage]  # the stub's handle
        RouterCoreHandle,
    )


def resolved_retry_policy(value: object) -> RetryPolicy | None:
    """The router's ``retry_policy`` as the Python router's constructor keeps it: a dict is read
    into a ``RetryPolicy``, a ``RetryPolicy`` is kept, anything else is no policy."""
    if isinstance(value, RetryPolicy):
        return value
    if isinstance(value, Mapping):
        fields: Final[Mapping[str, int | None]] = value  # pyright: ignore[reportUnknownVariableType]  # the policy's fields
        return RetryPolicy(**fields)  # pyright: ignore[reportArgumentType]  # validated by the model
    return None


def _is_policy(entry: object) -> bool:
    return isinstance(entry, RetryPolicy | Mapping)


def _reads_as_policy(entry: object) -> bool:
    """Whether ``resolve_retry_policy`` reads ``entry`` without raising: no entry, a ``RetryPolicy``,
    or a mapping ``RetryPolicy(**entry)`` accepts."""
    if entry is None or isinstance(entry, RetryPolicy):
        return True
    if not isinstance(entry, Mapping):
        return False
    try:
        _ = RetryPolicy(**entry)  # pyright: ignore[reportUnknownArgumentType]  # validated by the model
    except Exception:  # noqa: BLE001  # any error the lookup would raise
        return False
    return True


def is_policy_per_group(policies: object) -> bool:
    """Whether ``model_group_retry_policy`` is what router-core's settings hold: a mapping of
    group names to policies (or nothing)."""
    if policies is None:
        return True
    if not isinstance(policies, Mapping):
        return False
    entries: Final[Mapping[object, object]] = policies  # pyright: ignore[reportUnknownVariableType]  # read as objects
    return all(isinstance(group, str) and _is_policy(entry) for group, entry in entries.items())


@dataclasses.dataclass(frozen=True, slots=True)
class GroupPolicies:
    """A ``model_group_retry_policy`` as router-core reads it, resolved the way
    ``resolve_retry_policy`` reads it for each group the call may evaluate.

    ``document`` holds a group's policy where the lookup finds one, and an empty policy (which
    grants nothing) where it finds ``None``; a group it does not find is absent, so the router's
    ``retry_policy`` applies. ``raising`` names the groups whose lookup raises (an ``in``/``[]``
    the value does not support, or an entry that is not a policy, whose fields
    ``get_num_retries_from_retry_policy`` then fails to read): the Python retry loop raises that
    error in place of the attempt's on a hop's first failure.
    """

    document: Mapping[str, object]
    raising: frozenset[str]


def _looked_up(policies: object, group: str) -> tuple[bool, object]:
    """``(found, entry)`` as ``resolve_retry_policy`` looks ``group`` up; raises as it raises."""
    found: Final = group in policies  # pyright: ignore[reportOperatorIssue]  # any value, as the Python router reads it
    return (True, policies[group]) if found else (False, None)  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]  # see above


def _policy_document(entry: object) -> object:
    """A readable entry as router-core reads it: no entry grants nothing, a mapping is read the way
    ``RetryPolicy(**entry)`` reads it."""
    if entry is None:
        return {}
    if isinstance(entry, Mapping):
        return RetryPolicy(**entry)  # pyright: ignore[reportUnknownArgumentType]  # checked by _reads_as_policy
    return entry


def group_policies(policies: object, groups: Iterable[str]) -> GroupPolicies:
    """``policies`` resolved for ``groups`` (a mapping is read by its own keys)."""
    if policies is None:
        return GroupPolicies({}, raising=frozenset())
    if isinstance(policies, Mapping):
        entries: Final[Mapping[object, object]] = policies  # pyright: ignore[reportUnknownVariableType]  # read as objects
        # A key that is no string never matches a group name.
        named: Final = {group: entry for group, entry in entries.items() if isinstance(group, str)}
        readable: Final = {group: entry for group, entry in named.items() if _reads_as_policy(entry)}
        return GroupPolicies(
            {group: _policy_document(entry) for group, entry in readable.items()},
            raising=frozenset(named) - frozenset(readable),
        )
    document: Final[dict[str, object]] = {}  # mutable-ok: built here; router-core reads it as plain JSON
    raising: Final[set[str]] = set()  # mutable-ok: the groups whose lookup raises
    for group in groups:
        try:
            found, entry = _looked_up(policies, group)
        except Exception:  # noqa: BLE001  # the Python retry loop raises it on a hop's first failure
            raising.add(group)
            continue
        if not found:
            continue
        if not _reads_as_policy(entry):
            raising.add(group)
            continue
        document[group] = _policy_document(entry)
    return GroupPolicies(document, raising=frozenset(raising))


def static_group_policies(policies: object) -> object:
    """The router's ``model_group_retry_policy`` as router-core's static settings hold it. A value
    that is not a mapping of policies is resolved per request instead (``group_policies``)."""
    return group_policies(policies, ()).document if isinstance(policies, Mapping) else None  # pyright: ignore[reportUnknownArgumentType]  # read as objects


_STREAM_POLICY_ENDPOINTS: Final = frozenset({"aanthropic_messages", "anthropic_messages"})


def raising_groups(
    endpoint: str, request: Mapping[str, object], policies: object, resolved: GroupPolicies
) -> Sequence[str]:
    """The groups whose ``model_group_retry_policy`` lookup raises on a hop's first failure, unless
    the retry loop never reads it: the request opted out with ``num_retries=0``, or it is a mapping
    on a streamed Anthropic messages call, which resolves such an entry to no policy at all
    (``_anthropic_messages_resolved_retry_policy``)."""
    if not resolved.raising or request.get("num_retries") == 0:
        return ()
    if endpoint in _STREAM_POLICY_ENDPOINTS and request.get("stream") is True and isinstance(policies, Mapping):
        return ()
    return sorted(resolved.raising)


def policy_lookup_error(original: BaseException, model_group: str, retry_policy: object, policies: object) -> Exception:
    """The error ``get_num_retries_from_retry_policy`` raises for ``model_group`` after ``original``
    failed (the Python retry loop raises it in the attempt's place, chained to it); a lookup that
    unexpectedly reads is reported as such."""
    from litellm.litellm_core_utils.get_retry_from_policy import get_num_retries_from_retry_policy

    traceback: Final = original.__traceback__
    try:
        # Looked up while `original` is being handled, as in the retry loop's `except` block, so
        # the error chains to it the way Python chains it.
        try:
            raise original
        except BaseException:  # noqa: BLE001  # re-raised only to be handled here
            _ = get_num_retries_from_retry_policy(
                exception=original,  # pyright: ignore[reportArgumentType]  # any exception the hop raised
                retry_policy=retry_policy,  # pyright: ignore[reportArgumentType]  # the router's resolved policy
                model_group=model_group,
                model_group_retry_policy=policies,  # pyright: ignore[reportArgumentType]  # read as the Python router reads it
            )
    except Exception as raised:  # noqa: BLE001  # the error the hop ends with
        return raised
    finally:
        original.__traceback__ = traceback  # rebind-ok: put back as the hop left it
    return RuntimeError(f"the model_group_retry_policy lookup for {model_group!r} did not raise")


def router_policy_retries(
    exception: Exception, retry_policy: object, policies: object, model_group: str | None
) -> int | None:
    """``get_num_retries_from_retry_policy`` for a router ``model_group_retry_policy`` that is not
    a mapping of policies, read as the Python router reads it (raising as it raises)."""
    from litellm.litellm_core_utils.get_retry_from_policy import get_num_retries_from_retry_policy

    return get_num_retries_from_retry_policy(
        exception=exception,
        retry_policy=retry_policy,  # pyright: ignore[reportArgumentType]  # the router's resolved policy
        model_group=model_group,
        model_group_retry_policy=policies,  # pyright: ignore[reportArgumentType]  # read as the Python router reads it
    )


def _count(deployments: object) -> int:
    return len(deployments) if isinstance(deployments, list) else 0  # pyright: ignore[reportUnknownArgumentType]  # only the length is read


def _filled(fallbacks: object) -> bool:
    return isinstance(fallbacks, Sequence) and len(fallbacks) > 0  # pyright: ignore[reportUnknownArgumentType]  # only the length is read


class RustRetries:
    """Retry members read from the Rust core's handle."""

    _handle: "RouterCoreHandle"
    _resolved: "Resolved"

    def should_retry_this_error(
        self,
        error: Exception,
        healthy_deployments: list[object] | None = None,  # mutable-ok: the Python router's signature
        all_deployments: list[object] | None = None,  # mutable-ok: same
        context_window_fallbacks: list[object] | None = None,  # mutable-ok: same
        content_policy_fallbacks: list[object] | None = None,  # mutable-ok: same
        regular_fallbacks: list[object] | None = None,  # mutable-ok: same
    ) -> bool:
        """``True`` when the error is retried in the same group; otherwise the error is raised."""
        from litellm.router_rust_core import attempt_failure

        retried: Final = self._handle.should_retry(
            dict(attempt_failure(error)),
            (_count(healthy_deployments), _count(all_deployments)),
            (context_window_fallbacks is not None, content_policy_fallbacks is not None, _filled(regular_fallbacks)),
        )
        if not retried:
            raise error
        return True

    def get_num_retries_from_retry_policy(self, exception: Exception, model_group: str | None = None) -> int | None:
        from litellm.router_rust_core import attempt_failure

        resolved: Final = self._resolved
        if not is_policy_per_group(resolved.model_group_retry_policy):
            return router_policy_retries(
                exception, resolved.retry_policy, resolved.model_group_retry_policy, model_group
            )
        return self._handle.retry_policy_retries(dict(attempt_failure(exception)), model_group)


__all__ = [
    "GroupPolicies",
    "RustRetries",
    "group_policies",
    "is_policy_per_group",
    "policy_lookup_error",
    "raising_groups",
    "resolved_retry_policy",
    "router_policy_retries",
    "static_group_policies",
]

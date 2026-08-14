"""Applicability filters for the Generic Guardrail API.

Two independent ways to keep a request away from the guardrail endpoint
entirely:

- by call type (``run_only_on_call_types`` / ``skip_call_types``): every hook
  resolves its own call type, so request and response skip symmetrically with no
  correlation needed.
- by request content (``skip_if_system_prompt_matches`` /
  ``skip_if_first_role_in``): only the request side can decide, so the decision
  is recorded and replayed on the paired response via ``SkipDecisionStore``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.guardrails._content_utils import iter_role_text
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

# Roles carrying the operator's instructions. Matching these instead of the whole
# transcript avoids skipping a request because a user pasted a matching string.
INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})

# Fallback correlation cache, used only when the request and response hooks do
# not share a logging object. Bounded and short-lived: a response that never
# fires (stream abort, upstream error) must not leak an entry.
_SKIP_CACHE_MAX_ENTRIES: Final = 1000
_SKIP_CACHE_TTL_SECONDS: Final = 300


@dataclass(frozen=True, slots=True)
class SkipPolicy:
    """Resolved applicability configuration."""

    system_prompt_patterns: tuple[re.Pattern[str], ...] = ()
    first_role_in: frozenset[str] = frozenset()
    key_aliases: frozenset[str] = frozenset()
    team_ids: frozenset[str] = frozenset()
    run_only_on_call_types: frozenset[str] | None = None
    skip_call_types: frozenset[str] = frozenset()

    @property
    def filters_requests(self) -> bool:
        """Whether any message-based filter is configured.

        These read the request body, so only the request hook can decide and the
        decision has to be replayed on the response.
        """
        return bool(self.system_prompt_patterns) or bool(self.first_role_in)

    @property
    def filters_identities(self) -> bool:
        """Whether any caller-identity filter is configured.

        These read what authentication established, which both hooks see, so each
        side decides for itself and no correlation is needed.
        """
        return bool(self.key_aliases) or bool(self.team_ids)


def validate_call_types(
    raw: Sequence[str] | None,
    *,
    option_name: str,
    guardrail_name: str | None,
) -> frozenset[str]:
    """Validate configured call types against ``CallTypes``, warning on unknowns."""
    if not raw:
        return frozenset()
    known: Final = frozenset(call_type.value for call_type in CallTypes)
    unknown: Final = tuple(value for value in raw if value not in known)
    if unknown:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): %s contains unrecognized call type(s) %s. "
            "Values must be CallTypes names (e.g. acompletion, aembedding, anthropic_messages).",
            guardrail_name,
            option_name,
            unknown,
        )
    return frozenset(raw)


def call_type_allowed(policy: SkipPolicy, call_type: str | None) -> bool:
    """Whether the guardrail runs for this call type.

    An unresolved call type runs, so a filter misconfiguration can never
    silently blind the guardrail. The allowlist takes precedence over the
    denylist.
    """
    if call_type is None:
        return True
    if policy.run_only_on_call_types is not None:
        return call_type in policy.run_only_on_call_types
    return call_type not in policy.skip_call_types


def identity_matches_skip(policy: SkipPolicy, metadata: Mapping[str, object]) -> bool:
    """Whether the calling key or team is out of scope for the guardrail.

    Matched against the metadata the auth layer produced, not the request body,
    so a caller cannot exempt itself by choosing what it sends.
    """
    if not policy.filters_identities:
        return False
    key_alias: Final = metadata.get("user_api_key_alias")
    if policy.key_aliases and isinstance(key_alias, str) and key_alias in policy.key_aliases:
        return True
    team_id: Final = metadata.get("user_api_key_team_id")
    return bool(policy.team_ids and isinstance(team_id, str) and team_id in policy.team_ids)


def request_matches_skip(
    policy: SkipPolicy,
    structured_messages: Sequence[Mapping[str, object]] | None,
) -> bool:
    """Whether this request is out of scope for the guardrail."""
    if not policy.filters_requests or not structured_messages:
        return False

    first_role: Final = structured_messages[0].get("role")
    if policy.first_role_in and isinstance(first_role, str) and first_role in policy.first_role_in:
        return True

    if not policy.system_prompt_patterns:
        return False

    return any(
        pattern.search(text)
        for text in iter_role_text(structured_messages, INSTRUCTION_ROLES)
        for pattern in policy.system_prompt_patterns
    )


class SkipDecisionStore:
    """Carries a request-side skip decision to the paired response hook.

    Primary channel is the shared logging object, which is exact and dies with
    the call. The bounded cache keyed by ``litellm_call_id`` covers the paths
    where no logging object reaches the hooks; entries are evicted on read.
    """

    def __init__(self, *, guardrail_name: str | None, cache: InMemoryCache | None = None) -> None:
        self._detail_key: Final = f"_generic_guardrail_skip::{guardrail_name}"
        self._cache: Final = cache or InMemoryCache(
            max_size_in_memory=_SKIP_CACHE_MAX_ENTRIES,
            default_ttl=_SKIP_CACHE_TTL_SECONDS,
        )

    def _cache_key(self, call_id: str) -> str:
        return f"{self._detail_key}::{call_id}"

    @staticmethod
    def _call_details(
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> dict[str, bool] | None:  # mutable-ok: this is the logging object's live dict, written through on purpose
        """The shared write-through channel, or None when this call has no logging object."""
        details: Final[object] = getattr(logging_obj, "model_call_details", None) if logging_obj else None
        # model_call_details is a free-form dict; only our own bool marker is read
        # or written here, so the narrower value type is accurate for this use.
        return details if isinstance(details, dict) else None

    def record(self, *, logging_obj: Optional["LiteLLMLoggingObj"], call_id: str | None) -> None:
        details: Final = self._call_details(logging_obj)
        if details is not None:
            details[self._detail_key] = True
            return
        if call_id:
            self._cache.set_cache(self._cache_key(call_id), True, ttl=_SKIP_CACHE_TTL_SECONDS)

    def consume(self, *, logging_obj: Optional["LiteLLMLoggingObj"], call_id: str | None) -> bool:
        details: Final = self._call_details(logging_obj)
        if details is not None and details.get(self._detail_key) is True:
            return True
        if not call_id:
            return False
        key: Final = self._cache_key(call_id)
        if self._cache.get_cache(key) is not True:
            return False
        self._cache.delete_cache(key)
        return True

"""How often this guardrail records a spend-log / OTEL entry.

``@log_guardrail_information`` appends a ``StandardLoggingGuardrailInformation``
entry on every invocation, with no cap and no dedup, so a long-lived session
accumulates one entry per guardrail call in the same request metadata. This
module decides, per call, whether that entry should be recorded at all.

Suppression rides the decorator's existing contract: it clears
``_guardrail_self_recorded`` before calling the guardrail and skips its own
append if the flag is set afterwards. Setting the flag only on a normal return
(never before a raise) keeps blocks and guardrail failures recorded under every
scope, since the decorator's exception branch honors the same flag.
"""

from typing import Final, Literal

from litellm.caching.in_memory_cache import InMemoryCache

GuardrailInformationScope = Literal["per_call", "per_session", "off"]

DEFAULT_GUARDRAIL_INFORMATION_SCOPE: Final[GuardrailInformationScope] = "per_call"

# Bounded so a long-running proxy cannot accumulate session keys forever.
_SESSION_CACHE_MAX_ENTRIES: Final = 100_000
_SESSION_CACHE_TTL_SECONDS: Final = 3600


class RecordScope:
    """Tracks which sessions have already recorded an entry for one guardrail."""

    def __init__(
        self,
        scope: GuardrailInformationScope,
        *,
        cache: InMemoryCache | None = None,
    ) -> None:
        self._scope: Final = scope
        self._recorded_sessions: Final = cache or InMemoryCache(
            max_size_in_memory=_SESSION_CACHE_MAX_ENTRIES,
            default_ttl=_SESSION_CACHE_TTL_SECONDS,
        )

    @property
    def scope(self) -> GuardrailInformationScope:
        return self._scope

    def should_suppress(self, session_id: str | None, *, tenant: str | None = None) -> bool:
        """Whether this successful call should skip its logging entry.

        ``per_session`` claims the session on its first call, so the calls that
        follow it suppress. A session id is required: without one there is
        nothing to dedup against, so the call records as it would under
        ``per_call`` rather than silently dropping every entry.

        The session id is caller-supplied, so the key is namespaced by the
        authenticated caller: otherwise whoever claims an id first would
        suppress another tenant's telemetry for the same id.
        """
        if self._scope == "per_call":
            return False
        if self._scope == "off":
            return True
        if session_id is None:
            return False
        key: Final = f"{tenant or ''}::{session_id}"
        if self._recorded_sessions.get_cache(key) is True:
            return True
        self._recorded_sessions.set_cache(key, True, ttl=_SESSION_CACHE_TTL_SECONDS)
        return False

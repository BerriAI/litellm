import hashlib
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol
from uuid import uuid4

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from typing_extensions import assert_never

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import log_redis_failure
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GuardrailInformationScope
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging

DEFAULT_GUARDRAIL_INFORMATION_SCOPE: Final[GuardrailInformationScope] = "per_call"

_SESSION_CACHE_MAX_ENTRIES: Final = 100_000
_SESSION_CACHE_TTL_SECONDS: Final = 3600
_REWRITABLE_KEYS: Final = ("texts", "images", "tools", "structured_messages")
_SCOPE_ADAPTER: Final[TypeAdapter[GuardrailInformationScope]] = TypeAdapter(
    GuardrailInformationScope, config=ConfigDict(title="guardrail_information_scope")
)


class RecordedSessions(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object) -> None: ...


class SessionClaimStore(Protocol):
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object: ...

    async def async_get_cache(self, key: str) -> object: ...


def session_claim_store_of(proxy_logging: "ProxyLogging") -> SessionClaimStore | None:
    return proxy_logging.internal_usage_cache.dual_cache.redis_cache


def proxy_session_claim_store() -> SessionClaimStore | None:
    try:
        from litellm.proxy.proxy_server import (  # noqa: PLC0415  # proxy_server imports every guardrail hook at startup
            proxy_logging_obj,
        )
    except ImportError:
        return None
    return session_claim_store_of(proxy_logging_obj)


class Caller(NamedTuple):
    key_hash: str | None
    team_id: str | None
    user_id: str | None


def guardrail_information_scope_from_config(value: object) -> GuardrailInformationScope:
    if value is None:
        return DEFAULT_GUARDRAIL_INFORMATION_SCOPE
    try:
        return _SCOPE_ADAPTER.validate_python(value)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring guardrail_information_scope=%r, expected per_call, per_session or off. Recording every call",
            value,
        )
        return DEFAULT_GUARDRAIL_INFORMATION_SCOPE


def _jsonable(value: object) -> object:
    return to_jsonable_python(value, fallback=repr, bytes_mode="base64")


def _session_key(
    guardrail_name: str | None, caller: Caller, session_id: str, input_type: Literal["request", "response"]
) -> str:
    digest: Final = hashlib.sha256(json.dumps((guardrail_name, *caller, session_id, input_type)).encode()).hexdigest()
    return f"guardrail_information_scope:{digest}"


def _sent_value(sent: GenericGuardrailAPIInputs, key: str) -> object:
    return sent.get(key) if key in sent else ()


def returned_unchanged(sent: GenericGuardrailAPIInputs, returned: GenericGuardrailAPIInputs) -> bool:
    """The return builder always sets texts and sets other rewritable keys only when passing them through or
    rewriting them, so a key missing from ``returned`` is unchanged and missing texts were sent as ``[]``."""
    return all(
        key not in returned or _jsonable(returned.get(key)) == _jsonable(_sent_value(sent, key))
        for key in _REWRITABLE_KEYS
    )


class RecordScope:
    def __init__(
        self,
        scope: GuardrailInformationScope,
        *,
        guardrail_name: str | None,
        shared_store: Callable[[], SessionClaimStore | None] = proxy_session_claim_store,
        recorded_sessions: RecordedSessions | None = None,
    ) -> None:
        self._scope: Final = scope
        self._guardrail_name: Final = guardrail_name
        self._shared_store: Final = shared_store
        self._recorded_sessions: Final[RecordedSessions] = recorded_sessions or InMemoryCache(
            max_size_in_memory=_SESSION_CACHE_MAX_ENTRIES,
            default_ttl=_SESSION_CACHE_TTL_SECONDS,
        )

    @property
    def records_every_allow(self) -> bool:
        return self._scope == "per_call"

    async def should_record_allow(
        self,
        *,
        session_id: str | None,
        caller: Caller,
        input_type: Literal["request", "response"],
    ) -> bool:
        match self._scope:
            case "per_call":
                return True
            case "off":
                return False
            case "per_session":
                return session_id is None or await self._claim_session(
                    _session_key(self._guardrail_name, caller, session_id, input_type)
                )
        return assert_never(self._scope)

    async def _claim_session(self, key: str) -> bool:
        if self._recorded_sessions.get_cache(key) is True:
            return False
        claimed: Final = await self._claim_shared_session(key)
        self._recorded_sessions.set_cache(key, True)
        return claimed

    @with_service_target("guardrail_information_scope")
    async def _claim_shared_session(self, key: str) -> bool:
        claim: Final = uuid4().hex
        try:
            store: Final = self._shared_store()
            if store is None:
                return True
            if await store.async_set_cache(key, claim, nx=True, ttl=_SESSION_CACHE_TTL_SECONDS) is True:
                return True
            return await store.async_get_cache(key) in (None, claim)
        except Exception as claim_error:  # noqa: BLE001  # an unreachable Redis records the allow rather than losing it
            log_redis_failure(
                verbose_proxy_logger, logging.WARNING, "per_session claim failed, recording this allow", claim_error
            )
            return True

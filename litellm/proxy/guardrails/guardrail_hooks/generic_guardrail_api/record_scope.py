import json
from typing import Final, Literal

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GuardrailInformationScope
from litellm.types.utils import GenericGuardrailAPIInputs

DEFAULT_GUARDRAIL_INFORMATION_SCOPE: Final[GuardrailInformationScope] = "per_call"

_SESSION_CACHE_MAX_ENTRIES: Final = 100_000
_SESSION_CACHE_TTL_SECONDS: Final = 3600
_REWRITABLE_KEYS: Final = ("texts", "images", "tools", "structured_messages")
_SCOPE_ADAPTER: Final[TypeAdapter[GuardrailInformationScope]] = TypeAdapter(
    GuardrailInformationScope, config=ConfigDict(title="guardrail_information_scope")
)


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


def returned_unchanged(sent: GenericGuardrailAPIInputs, returned: GenericGuardrailAPIInputs) -> bool:
    """The return builder always sets texts and sets other rewritable keys only when passing them through or
    rewriting them, so a key missing from ``returned`` is unchanged and missing texts were sent as ``[]``."""
    return all(
        key not in returned or _jsonable(returned.get(key)) == _jsonable(sent.get(key, [])) for key in _REWRITABLE_KEYS
    )


class RecordScope:
    def __init__(self, scope: GuardrailInformationScope, *, recorded_sessions: InMemoryCache | None = None) -> None:
        self._scope: Final = scope
        self._recorded_sessions: Final = recorded_sessions or InMemoryCache(
            max_size_in_memory=_SESSION_CACHE_MAX_ENTRIES,
            default_ttl=_SESSION_CACHE_TTL_SECONDS,
        )

    def should_record_allow(
        self,
        *,
        session_id: str | None,
        tenant: str | None,
        input_type: Literal["request", "response"],
    ) -> bool:
        match self._scope:
            case "per_call":
                return True
            case "off":
                return False
            case "per_session":
                return session_id is None or self._claim_session(json.dumps([tenant, session_id, input_type]))
            case _:
                assert_never(self._scope)

    def _claim_session(self, key: str) -> bool:
        if self._recorded_sessions.get_cache(key) is True:
            return False
        self._recorded_sessions.set_cache(key, True)
        return True

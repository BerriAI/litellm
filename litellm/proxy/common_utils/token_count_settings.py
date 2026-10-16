from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import ProxyException
from litellm.types.token_count import TokenCountSourceFields
from litellm.types.utils import TokenCountResponse

CANNOT_COUNT_MODEL_STATUSES: Final = frozenset({400, 404})
MODEL_UNAVAILABLE_STATUS: Final = 404
UNAVAILABLE_FALLBACK_TTL_SECONDS: Final = 3600


class UnavailableFallbacks(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: bool) -> None: ...


UNAVAILABLE_TOKEN_COUNT_FALLBACKS: Final[UnavailableFallbacks] = InMemoryCache(
    default_ttl=UNAVAILABLE_FALLBACK_TTL_SECONDS
)


@dataclass(frozen=True, slots=True)
class TokenCountSettings:
    framing: int | None
    fallback_model: str | None


@dataclass(frozen=True, slots=True)
class ProviderCountFailure:
    status_code: int
    message: str

    @property
    def cannot_count_model(self) -> bool:
        return self.status_code in CANNOT_COUNT_MODEL_STATUSES

    @property
    def model_unavailable(self) -> bool:
        return self.status_code == MODEL_UNAVAILABLE_STATUS

    def to_proxy_exception(self) -> ProxyException:
        return ProxyException(
            message=self.message,
            type="token_counting_error",
            param="model",
            code=self.status_code,
        )


def cost_map_entry(model: str) -> Mapping[str, object]:
    try:
        return litellm.get_model_info(model=model)
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return MappingProxyType({})


def _framing(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int() if value >= 0:
            return value
        case float() if value.is_integer() and value >= 0:
            return int(value)
        case _:
            return None


def _fallback_model(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _resolved_fallback_model(configured: object, default: object) -> str | None:
    match configured:
        case str():
            return _fallback_model(configured)
        case _:
            return _fallback_model(default)


def resolve_token_count_settings(configured: Mapping[str, object], default: Mapping[str, object]) -> TokenCountSettings:
    framing: Final = _framing(configured.get("token_count_framing"))
    return TokenCountSettings(
        framing=framing if framing is not None else _framing(default.get("token_count_framing")),
        fallback_model=_resolved_fallback_model(
            configured.get("token_count_fallback_model"), default.get("token_count_fallback_model")
        ),
    )


def count_source_fields(response: TokenCountResponse) -> TokenCountSourceFields:
    match response.count_source:
        case None:
            return {}
        case "fallback_model":
            return {"litellm_count_source": "fallback_model", "litellm_count_fallback_model": response.model_used}
        case source:
            return {"litellm_count_source": source}

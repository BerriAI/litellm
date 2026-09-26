from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.types.utils import LlmProviders

SAIL: Final = LlmProviders.SAIL.value

CompletionWindow: TypeAlias = Literal["asap", "balanced", "flex"]

_WINDOW_FOR_SERVICE_TIER: Final[Mapping[str, CompletionWindow | None]] = MappingProxyType(
    {"auto": None, "default": "asap", "priority": "asap", "flex": "flex", "balanced": "balanced"}
)
_BILLED_TIER_FOR_WINDOW: Final[Mapping[str, str | None]] = MappingProxyType(
    {"asap": None, "balanced": "balanced", "standard": "balanced", "flex": "flex"}
)
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_DROP_PARAMS_HINT: Final = (
    "To drop it, set `litellm.drop_params=True` or for proxy: `litellm_settings: drop_params: true`"
)


def sail_provider_config() -> SimpleProviderConfig:
    provider: Final = JSONProviderRegistry.get(SAIL)
    assert provider is not None, "litellm/llms/openai_like/providers.json ships a 'sail' entry"
    return provider


def _unsupported(message: str, model: str) -> litellm.UnsupportedParamsError:
    return litellm.UnsupportedParamsError(message=f"{message} {_DROP_PARAMS_HINT}", llm_provider=SAIL, model=model)


def _dropping(drop_params: bool) -> bool:
    return drop_params or bool(litellm.drop_params)


def without_keys(mapping: Mapping[str, object], keys: frozenset[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in mapping.items() if key not in keys})


def _entry(key: str, value: object) -> Mapping[str, object]:
    return MappingProxyType({key: value})


def json_body(mapping: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: HTTP bodies are plain dicts
    return {key: _json_value(value) for key, value in mapping.items()}  # mutable-ok: HTTP bodies are plain dicts


def _json_value(value: object) -> object:
    return json_body(value) if isinstance(value, MappingProxyType) else value


def completion_window_for_service_tier(
    service_tier: object, *, model: str, drop_params: bool
) -> CompletionWindow | None:
    """Sail picks speed and price by ``metadata.completion_window`` and rejects
    ``service_tier``, so the tier is translated."""
    if service_tier is None:
        return None
    tier: Final = service_tier.lower() if isinstance(service_tier, str) else None
    if tier in _WINDOW_FOR_SERVICE_TIER:
        return _WINDOW_FOR_SERVICE_TIER[tier]
    if _dropping(drop_params):
        return None
    raise _unsupported(
        f"sail does not support service_tier={service_tier!r}. Supported values: {', '.join(_WINDOW_FOR_SERVICE_TIER)}.",
        model,
    )


def _metadata_without_caller_window(
    metadata: object, *, field: str, model: str, drop_params: bool
) -> Mapping[str, object]:
    """Chat bills from ``service_tier``, so a window written into metadata would
    run on Sail at a price LiteLLM never charges."""
    if not isinstance(metadata, Mapping):
        return _EMPTY
    if "completion_window" in metadata and not _dropping(drop_params):
        raise _unsupported(f"sail does not accept {field}.completion_window. Send service_tier instead.", model)
    return without_keys(metadata, frozenset({"completion_window"}))


def extra_body_for_sail(
    extra_body: Mapping[str, object], request_metadata: object, *, model: str, drop_params: bool
) -> Mapping[str, object]:
    """``extra_body`` keys are sent over the request body, so its ``metadata``
    would replace the metadata carrying the window. The two are merged, and a
    tier or window set in ``extra_body`` is rejected because billing cannot see it."""
    if "service_tier" in extra_body and not _dropping(drop_params):
        raise _unsupported("sail does not accept service_tier inside extra_body. Send service_tier instead.", model)
    caller_metadata: Final = _metadata_without_caller_window(
        extra_body.get("metadata"), field="extra_body.metadata", model=model, drop_params=drop_params
    )
    merged_metadata: Final = MappingProxyType(
        {**caller_metadata, **(request_metadata if isinstance(request_metadata, Mapping) else _EMPTY)}
    )
    rest: Final = without_keys(extra_body, frozenset({"service_tier", "metadata"}))
    raw_metadata: Final = extra_body.get("metadata")
    if merged_metadata:
        return json_body(MappingProxyType({**rest, "metadata": merged_metadata}))
    if isinstance(raw_metadata, Mapping) or "metadata" not in extra_body:
        return json_body(rest)
    return json_body(MappingProxyType({**rest, "metadata": raw_metadata}))


def chat_request_for_sail(request: Mapping[str, object], *, model: str, drop_params: bool) -> Mapping[str, object]:
    raw_tier: Final = request.get("service_tier")
    window: Final = completion_window_for_service_tier(raw_tier, model=model, drop_params=drop_params)
    caller_metadata: Final = _metadata_without_caller_window(
        request.get("metadata"), field="metadata", model=model, drop_params=drop_params
    )
    metadata: Final = MappingProxyType({**caller_metadata, "completion_window": window}) if window else caller_metadata
    extra_body: Final = request.get("extra_body")
    return MappingProxyType(
        {
            **without_keys(request, frozenset({"service_tier", "metadata", "extra_body"})),
            **(_entry("metadata", metadata) if metadata else _EMPTY),
            **(
                _entry("extra_body", extra_body_for_sail(extra_body, metadata, model=model, drop_params=drop_params))
                if isinstance(extra_body, Mapping)
                else _EMPTY
            ),
        }
    )


def _caller_completion_window(window: object, *, model: str, drop_params: bool) -> str | None:
    if isinstance(window, str) and window.lower() in _BILLED_TIER_FOR_WINDOW:
        return window.lower()
    if _dropping(drop_params):
        return None
    raise _unsupported(
        f"sail does not support metadata.completion_window={window!r}. Supported values: "
        f"{', '.join(_BILLED_TIER_FOR_WINDOW)}.",
        model,
    )


def responses_params_with_completion_window(
    params: Mapping[str, object], *, model: str, drop_params: bool
) -> Mapping[str, object]:
    """Responses billing reads these mapped params, so ``service_tier`` is kept
    as the tier whose price columns match the window and stripped from the body later."""
    raw_tier: Final = params.get("service_tier")
    raw_metadata: Final = params.get("metadata")
    metadata: Final[Mapping[str, object]] = raw_metadata if isinstance(raw_metadata, Mapping) else _EMPTY
    tier_window: Final = completion_window_for_service_tier(raw_tier, model=model, drop_params=drop_params)
    caller_window: Final = (
        _caller_completion_window(metadata["completion_window"], model=model, drop_params=drop_params)
        if "completion_window" in metadata
        else None
    )
    if (
        caller_window is not None
        and tier_window is not None
        and _BILLED_TIER_FOR_WINDOW[caller_window] != _BILLED_TIER_FOR_WINDOW[tier_window]
    ):
        raise _unsupported(
            f"sail got service_tier={raw_tier!r} and metadata.completion_window={caller_window!r}, which "
            "select different completion windows. Send one of them.",
            model,
        )
    window: Final = caller_window or tier_window
    other_metadata: Final = without_keys(metadata, frozenset({"completion_window"}))
    wire_metadata: Final = (
        MappingProxyType({**other_metadata, "completion_window": window}) if window else other_metadata
    )
    billed_tier: Final = _BILLED_TIER_FOR_WINDOW[window] if window else None
    return MappingProxyType(
        {
            **without_keys(params, frozenset({"service_tier", "metadata"})),
            **(_entry("metadata", wire_metadata) if wire_metadata or raw_metadata is not None else _EMPTY),
            **(_entry("service_tier", billed_tier) if billed_tier else _EMPTY),
        }
    )

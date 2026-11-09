from collections.abc import Mapping, Sequence
from typing import Final

from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry


def _entry(key: str, catalog: Mapping[str, object]) -> AliasedCostEntry | None:
    value: Final = catalog.get(key)
    return value if isinstance(value, AliasedCostEntry) and value.names[:1] not in ((), (key,)) else None


def _provider(key: str, value: AliasedCostEntry, providers: Sequence[str]) -> str | None:
    prefix, separator, _ = key.partition("/")
    if separator and prefix in providers:
        return prefix
    provider: Final = value.get("litellm_provider")
    return provider if isinstance(provider, str) else None


def _target(key: str, provider: str, catalog: Mapping[str, object], providers: Sequence[str]) -> str | None:
    value: Final = _entry(key, catalog)
    if value is None or _provider(key, value, providers) != provider:
        return None
    return value.names[0].removeprefix(f"{provider}/")


def expand(model: str, catalog: Mapping[str, object], providers: Sequence[str]) -> tuple[str, ...]:
    """Alias expansion frozen from main 2f430873aead937b9112e683703956eb1ab179af."""
    provider, separator, name = model.partition("/")
    if not separator:
        value: Final = _entry(model, catalog)
        return (model,) if value is None else (model, value.names[0])
    target: Final = (
        _target(f"{provider}/{name}", provider, catalog, providers)
        or _target(name, provider, catalog, providers)
        or name
    )
    return (model, f"{provider}/{target}") if target != name else (model,)

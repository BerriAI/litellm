from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal, ParamSpec, TypeVar, overload

from pydantic import TypeAdapter, ValidationError

from litellm._catalog import _install
from litellm.rust_bridge import _native

if TYPE_CHECKING:
    from litellm.litellm_core_utils.get_model_cost_map import ProviderModelSetOverride

_ROW: Final = TypeAdapter(dict[str, object])
QUIET: Final = ContextVar("catalog_metadata_read", default=False)
_P = ParamSpec("_P")
_T = TypeVar("_T")


def quiet(function: Callable[_P, _T]) -> Callable[_P, _T]:
    @wraps(function)
    def read(*args: _P.args, **kwargs: _P.kwargs) -> _T:  # kwargs-ok: ParamSpec preserves decorated signatures
        token: Final = QUIET.set(True)
        try:
            refresh_provider_sets()
            return function(*args, **kwargs)
        finally:
            QUIET.reset(token)

    return read


@dataclass(slots=True)
class _ProviderIndex:
    rows: dict[str, object] | None = None
    length: int = -1


_PROVIDERS: Final = _ProviderIndex()


def registration_provider_overrides(key: str, current: object, include: bool) -> tuple[ProviderModelSetOverride, ...]:
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import ProviderModelSetOverride

    previous: Final = provider_overrides(current)
    retained: Final = tuple(override for override in previous if override.key != key)
    if include:
        return retained
    memberships: Final = litellm.provider_model_set_memberships(key)
    return (*retained, ProviderModelSetOverride(key, memberships))


def provider_overrides(entry: object) -> tuple[ProviderModelSetOverride, ...]:
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry

    return entry.provider_model_set_overrides if isinstance(entry, AliasedCostEntry) else ()


def _provider_overrides_for_key(key: str) -> tuple[ProviderModelSetOverride, ...]:
    entry: Final = _install._live_rows().get(key)  # pyright: ignore[reportPrivateUsage]  # keyed host metadata, never row values
    return tuple(override for override in provider_overrides(entry) if override.key == key)


def provider_overrides_for_keys(keys: tuple[str, ...]) -> tuple[ProviderModelSetOverride, ...]:
    return tuple(chain.from_iterable(_provider_overrides_for_key(key) for key in keys))


def refresh_provider_sets(*, force: bool = False) -> None:
    import litellm

    rows: Final = _install._live_rows()  # pyright: ignore[reportPrivateUsage]  # read the public dict without copying
    if not force and _PROVIDERS.rows is rows and _PROVIDERS.length == len(rows):
        return
    if not hasattr(litellm, "models_by_provider"):
        return
    with _install.LOCK:
        length: Final = len(rows)
        litellm._rebuild_provider_model_sets()  # pyright: ignore[reportPrivateUsage]  # catalog index publication
        _PROVIDERS.rows = rows
        _PROVIDERS.length = length


def invalidate() -> None:
    _native.invalidate_model_cost_index()
    refresh_provider_sets(force=True)


def infer(model: str) -> tuple[str, str | None]:
    import litellm

    try:
        normalized, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:  # noqa: BLE001  # host routing failures mean uninferred catalog intent
        return model, None
    return normalized, provider


@overload
def resolve(
    model: str,
    provider: str | None = None,
    *,
    base_model: str | None = None,
    region: str | None = None,
    inferred: tuple[str, str | None] | None = None,
    info_read: Literal[False] = False,
) -> _native.ResolvedModel | _native.ModelNotFound: ...


@overload
def resolve(
    model: str,
    provider: str | None = None,
    *,
    base_model: str | None = None,
    region: str | None = None,
    inferred: tuple[str, str | None] | None = None,
    info_read: bool,
) -> _native.ResolvedModel | _native.ModelNotFound | _native.InvalidProvider: ...


def resolve(
    model: str,
    provider: str | None = None,
    *,
    base_model: str | None = None,
    region: str | None = None,
    inferred: tuple[str, str | None] | None = None,
    info_read: bool = False,
) -> _native.ResolvedModel | _native.ModelNotFound | _native.InvalidProvider:
    refresh_provider_sets()
    return _native.resolve_model(
        _install._live_rows(),  # pyright: ignore[reportPrivateUsage]  # native reads must retain the public dict
        model,
        provider=provider,
        base_model=base_model,
        region=region,
        inferred=inferred,
        info_read=info_read,
    )


def model_info(key: str) -> dict[str, object]:
    rows: Final = _install._live_rows()  # pyright: ignore[reportPrivateUsage]  # lower the selected live row
    result: Final = _native.model_info(rows, key)
    raw: Final = _ROW.validate_python(dict[str, object].get(rows, key))
    if isinstance(result, _native.RowErrors):
        return raw
    return {name: raw[name] for name in result.fields}


def is_priced(model: str, provider: str | None = None) -> bool:
    resolved: Final = resolve(model, provider)
    if not isinstance(resolved, _native.ResolvedModel):
        return False
    try:
        row: Final = _ROW.validate_python(
            dict[str, object].get(
                _install._live_rows(),  # pyright: ignore[reportPrivateUsage]  # presence uses the selected live row
                resolved.key,
            )
        )
    except ValidationError:
        return False
    contextual_row: Final = row if "litellm_provider" in row else {"litellm_provider": provider or "custom", **row}
    match _native.has_token_prices(contextual_row, policy="sdk_tokens"):
        case bool() as present:
            return present
        case _native.RowErrors():
            return False


def supports(fields: Mapping[str, object], capability: str) -> bool | None:
    value: Final = fields.get(capability)
    return None if value is None else value is True


def clear_cache() -> None:
    return None

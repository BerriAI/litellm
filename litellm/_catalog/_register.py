from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm._catalog import _install
from litellm._logging import verbose_logger
from litellm.rust_bridge._native import RegistrationTarget, registration_target

_ROW: Final = TypeAdapter(Mapping[str, object])


def _base_fields(
    target: RegistrationTarget,
    inherited: Mapping[str, object],
    value: Mapping[str, object],
    custom_llm_provider: str | None,
) -> Mapping[str, object]:
    from litellm import utils

    lookups: Final = (utils.lookup_model_info_for_registration(target.write, custom_llm_provider) for _ in range(3))
    read: Final = next(
        (lookup for lookup in lookups if not (isinstance(lookup, utils.ModelInfoLookupMiss) and lookup.tripped)),
        utils.ModelInfoLookupMiss(tripped=True),
    )
    primary: Final = not isinstance(read, utils.ModelInfoLookupMiss) and read.get("key") == target.inherit_from
    if primary:
        return {name: field for name, field in inherited.items() if name != "litellm_provider" or field is not None}
    rates: Final = (
        inherited
        if target.names_the_row
        else utils._without_overridden_price_variants(  # pyright: ignore[reportPrivateUsage]  # shared rate policy
            inherited, value
        )
    )
    generalized: Final = _ROW.validate_python(
        utils.match_capability_generalizations(  # pyright: ignore[reportUnknownMemberType]  # validate legacy rules
            target.write
        )
        or {}
    )
    own_fields: Final = {
        name: field
        for name, field in rates.items()
        if name != "litellm_provider" or (target.write == target.inherit_from and field is not None)
    }
    return {**generalized, **own_fields}


def registration_row(key: str, provider: str | None) -> Mapping[str, object]:
    with _install.LOCK:
        live: Final = _install._live_rows()  # pyright: ignore[reportPrivateUsage]  # retain public dict identity
        target: Final = registration_target(live, key, provider=provider)
        return _ROW.validate_python(live.get(target.inherit_from, {}) if target.inherit_from is not None else {})


def _base_fields_replaced_by(value: Mapping[str, object]) -> frozenset[str]:
    return (
        frozenset({"explicit_cache_read_input_token_cost"})
        if value.get("cache_read_input_token_cost") is not None
        and value.get("explicit_cache_read_input_token_cost") is None
        else frozenset()
    )


def _register_entry(
    key: str,
    value: Mapping[str, object],
    *,
    persist: bool,
    custom_llm_provider: str | None,
    warning_display_name: str | None,
    add_to_provider_model_sets: bool,
) -> bool:
    from litellm import utils

    generation: Final = _install.STATE.generation
    live: Final = _install._live_rows()  # pyright: ignore[reportPrivateUsage]  # retain public dict identity
    provider: Final = value.get("litellm_provider")
    target: Final = registration_target(
        live, key, provider=custom_llm_provider or (provider if isinstance(provider, str) else None)
    )
    inherited: Final = _ROW.validate_python(
        live.get(target.inherit_from, {}) if target.inherit_from is not None else {}
    )
    base: Final = _base_fields(target, inherited, value, custom_llm_provider)
    merged: Final = utils.update_dictionary(dict(base), value)
    fields: Final = {
        name: field
        for name, field in merged.items()
        if value.get(name) is not None or name not in inherited or field != inherited[name]
    }
    erased: Final = _base_fields_replaced_by(value)
    previous: Final = _ROW.validate_python(live.get(key, {}))
    removed: Final = (frozenset(inherited) | frozenset(previous)) - frozenset(merged)
    overrides: Final = {**fields, **dict.fromkeys(removed | erased)}
    if generation != _install.STATE.generation:
        return False
    if not _install.registration_write(
        target.write,
        base,
        overrides,
        persist=persist,
        inherits_from=target.inherit_from if target.inherit_from != target.write else None,
        generation=generation,
        add_to_provider_model_sets=add_to_provider_model_sets,
    ):
        return False
    if target.inherit_from is None and (
        value.get("cache_creation_input_token_cost") is None
        and value.get("cache_read_input_token_cost") is None
        and value.get("tiered_pricing") is None
        and (value.get("input_cost_per_token") is not None or value.get("output_cost_per_token") is not None)
    ):
        verbose_logger.warning(
            "register_model: model=%s has custom pricing but not in built-in cost map and no prefix/region variant "
            "matched; cache_creation_input_token_cost and cache_read_input_token_cost will default to 0 for this "
            "model (input/output cost tracking is unaffected). To track cache cost, add them to model_info",
            warning_display_name or key,
        )
    return True


def register(
    entries: Mapping[str, Mapping[str, object]],
    *,
    persist: bool,
    custom_llm_provider: str | None,
    warning_display_name: str | None,
    add_to_provider_model_sets: bool = True,
) -> None:
    for key, value in entries.items():
        with _install.LOCK:
            while not _register_entry(
                key,
                value,
                persist=persist,
                custom_llm_provider=custom_llm_provider,
                warning_display_name=warning_display_name,
                add_to_provider_model_sets=add_to_provider_model_sets,
            ):
                pass


class RuntimeRegistrations(Mapping[str, Mapping[str, object]]):
    @staticmethod
    def _snapshot() -> Mapping[str, Mapping[str, object]]:
        with _install.LOCK:
            entries: Final[dict[str, Mapping[str, object]]] = {}  # mutable-ok: fold ordered native sparse records
            for key, _, fields in _install.LAYERS.registrations():
                entries[key] = MappingProxyType(
                    {
                        **entries.get(key, {}),
                        **{
                            name: _install._decoded_value(  # pyright: ignore[reportPrivateUsage]  # opaque metadata
                                value
                            )
                            for name, value in fields.items()
                        },
                    }
                )
            return MappingProxyType(entries)

    def __getitem__(self, key: str) -> Mapping[str, object]:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def items(self) -> ItemsView[str, Mapping[str, object]]:
        return self._snapshot().items()

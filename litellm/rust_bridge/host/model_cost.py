"""The ``litellm.model_cost`` questions Rust asks while registering models and pricing calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils import model_cost_table
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry, Landing


def entry(fields: Mapping[str, object], names: tuple[str, ...]) -> AliasedCostEntry:
    """The row object every name in ``names`` shares, the owner first."""
    return AliasedCostEntry(fields, names)


published: Final = model_cost_table.published
write: Final = model_cost_table.write


def generation() -> tuple[Mapping[str, object], int, int, int, int, int]:
    """The published table with what a pricing snapshot is keyed on: its ``ModelCostVersion`` and the identity of the
    cost discount and margin configs, which are only ever replaced whole."""
    import litellm

    table: Final = model_cost_table.published()
    version: Final = model_cost_table.model_cost_version()
    return (
        table,
        version.table,
        version.size,
        version.writes,
        id(litellm.cost_discount_config),
        id(litellm.cost_margin_config),
    )


def known_names() -> tuple[frozenset[str], frozenset[str]]:
    """Every provider name, and the model names ``get_model_info`` renames before it looks one up."""
    import litellm
    from litellm.types.utils import LlmProvidersSet

    return frozenset(LlmProvidersSet), frozenset({**litellm.azure_llms, **litellm.azure_embedding_models})


def resolve(
    rows: Mapping[str, Mapping[str, object]] | None,
    key: str,
    registration: Mapping[str, object],
    custom_llm_provider: str | None,
) -> Landing | None:
    """Where a ``register_model`` entry lands in ``rows`` and the row it starts from, ``None`` when ``rows`` is no
    longer the published table. ``rows`` of ``None`` is the published table."""
    import litellm
    from litellm.utils import registration_base

    if rows is not None and rows is not litellm.model_cost:
        return None
    return registration_base(key, registration, custom_llm_provider)


def merged(
    current: Mapping[str, object] | None, base: Mapping[str, object], registration: Mapping[str, object]
) -> AliasedCostEntry:
    """The row a registration writes over ``current``, the row now under its landing key."""
    from litellm.utils import model_cost_registration_write

    return model_cost_registration_write(current, base, registration)


def kept(previous: dict[str, object] | None, registration: Mapping[str, object]) -> dict[str, object]:
    """The registration ``register_model(persist_across_reloads=True)`` keeps for a key: ``registration`` merged
    into the one kept before it."""
    from litellm.utils import base_fields_replaced_by, update_dictionary

    replaced: Final = base_fields_replaced_by(registration)
    return {
        name: value
        for name, value in update_dictionary(dict(previous or {}), registration).items()
        if name not in replaced
    }

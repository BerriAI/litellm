"""The ``litellm.model_cost`` questions Rust asks while registering models."""

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
    from litellm.utils import update_dictionary

    return update_dictionary(dict(previous or {}), registration)

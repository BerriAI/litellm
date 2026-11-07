from __future__ import annotations

from collections.abc import Mapping

from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry


def entry(fields: dict[str, object], names: tuple[str, ...], registered_fields: frozenset[str]) -> AliasedCostEntry:
    """The row object every alias name of one model shares in ``litellm.model_cost``."""
    return AliasedCostEntry(fields, names, registered_fields)


def write(
    rows: dict[str, Mapping[str, object]],
    key: str,
    registration: Mapping[str, object],
    custom_llm_provider: str | None,
    warning_display_name: str | None,
) -> str:
    """Writes one ``register_model`` entry into ``rows`` and returns the row key it landed on."""
    from litellm.utils import write_model_cost_registration

    return write_model_cost_registration(rows, key, registration, custom_llm_provider, warning_display_name)

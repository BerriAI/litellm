from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger
from litellm.rust_bridge._native import ModelRegistry

if TYPE_CHECKING:
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry


@dataclass(frozen=True, slots=True)
class LoadedCatalog:
    registry: ModelRegistry


@dataclass(frozen=True, slots=True)
class CatalogUnusable:
    reason: str


def load(body: bytes | None) -> LoadedCatalog | CatalogUnusable:
    try:
        registry: Final = ModelRegistry.bundled() if body is None else ModelRegistry.parse(body)
    except ValueError as error:
        return CatalogUnusable(str(error))
    problems: Final = tuple(
        f"{error.key}.{error.field}: {error.message}" if error.field else f"{error.key}: {error.message}"
        for error in registry.errors()
    )
    if problems:
        verbose_logger.warning(
            "LiteLLM: %d validation or alias conflict issues in the model cost map, serving every entry as written: %s",
            len(problems),
            "; ".join(problems),
        )
    return LoadedCatalog(registry)


def served_rows(
    catalog: LoadedCatalog,
) -> dict[str, AliasedCostEntry]:  # mutable-ok: model_cost is a public mutable dict
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry

    rows: Final = catalog.registry.served()
    grouped: Final = groupby(sorted(rows, key=lambda key: id(rows[key])), key=lambda key: id(rows[key]))
    names_by_row: Final = {identity: tuple(names) for identity, names in grouped}
    entries: Final = {
        identity: AliasedCostEntry(rows[names[0]], tuple(names)) for identity, names in names_by_row.items()
    }
    return {key: entries[id(row)] for key, row in rows.items()}

import json
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from e2e.coverage_registry.registry import load_registry

MAPPING: Final = TypeAdapter(dict[str, tuple[str, ...]])
OWNED_DIRECTORIES: Final = frozenset(
    {
        "management",
        "authorization",
        "database",
        "pricing",
        "spend",
        "routing",
        "providers",
        "streaming",
        "configuration",
        "mcp",
        "observability",
        "compatibility",
    }
)


def contracts() -> dict[str, tuple[str, ...]]:
    document: Final = json.loads((Path(__file__).resolve().parents[1] / "contracts.json").read_bytes())
    result: Final = MAPPING.validate_python(document["tests"])
    if not result or any(not values for values in result.values()):
        raise ValueError("Integration manifest must contain nodes with contract IDs")
    registered: Final = {cell.id for cell in load_registry()}
    unknown: Final = {identity for identities in result.values() for identity in identities} - registered
    if unknown:
        raise ValueError(f"Unknown coverage-registry contracts: {sorted(unknown)}")
    return result

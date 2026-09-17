import json
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

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
    if not result or any(not values or any(not value.strip() for value in values) for values in result.values()):
        raise ValueError("Integration manifest must contain nodes with contract IDs")
    return result

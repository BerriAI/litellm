from typing import Final

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
        "sdk",
        "cost_calculation",
    }
)

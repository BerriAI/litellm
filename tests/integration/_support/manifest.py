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
        "messages",
        "configuration",
        "mcp",
        "observability",
        "compatibility",
        "sdk",
        "cost_calculation",
    }
)

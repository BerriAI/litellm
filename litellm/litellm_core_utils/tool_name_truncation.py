import hashlib
from typing import Final

OPENAI_MAX_TOOL_NAME_LENGTH: Final = 64
TOOL_NAME_HASH_LENGTH: Final = 8
TOOL_NAME_PREFIX_LENGTH: Final = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1


def _tool_name_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]


def truncate_tool_name(name: str) -> str:
    """Cut a name over 64 characters to a 55-character prefix plus a hash of the full name, else keep it."""
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{_tool_name_hash(name)}"


def namespaced_tool_name(namespace: str, name: str) -> str:
    """Join as ``namespace__name`` within 64 characters, cutting the namespace before the tool name.

    The hash covers the full join, so a pair gets the same name on every turn.
    """
    joined: Final = f"{namespace}__{name}"
    if len(joined) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return joined
    suffix: Final = f"_{_tool_name_hash(joined)}__{name}"
    namespace_budget: Final = OPENAI_MAX_TOOL_NAME_LENGTH - len(suffix)
    return f"{namespace[:namespace_budget]}{suffix}" if namespace_budget > 0 else truncate_tool_name(joined)

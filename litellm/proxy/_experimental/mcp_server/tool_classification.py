"""Classify an MCP tool name (+ optional description) into a coarse operation kind.

The MCP tool-permission convention treats ``delete`` differently from
everything else (create/read/update/unknown all count as non-delete), so the
single exported function is all callers need.
"""

import re
from collections.abc import Iterable
from itertools import chain
from typing import Final, Literal, TypeAlias

ToolOperation: TypeAlias = Literal["read", "create", "update", "delete", "unknown"]

_READ_TOKENS: Final = frozenset(
    {
        "get",
        "read",
        "list",
        "fetch",
        "search",
        "find",
        "query",
        "retrieve",
        "show",
        "view",
        "check",
        "describe",
        "info",
        "lookup",
        "count",
        "export",
        "download",
    }
)
_DELETE_TOKENS: Final = frozenset(
    {
        "delete",
        "remove",
        "destroy",
        "purge",
        "drop",
        "erase",
        "unlink",
        "wipe",
        "clear",
        "revoke",
        "uninstall",
        "trash",
        "truncate",
        "rm",
        "del",
    }
)
_UPDATE_TOKENS: Final = frozenset(
    {
        "update",
        "edit",
        "modify",
        "change",
        "patch",
        "put",
        "set",
        "rename",
        "move",
        "transform",
        "toggle",
        "enable",
        "disable",
        "archive",
        "restore",
    }
)
_CREATE_TOKENS: Final = frozenset(
    {
        "create",
        "add",
        "insert",
        "new",
        "post",
        "submit",
        "register",
        "make",
        "generate",
        "write",
        "upload",
        "send",
        "publish",
    }
)

_SPLIT_RE: Final = re.compile(r"[_\-./\s]+")
_CAMEL_BOUNDARY_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _name_tokens(name: str) -> tuple[str, ...]:
    return tuple(
        token.lower() for token in chain.from_iterable(map(_CAMEL_BOUNDARY_RE.split, _SPLIT_RE.split(name))) if token
    )


def _description_tokens(description: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in re.split(r"[^\w]+", description) if token)


def _token_variants(token: str) -> frozenset[str]:
    variants: Final = frozenset(
        (
            token,
            token.removesuffix("es"),
            token.removesuffix("s"),
            token.removesuffix("ed"),
            token.removesuffix("ing"),
        )
    )
    return frozenset(variant for variant in variants if variant)


_CONJUNCTION_TOKENS: Final = frozenset({"and", "then", "or", "n"})


def _has_bare_delete_verb(tokens: tuple[str, ...]) -> bool:
    return any(
        token in _DELETE_TOKENS and (index == 0 or tokens[index - 1] in _CONJUNCTION_TOKENS)
        for index, token in enumerate(tokens)
    )


def _classify_tokens(tokens: Iterable[str]) -> ToolOperation:
    token_tuple: Final = tuple(tokens)
    if _has_bare_delete_verb(token_tuple):
        return "delete"
    token_set: Final = frozenset(chain.from_iterable(map(_token_variants, token_tuple)))
    if token_set & _READ_TOKENS:
        return "read"
    if token_set & _DELETE_TOKENS:
        return "delete"
    if token_set & _UPDATE_TOKENS:
        return "update"
    if token_set & _CREATE_TOKENS:
        return "create"
    return "unknown"


def classify_tool_op(name: str, description: str | None = None) -> ToolOperation:
    """Classify a tool by exact token match on its name, falling back to the
    description's words only when the name yields no recognized token.

    A bare destructive verb outranks read tokens only when it leads the name
    or follows a conjunction, so ``get_and_delete_item`` is delete while
    ``describe_purge_job`` and ``getDeleteStatus`` are read. Inflected forms
    only match through variants under read > delete > update > create, and a
    misleading description cannot override a recognized name."""
    by_name: Final = _classify_tokens(_name_tokens(name))
    if by_name != "unknown":
        return by_name
    if description:
        return _classify_tokens(_description_tokens(description))
    return "unknown"

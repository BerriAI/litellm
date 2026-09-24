"""Classify an MCP tool into a coarse operation kind from its name and description.

Port of ``ui/litellm-dashboard/src/utils/mcpToolCrudClassification.ts``
``classifyToolOp``; keep the regexes, the read-before-delete ordering, and the
name-then-description fallback in lockstep with it. The shared fixture
``mcpToolCrudClassification.fixture.json`` pins parity on both sides.

The name is checked first, and the description is consulted only when the name
matched nothing, so incidental phrasing in free-form descriptions cannot promote
a safe tool into a high-risk bucket. ``re.ASCII`` keeps ``\\b`` matching the JS
``\\b``: ``_`` is a word character, so ``delete_link`` does not match by name.
"""

import re
from typing import Final, Literal

CrudOp = Literal["read", "create", "update", "delete", "unknown"]

_FLAGS: Final = re.IGNORECASE | re.ASCII

_READ_RE: Final = re.compile(
    r"\b(get|read|list|fetch|search|find|query|retrieve|show|view|check|describe|info)\b", _FLAGS
)
_DELETE_RE: Final = re.compile(r"\b(delete|remove|destroy|purge|drop|erase|unlink)\b", _FLAGS)
_UPDATE_RE: Final = re.compile(r"\b(update|edit|modify|change|patch|put|set|rename|move|transform)\b", _FLAGS)
_CREATE_RE: Final = re.compile(r"\b(create|add|insert|new|post|submit|register|make|generate|write|upload)\b", _FLAGS)

_ORDERED: Final = (
    (_READ_RE, "read"),
    (_DELETE_RE, "delete"),
    (_UPDATE_RE, "update"),
    (_CREATE_RE, "create"),
)


def classify_tool_op(name: str, description: str = "") -> CrudOp:
    for pattern, op in _ORDERED:
        if pattern.search(name):
            return op
    if description:
        for pattern, op in _ORDERED:
            if pattern.search(description):
                return op
    return "unknown"

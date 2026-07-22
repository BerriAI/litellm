"""Shared exception-tree traversal for fault classification.

Failures cross the MCP SDK's anyio task groups wrapped in ``ExceptionGroup``s and chained through
``raise ... from`` causes, so every classifier that needs an exception buried in the tree (an
upstream ``httpx.Response``, a context-window overflow) has to walk the same shapes. One traversal
with one deliberate order keeps blame assignment consistent across classifiers: explicit links are
searched before incidental ones, so an exception raised while handling the real failure can never
shadow the failure itself.
"""

from __future__ import annotations

from collections.abc import Iterator


def iter_exception_tree(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception reachable from it, explicit links first: each node's
    ``raise ... from`` cause subtree, then ``ExceptionGroup`` members in raise order, then the
    incidental ``__context__`` chain last. Cycle-safe via identity tracking, and iterative so a
    deep chain cannot overflow the interpreter stack."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__context__ is not None:
            stack.append(current.__context__)
        exceptions = getattr(current, "exceptions", None)
        if isinstance(exceptions, tuple):
            stack.extend(reversed(exceptions))
        if current.__cause__ is not None:
            stack.append(current.__cause__)

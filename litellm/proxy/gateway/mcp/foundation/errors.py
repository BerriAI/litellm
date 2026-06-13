"""GatewayError — the single typed failure vocabulary for the gateway.

Leaf module: imports only the Expression spine. Must NOT import result.py
(result.py imports this), to keep the dependency acyclic.

Errors here are transport-agnostic; they describe what went wrong, not how a
channel renders it. Mapping an error to a JSON-RPC error code or an HTTP status
is an edge concern that lives with each transport, not in this leaf.
"""

from __future__ import annotations

from typing import Literal, assert_never, cast

from expression import case, tag, tagged_union

_Tag = Literal["db_unavailable", "unauthorized", "invalid_input", "not_implemented"]


@tagged_union(frozen=True)
class GatewayError:
    tag: _Tag = cast(_Tag, tag())
    db_unavailable: str = cast(str, case())
    unauthorized: str = cast(str, case())
    invalid_input: str = cast(str, case())
    not_implemented: str = cast(str, case())


def reason(e: GatewayError) -> str:
    match e.tag:
        case "db_unavailable":
            return e.db_unavailable
        case "unauthorized":
            return e.unauthorized
        case "invalid_input":
            return e.invalid_input
        case "not_implemented":
            return e.not_implemented
        case _:
            assert_never(e.tag)

#!/usr/bin/env python3
"""List the POST operations in the proxy's OpenAPI document that Swagger UI cannot explain.

Every finding is one line, ``<METHOD> <path> <code>``, so scripts/openapi_docs_gate.py can
count them per rule the same delta-vs-base way as the other lint gates. The rules:

OA001  no request body schema (the handler reads ``request.body()`` and never told FastAPI
       what is in it, see ``inline_request_body`` in custom_openapi_spec.py for the fix)
OA002  no description (the handler has no docstring and the route no ``description=``)
OA003  the 200 response has no schema (no ``response_model`` and no ``responses={200: ...}``)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from typing import Final, NamedTuple

GATED_METHODS: Final = frozenset({"post"})


class Violation(NamedTuple):
    method: str
    path: str
    code: str


def _as_mapping(node: object) -> Mapping[str, object]:
    return node if isinstance(node, Mapping) else {}


def _json_schema(container: object) -> object:
    content: Final = _as_mapping(_as_mapping(container).get("content"))
    return _as_mapping(content.get("application/json")).get("schema")


def _operation_codes(operation: Mapping[str, object]) -> Iterator[str]:
    if not _json_schema(operation.get("requestBody")):
        yield "OA001"
    description: Final = operation.get("description")
    if not isinstance(description, str) or not description.strip():
        yield "OA002"
    if not _json_schema(_as_mapping(operation.get("responses")).get("200")):
        yield "OA003"


def violations(spec: Mapping[str, object]) -> tuple[Violation, ...]:
    paths: Final = _as_mapping(spec.get("paths"))
    return tuple(
        Violation(method.upper(), path, code)
        for path, operations in sorted(paths.items())
        for method, operation in sorted(_as_mapping(operations).items())
        if method in GATED_METHODS
        for code in _operation_codes(_as_mapping(operation))
    )  # comprehension-ok: one pass over path, method and code is the whole scan


def proxy_spec() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import app

    return _as_mapping(app.openapi())


def main(argv: list[str] = sys.argv[1:]) -> int:
    spec: Final = json.loads(open(argv[0]).read()) if argv else proxy_spec()
    for violation in violations(spec):
        print(f"{violation.method} {violation.path} {violation.code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

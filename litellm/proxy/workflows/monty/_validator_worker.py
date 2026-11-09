from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Final, NoReturn, TypeAlias, cast  # noqa: TID251  # stdlib decoder produces the recursive JSON union

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import Schema

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
MAX_MESSAGE_BYTES: Final = 64 * 1024 * 1024


def refuse_retrieval(uri: str) -> NoReturn:
    raise NoSuchResource(ref=uri)


def closed_registry() -> Registry[Schema]:
    return Registry(retrieve=refuse_retrieval)


def validate_request(line: bytes) -> Mapping[str, JsonValue]:
    try:
        request: Final = cast(JsonValue, json.loads(line))  # cast-ok: stdlib JSON decoder returns only JSON values
        if not isinstance(request, dict) or not isinstance(schema := request.get("schema"), dict):
            return {"valid": False, "message": "invalid validator request"}
        validator: Final = Draft202012Validator(schema, registry=closed_registry())
        validator.validate(request.get("value"))  # pyright: ignore[reportUnknownMemberType]  # upstream kwargs are untyped
    except (ValidationError, Unresolvable, ValueError, RecursionError) as exc:
        return {"valid": False, "message": str(exc)[:4096]}
    else:
        return {"valid": True, "message": ""}


def main() -> None:
    sys.stdout.buffer.write(b"ready\n")
    sys.stdout.buffer.flush()
    while line := sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1):
        if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
            return
        sys.stdout.buffer.write(json.dumps(validate_request(line), ensure_ascii=True).encode() + b"\n")
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()

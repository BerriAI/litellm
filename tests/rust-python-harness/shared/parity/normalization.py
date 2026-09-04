from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Final, TypeAlias, cast

from pydantic import JsonValue

from .models import Execution

JsonPathPart: TypeAlias = str | int
JsonPath: TypeAlias = tuple[JsonPathPart, ...]
MASKED_VALUE: Final = "<volatile>"


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    request_headers: frozenset[str] = frozenset()
    request_body_paths: tuple[JsonPath, ...] = ()
    report_paths: tuple[JsonPath, ...] = ()


def _mask_path(value: JsonValue, path: JsonPath) -> JsonValue:
    if not path:
        return MASKED_VALUE
    head, *tail = path
    remaining: Final = tuple(tail)
    if isinstance(head, str) and isinstance(value, dict) and head in value:
        return {**value, head: _mask_path(value[head], remaining)}
    if isinstance(head, int) and isinstance(value, list) and 0 <= head < len(value):
        return [_mask_path(item, remaining) if index == head else item for index, item in enumerate(value)]
    return value


def _mask_paths(value: JsonValue, paths: tuple[JsonPath, ...]) -> JsonValue:
    return reduce(_mask_path, paths, value)


def normalize_execution(execution: Execution, spec: NormalizationSpec) -> Execution:
    lowered_headers: Final = frozenset(name.lower() for name in spec.request_headers)
    dumped: Final = cast(dict[str, JsonValue], execution.model_dump(mode="json"))
    requests: Final = cast(list[dict[str, JsonValue]], dumped["requests"])
    normalized_requests: Final = [
        {
            **request,
            "headers": [
                header
                for header in cast(list[list[str]], request["headers"])
                if header[0].lower() not in lowered_headers
            ],
            "body": _mask_paths(request["body"], spec.request_body_paths),
        }
        for request in requests
    ]
    report: Final = _mask_paths(dumped["report"], spec.report_paths)
    return Execution.model_validate({**dumped, "requests": normalized_requests, "report": report})

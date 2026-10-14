from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm.exceptions import BadRequestError

_FILTERS: Final = TypeAdapter(list[dict[str, object]])
_MATCH_VALUES: Final[TypeAdapter[Sequence[str] | Sequence[int]]] = TypeAdapter(Sequence[str] | Sequence[int])


def configuration_error(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="qdrant")


@dataclass(frozen=True, slots=True)
class _Visit:
    value: Mapping[str, object]
    depth: int


@dataclass(frozen=True, slots=True)
class _Join:
    operator: Literal["must", "should"]
    count: int


def translate_filter(value: Mapping[str, object], depth: int = 0) -> Mapping[str, object]:
    pending: Final[list[_Visit | _Join]] = [_Visit(value, depth)]  # mutable-ok: iterative traversal work stack
    results: Final[list[Mapping[str, object]]] = []  # mutable-ok: completed subtrees are joined in postorder
    while pending:
        task: _Visit | _Join = pending.pop()
        if isinstance(task, _Join):
            joined: Mapping[str, object] = {task.operator: results[-task.count :]}
            del results[-task.count :]
            results.append(joined)
            continue
        if task.depth >= 20:
            raise configuration_error("Qdrant filters may be nested at most 20 levels")
        operator: object = task.value.get("type")
        if operator in ("and", "or"):
            children: Sequence[Mapping[str, object]] = _FILTERS.validate_python(task.value.get("filters"))
            if not children or set(task.value) != {"type", "filters"}:
                raise configuration_error("Compound filters require a non-empty filters array and a type")
            pending.append(_Join("must" if operator == "and" else "should", len(children)))
            pending.extend(_Visit(child, task.depth + 1) for child in reversed(children))
            continue
        key: object = task.value.get("key")
        if not isinstance(key, str) or not key or set(task.value) != {"type", "key", "value"}:
            raise configuration_error("Comparison filters require type, a non-empty key, and value")
        results.append(_comparison_filter(key, operator, task.value.get("value")))
    return results[0]


def _comparison_filter(key: str, operator: object, operand: object) -> Mapping[str, object]:
    match operator:
        case "eq" | "ne":
            if not isinstance(operand, (str, int, bool)):
                raise configuration_error("Qdrant eq/ne filters require a string, integer, or boolean value")
            condition: Final = {"key": key, "match": {"value": operand}}
            return {"must" if operator == "eq" else "must_not": [condition]}
        case "gt" | "gte" | "lt" | "lte":
            if isinstance(operand, bool) or not isinstance(operand, (int, float)) or not isfinite(operand):
                raise configuration_error("Qdrant range filters require a finite number")
            return {"must": [{"key": key, "range": {operator: operand}}]}
        case "in" | "nin":
            if not isinstance(operand, list) or not operand:
                raise configuration_error("Qdrant in/nin filters require a non-empty array")
            items: Final = _MATCH_VALUES.validate_python(operand, strict=True)
            return {"must" if operator == "in" else "must_not": [{"key": key, "match": {"any": items}}]}
        case _:
            raise configuration_error(f"Unsupported Qdrant filter operator '{operator}'")

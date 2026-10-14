from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Final

from pydantic import TypeAdapter

from litellm.exceptions import BadRequestError

_FILTERS: Final = TypeAdapter(list[dict[str, object]])
_MATCH_VALUES: Final[TypeAdapter[Sequence[str] | Sequence[int]]] = TypeAdapter(Sequence[str] | Sequence[int])


def configuration_error(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="qdrant")


def translate_filter(value: Mapping[str, object], depth: int = 0) -> Mapping[str, object]:
    if depth >= 20:
        raise configuration_error("Qdrant filters may be nested at most 20 levels")
    operator: Final = value.get("type")
    if operator in ("and", "or"):
        children: Final = _FILTERS.validate_python(value.get("filters"))
        if not children or set(value) != {"type", "filters"}:
            raise configuration_error("Compound filters require a non-empty filters array and a type")
        return {"must" if operator == "and" else "should": [translate_filter(child, depth + 1) for child in children]}
    key: Final = value.get("key")
    if not isinstance(key, str) or not key or set(value) != {"type", "key", "value"}:
        raise configuration_error("Comparison filters require type, a non-empty key, and value")
    return _comparison_filter(key, operator, value.get("value"))


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

from collections.abc import Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter

JsonObject: TypeAlias = Mapping[str, JsonValue]
JSON_VALUE: Final = TypeAdapter[JsonValue](JsonValue)
JSON_OBJECT: Final = TypeAdapter(JsonObject)
JSON_ARRAY: Final = TypeAdapter(tuple[JsonValue, ...])
STRING: Final = TypeAdapter(str)
INTEGER: Final = TypeAdapter(int)


def object_value(value: JsonValue | JsonObject) -> JsonObject:
    if isinstance(value, Mapping):
        return value
    raise ValueError("Expected a JSON object")


def string_value(value: object) -> str:
    return STRING.validate_python(value, strict=True)

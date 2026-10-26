from collections.abc import Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter

JsonObject: TypeAlias = Mapping[str, JsonValue]
JSON_VALUE: Final = TypeAdapter[JsonValue](JsonValue)
JSON_OBJECT: Final = TypeAdapter(JsonObject)
JSON_ARRAY: Final = TypeAdapter(tuple[JsonValue, ...])
STRING: Final = TypeAdapter(str)
INTEGER: Final = TypeAdapter(int)


def object_value(value: object) -> JsonObject:
    return JSON_OBJECT.validate_python(value, strict=True)


def string_value(value: object) -> str:
    return STRING.validate_python(value, strict=True)

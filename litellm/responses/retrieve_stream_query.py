from dataclasses import dataclass
from typing import Final, Literal

RETRIEVE_STREAM_QUERY_KEYS: Final = frozenset({"stream", "starting_after"})


@dataclass(frozen=True, slots=True)
class RetrieveStreamQuery:
    stream: bool
    starting_after: int | None


@dataclass(frozen=True, slots=True)
class InvalidRetrieveStreamQuery:
    param: Literal["stream", "starting_after"]
    message: str


def _parsed_stream(value: object) -> bool | InvalidRetrieveStreamQuery:
    match value:
        case None:
            return False
        case bool():
            return value
        case str() if value.lower() in ("true", "false"):
            return value.lower() == "true"
        case _:
            return InvalidRetrieveStreamQuery("stream", f"Invalid 'stream': expected true or false, got {value!r}")


def _parsed_starting_after(value: object) -> int | None | InvalidRetrieveStreamQuery:
    match value:
        case None:
            return None
        case int() if not isinstance(value, bool) and value >= 0:
            return value
        case str() if value.isascii() and value.isdigit():
            return int(value)
        case _:
            return InvalidRetrieveStreamQuery(
                "starting_after", f"Invalid 'starting_after': expected an integer >= 0, got {value!r}"
            )


def parse_retrieve_stream_query(
    stream: object, starting_after: object
) -> RetrieveStreamQuery | InvalidRetrieveStreamQuery:
    parsed_stream: Final = _parsed_stream(stream)
    if isinstance(parsed_stream, InvalidRetrieveStreamQuery):
        return parsed_stream
    parsed_starting_after: Final = _parsed_starting_after(starting_after)
    if isinstance(parsed_starting_after, InvalidRetrieveStreamQuery):
        return parsed_starting_after
    return RetrieveStreamQuery(stream=parsed_stream, starting_after=parsed_starting_after)

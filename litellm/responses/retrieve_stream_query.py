from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

RETRIEVE_STREAM_QUERY_KEYS: Final = frozenset({"stream", "starting_after"})
RETRIEVE_INCLUDE_QUERY_KEYS: Final = frozenset({"include", "include[]", "include_obfuscation"})
_QUERY_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_MAX_STARTING_AFTER_DIGITS: Final = len(str(2**63 - 1))


@dataclass(frozen=True, slots=True)
class RetrieveStreamQuery:
    stream: bool
    starting_after: int | None


@dataclass(frozen=True, slots=True)
class InvalidRetrieveStreamQuery:
    param: Literal["stream", "starting_after"]
    message: str


_NO_STREAM: Final = RetrieveStreamQuery(stream=False, starting_after=None)
_FULL_REPLAY: Final = RetrieveStreamQuery(stream=True, starting_after=None)


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
        case str() if value.isascii() and value.isdigit() and len(value) <= _MAX_STARTING_AFTER_DIGITS:
            return int(value)
        case _:
            return InvalidRetrieveStreamQuery(
                "starting_after", f"Invalid 'starting_after': expected an integer >= 0, got {value!r}"
            )


def parse_retrieve_stream_query(
    stream: object, starting_after: object, drop_malformed: bool
) -> RetrieveStreamQuery | InvalidRetrieveStreamQuery:
    parsed_stream: Final = _parsed_stream(stream)
    if isinstance(parsed_stream, InvalidRetrieveStreamQuery):
        return _NO_STREAM if drop_malformed else parsed_stream
    if not parsed_stream:
        return _NO_STREAM
    parsed_starting_after: Final = _parsed_starting_after(starting_after)
    if isinstance(parsed_starting_after, InvalidRetrieveStreamQuery):
        return _FULL_REPLAY if drop_malformed else parsed_starting_after
    return RetrieveStreamQuery(stream=True, starting_after=parsed_starting_after)


def requests_streamed_retrieval(request: Mapping[str, object]) -> bool:
    extra_query: Final = request.get("extra_query")
    return (
        "response_id" in request
        and isinstance(extra_query, Mapping)
        and _parsed_stream(_QUERY_ADAPTER.validate_python(extra_query).get("stream")) is True
    )


def without_retrieve_stream_defaults(
    deployment_params: Mapping[str, object], request: Mapping[str, object]
) -> Mapping[str, object]:
    if "response_id" not in request:
        return deployment_params
    return {key: value for key, value in deployment_params.items() if key not in RETRIEVE_STREAM_QUERY_KEYS}

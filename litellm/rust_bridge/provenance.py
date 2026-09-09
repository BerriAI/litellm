from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeVar

from pydantic import InstanceOf, TypeAdapter, ValidationError
from typing_extensions import TypedDict

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"
RUST_RESPONSE_HEADER_VALUE: Final = "true"

RUST_RESPONSE_ENTRYPOINTS: Final = frozenset(
    {
        "achat_completions",
        "amessages",
        "aocr",
        "atranscription",
        "chat_completions",
        "messages",
        "ocr",
        "transcription",
    }
)
RUST_NON_RESPONSE_ENTRYPOINTS: Final = frozenset(
    {
        "chat_completions_decline",
        "responses_websocket",
    }
)

_ResponseT = TypeVar("_ResponseT")
_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_RESPONSE_DICT_ADAPTER: Final[TypeAdapter[InstanceOf[dict[str, object]]]] = TypeAdapter(
    InstanceOf[dict[str, object]]
)


class RustStreamHiddenParams(TypedDict):
    additional_headers: dict[str, str]


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value, strict=True)
    except ValidationError:
        return None


def rust_response_headers(existing: Mapping[str, object] | None = None) -> dict[str, object]:
    return {
        **(existing or {}),
        RUST_RESPONSE_HEADER: RUST_RESPONSE_HEADER_VALUE,
    }


def rust_response_hidden_params(existing: Mapping[str, object] | None = None) -> dict[str, object]:
    current: Final = existing or {}
    headers: Final = _mapping(current.get("additional_headers"))
    return {
        **current,
        "additional_headers": rust_response_headers(headers),
    }


def rust_stream_hidden_params(existing: Mapping[str, str] | None = None) -> RustStreamHiddenParams:
    return RustStreamHiddenParams(
        additional_headers={
            **(existing or {}),
            RUST_RESPONSE_HEADER: RUST_RESPONSE_HEADER_VALUE,
        }
    )


def mark_rust_response(response: _ResponseT) -> _ResponseT:
    original: Final[_ResponseT] = response
    if isinstance(response, dict):
        response_dict: Final = _RESPONSE_DICT_ADAPTER.validate_python(response)
        mapping_hidden: Final = _mapping(response_dict.get("_hidden_params"))
        response_dict["_hidden_params"] = rust_response_hidden_params(mapping_hidden)
    else:
        object_hidden: Final[object] = getattr(response, "_hidden_params", None)
        setattr(
            response,
            "_hidden_params",
            rust_response_hidden_params(_mapping(object_hidden)),
        )
    return original


def has_rust_response_marker(response: object) -> bool:
    direct: Final[object] = getattr(response, "_hidden_params", None)
    response_mapping: Final = _mapping(response)
    hidden: Final = _mapping(direct) or _mapping(
        response_mapping.get("_hidden_params") if response_mapping is not None else None
    )
    headers: Final = _mapping(hidden.get("additional_headers")) if hidden is not None else None
    return headers is not None and headers.get(RUST_RESPONSE_HEADER) == RUST_RESPONSE_HEADER_VALUE

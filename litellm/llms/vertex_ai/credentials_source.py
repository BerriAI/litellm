import base64
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.secret_redaction import redact_string


@dataclass(frozen=True, slots=True)
class VertexCredentialsJson:
    value: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VertexCredentialsFileUnreadable:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsFileNotJson:
    path: str
    detail: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsInlineNotJson:
    detail: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsUnusable:
    problem: str


@dataclass(frozen=True, slots=True)
class _NotAJsonObject:
    detail: str


VertexCredentialsFailure: TypeAlias = (
    VertexCredentialsFileUnreadable
    | VertexCredentialsFileNotJson
    | VertexCredentialsInlineNotJson
    | VertexCredentialsUnusable
)
VertexCredentialsSource: TypeAlias = VertexCredentialsJson | VertexCredentialsFailure
_VertexCredentialsFile: TypeAlias = (
    VertexCredentialsJson | VertexCredentialsFileUnreadable | VertexCredentialsFileNotJson
)


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_INLINE_JSON_LEADING_NOISE: Final = " \t\r\n﻿\"'"


def _parse_json_object(raw: str) -> Mapping[str, object] | _NotAJsonObject:
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(raw)
    except ValidationError as e:
        return _NotAJsonObject("; ".join(detail["msg"] for detail in e.errors()))


def _read_json_file(path: str) -> _VertexCredentialsFile:
    try:
        with open(path, encoding="utf-8") as f:
            contents: Final = f.read()
    except OSError as e:
        return VertexCredentialsFileUnreadable(path, f"{e.strerror or e} ({type(e).__name__})")
    except UnicodeDecodeError:
        return VertexCredentialsFileNotJson(path, "file is not UTF-8 text")
    except ValueError as e:
        return VertexCredentialsFileUnreadable(path, f"{e} ({type(e).__name__})")

    parsed: Final = _parse_json_object(contents)
    if isinstance(parsed, _NotAJsonObject):
        return VertexCredentialsFileNotJson(path, parsed.detail)
    return VertexCredentialsJson(parsed)


def _is_inline_credentials_json(credentials: str) -> bool:
    return credentials.lstrip(_INLINE_JSON_LEADING_NOISE).startswith("{")


def _is_base64_encoded_json(credentials: str) -> bool:
    try:
        decoded: Final = base64.b64decode("".join(credentials.split()), validate=True)
    except ValueError:
        return False
    return decoded.lstrip().startswith(b"{")


def _load_inline_credentials(credentials: str) -> VertexCredentialsSource:
    inline: Final = _parse_json_object(credentials)
    if not isinstance(inline, _NotAJsonObject):
        return VertexCredentialsJson(inline)

    from_file_named_like_json: Final = _read_json_file(credentials)
    if isinstance(from_file_named_like_json, VertexCredentialsFileUnreadable):
        return VertexCredentialsInlineNotJson(inline.detail)
    return from_file_named_like_json


def _load_credentials_file(credentials: str) -> VertexCredentialsSource:
    from_file: Final = _read_json_file(credentials)
    if isinstance(from_file, VertexCredentialsFileUnreadable) and _is_base64_encoded_json(credentials):
        return VertexCredentialsUnusable("is base64-encoded JSON, not the JSON itself")
    return from_file


def load_vertex_credentials_source(credentials: str) -> VertexCredentialsSource:
    if not credentials.strip():
        return VertexCredentialsUnusable("is empty")

    inline_first: Final = _is_inline_credentials_json(credentials)
    verbose_logger.debug(
        "Vertex: Loading vertex credentials, is_file_path=%s, current dir %s", not inline_first, os.getcwd()
    )
    if inline_first:
        return _load_inline_credentials(credentials)
    return _load_credentials_file(credentials)


def raise_vertex_credentials_failure(failure: VertexCredentialsFailure) -> NoReturn:
    match failure:
        case VertexCredentialsFileUnreadable():
            verbose_logger.error(
                "Vertex: cannot read the credentials file at %s: %s", redact_string(failure.path), failure.reason
            )
            raise ValueError(
                f"Unable to read the vertex credentials file: {failure.reason}. LiteLLM's log names the path. "
                "Set `vertex_credentials` to a readable file path, or to the credentials JSON itself."
            )
        case VertexCredentialsFileNotJson():
            verbose_logger.error(
                "Vertex: credentials file at %s is not valid JSON: %s", redact_string(failure.path), failure.detail
            )
            raise ValueError(
                f"The vertex credentials file is not valid JSON: {failure.detail}. LiteLLM's log names the path. "
                "Check for unescaped newlines in private_key."
            )
        case VertexCredentialsInlineNotJson():
            raise ValueError(
                f"The inline `vertex_credentials` value is not valid JSON: {failure.detail}. "
                "Check for unescaped newlines in private_key."
            )
        case VertexCredentialsUnusable():
            raise ValueError(
                f"The `vertex_credentials` value {failure.problem}. "
                "Set it to a readable file path, or to the credentials JSON itself."
            )
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(failure)  # pragma: no cover

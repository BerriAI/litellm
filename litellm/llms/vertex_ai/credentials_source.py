import base64
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias, TypeVar

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.secret_redaction import redact_string


@dataclass(frozen=True, slots=True)
class VertexCredentialsJson:
    value: Mapping[str, object]
    path: str | None


@dataclass(frozen=True, slots=True)
class VertexCredentialsFileUnreadable:
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsFileInvalid:
    path: str
    problem: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsInlineNotJson:
    detail: str
    hint: str


@dataclass(frozen=True, slots=True)
class VertexCredentialsUnusable:
    problem: str


@dataclass(frozen=True, slots=True)
class _InvalidJson:
    detail: str


@dataclass(frozen=True, slots=True)
class _JsonNotAnObject:
    pass


VertexCredentialsFailure: TypeAlias = (
    VertexCredentialsFileUnreadable
    | VertexCredentialsFileInvalid
    | VertexCredentialsInlineNotJson
    | VertexCredentialsUnusable
)
VertexCredentialsSource: TypeAlias = VertexCredentialsJson | VertexCredentialsFailure
_VertexCredentialsFile: TypeAlias = (
    VertexCredentialsJson | VertexCredentialsFileUnreadable | VertexCredentialsFileInvalid
)
_ParsedJson: TypeAlias = Mapping[str, object] | _InvalidJson | _JsonNotAnObject
_Credentials = TypeVar("_Credentials")


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_INLINE_JSON_LEADING_NOISE: Final = " \t\r\n﻿\"'"
_BASE64_PREFIX: Final = "base64:"
_MAX_LOGGED_PATH_LENGTH: Final = 260
_MAX_CREDENTIALS_FILE_CHARACTERS: Final = 1 << 20
_NOT_A_JSON_OBJECT: Final = "is JSON but not a JSON object, it may be quoted or encoded twice"
_PRIVATE_KEY_HINT: Final = "Check for unescaped newlines in private_key."
_LEADING_NOISE_HINT: Final = "Remove the quotes or byte-order mark before the opening `{`."
_FILE_FAILURE_MESSAGE: Final = (
    "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason."
)


def _parse_json_object(raw: str | bytes) -> _ParsedJson:
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(raw)
    except ValidationError as e:
        errors: Final = e.errors()
        if any(error["type"] == "json_invalid" for error in errors):
            return _InvalidJson("; ".join(error["msg"] for error in errors))
        return _JsonNotAnObject()


def _read_json_file(path: str) -> _VertexCredentialsFile:
    try:
        with open(path, encoding="utf-8") as f:
            contents: Final = f.read(_MAX_CREDENTIALS_FILE_CHARACTERS + 1)
    except OSError as e:
        return VertexCredentialsFileUnreadable(path, f"{e.strerror or 'OS error'} ({type(e).__name__})")
    except UnicodeDecodeError:
        return VertexCredentialsFileInvalid(path, "is not UTF-8 text")
    except ValueError as e:
        return VertexCredentialsFileUnreadable(path, f"{e} ({type(e).__name__})")

    if len(contents) > _MAX_CREDENTIALS_FILE_CHARACTERS:
        return VertexCredentialsFileInvalid(
            path, f"is over {_MAX_CREDENTIALS_FILE_CHARACTERS} characters, larger than any credentials file"
        )
    parsed: Final = _parse_json_object(contents)
    match parsed:
        case _InvalidJson(detail=detail):
            return VertexCredentialsFileInvalid(path, f"is not valid JSON: {detail}. {_PRIVATE_KEY_HINT}")
        case _JsonNotAnObject():
            return VertexCredentialsFileInvalid(path, _NOT_A_JSON_OBJECT)
        case _:
            return VertexCredentialsJson(parsed, path)


def _is_inline_credentials_json(credentials: str) -> bool:
    return credentials.lstrip(_INLINE_JSON_LEADING_NOISE).startswith("{")


def _is_base64_encoded_json(credentials: str) -> bool:
    encoded: Final = "".join(credentials.split()).strip("\"'").removeprefix(_BASE64_PREFIX)
    standard_alphabet: Final = encoded.replace("-", "+").replace("_", "/")
    try:
        decoded: Final = base64.b64decode(standard_alphabet + "=" * (-len(encoded) % 4), validate=True)
    except ValueError:
        return False
    return isinstance(_parse_json_object(decoded), Mapping)


def _load_inline_credentials(credentials: str) -> VertexCredentialsSource:
    inline: Final = _parse_json_object(credentials)
    if isinstance(inline, Mapping):
        return VertexCredentialsJson(inline, None)
    if os.path.isfile(credentials):
        return _read_json_file(credentials)

    match inline:
        case _JsonNotAnObject():
            return VertexCredentialsUnusable(_NOT_A_JSON_OBJECT)
        case _InvalidJson(detail=detail) if credentials.lstrip().startswith("{"):
            return VertexCredentialsInlineNotJson(detail, _PRIVATE_KEY_HINT)
        case _InvalidJson(detail=detail):
            return VertexCredentialsInlineNotJson(detail, _LEADING_NOISE_HINT)
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(inline)  # pragma: no cover


def load_vertex_credentials_source(credentials: str) -> VertexCredentialsSource:
    if not credentials.strip():
        return VertexCredentialsUnusable("is empty")

    inline_first: Final = _is_inline_credentials_json(credentials)
    verbose_logger.debug(
        "Vertex: Loading vertex credentials, is_file_path=%s, current dir %s", not inline_first, os.getcwd()
    )
    if inline_first:
        return _load_inline_credentials(credentials)
    if _is_base64_encoded_json(credentials):
        return VertexCredentialsUnusable("is base64-encoded JSON, not the JSON itself")
    return _read_json_file(credentials)


def build_vertex_credentials(
    source: VertexCredentialsJson, build: Callable[[Mapping[str, object]], _Credentials]
) -> _Credentials:
    if source.path is None:
        return build(source.value)
    try:
        return build(source.value)
    except ValueError as e:
        rejected: Final = VertexCredentialsFileInvalid(source.path, f"holds credentials that cannot be used: {e}")
    raise_vertex_credentials_failure(rejected)


def _path_for_log(path: str) -> str:
    if len(path) > _MAX_LOGGED_PATH_LENGTH:
        return f"<a {len(path)}-character value>"
    return repr(redact_string(path))


def raise_vertex_credentials_failure(failure: VertexCredentialsFailure) -> NoReturn:
    match failure:
        case VertexCredentialsFileUnreadable():
            verbose_logger.error(
                "Vertex: cannot read the credentials file at %s: %s", _path_for_log(failure.path), failure.reason
            )
            raise ValueError(_FILE_FAILURE_MESSAGE)
        case VertexCredentialsFileInvalid():
            verbose_logger.error("Vertex: the credentials file at %s %s", _path_for_log(failure.path), failure.problem)
            raise ValueError(_FILE_FAILURE_MESSAGE)
        case VertexCredentialsInlineNotJson():
            raise ValueError(
                f"The inline `vertex_credentials` value is not valid JSON: {failure.detail}. {failure.hint}"
            )
        case VertexCredentialsUnusable():
            raise ValueError(
                f"The `vertex_credentials` value {failure.problem}. "
                "Set it to a readable file path, or to the credentials JSON itself."
            )
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(failure)  # pragma: no cover

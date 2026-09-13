"""
Resolve the `vertex_credentials` config value into the JSON object google-auth needs.

The value is either the credentials JSON itself or a path to a file holding it. Which one
it is decides what a failure means, so the two are told apart by the shape of the value and
each failure is returned as its own case instead of collapsing into one parse error.
"""

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
class _NotAJsonObject:
    detail: str


VertexCredentialsFailure: TypeAlias = (
    VertexCredentialsFileUnreadable | VertexCredentialsFileNotJson | VertexCredentialsInlineNotJson
)
VertexCredentialsSource: TypeAlias = VertexCredentialsJson | VertexCredentialsFailure
_VertexCredentialsFile: TypeAlias = (
    VertexCredentialsJson | VertexCredentialsFileUnreadable | VertexCredentialsFileNotJson
)


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parse_json_object(raw: str) -> Mapping[str, object] | _NotAJsonObject:
    """Parse *raw*, describing any failure without echoing it: the input can be key material."""
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(raw)
    except ValidationError as e:
        # Only "msg" is reported. Pydantic keeps the offending value under "input", and that
        # value is the credential.
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
        # open() rejects a few paths before touching the filesystem, e.g. "embedded null byte".
        return VertexCredentialsFileUnreadable(path, f"{e} ({type(e).__name__})")

    parsed: Final = _parse_json_object(contents)
    if isinstance(parsed, _NotAJsonObject):
        return VertexCredentialsFileNotJson(path, parsed.detail)
    return VertexCredentialsJson(parsed)


def _is_inline_credentials_json(credentials: str) -> bool:
    """Whether *credentials* carries the JSON itself rather than a path to a file holding it."""
    return credentials.lstrip().startswith("{")


def load_vertex_credentials_source(credentials: str) -> VertexCredentialsSource:
    """Read *credentials* as the credentials JSON when it is shaped like one, else as a path to it.

    Telling the two apart by shape rather than by `os.path.exists()` is what keeps a read
    failure recognisable: `os.path.exists()` answers False for an unreadable path as well as
    an absent one, so both used to reach the inline branch and be reported as malformed JSON.
    """
    inline_first: Final = _is_inline_credentials_json(credentials)
    verbose_logger.debug(
        "Vertex: Loading vertex credentials, is_file_path=%s, current dir %s", not inline_first, os.getcwd()
    )
    if not inline_first:
        return _read_json_file(credentials)

    inline: Final = _parse_json_object(credentials)
    if not isinstance(inline, _NotAJsonObject):
        return VertexCredentialsJson(inline)

    # A file can legitimately be named "{vertex}.json", so a brace-prefixed value that does
    # not parse is still given the file read it used to get before dispatch moved to shape.
    from_file: Final = _read_json_file(credentials)
    if isinstance(from_file, VertexCredentialsJson):
        return from_file
    return VertexCredentialsInlineNotJson(inline.detail)


def raise_vertex_credentials_failure(failure: VertexCredentialsFailure) -> NoReturn:
    """Map a load failure onto the ValueError the auth flow already surfaces to the caller.

    The caller is told which of the three faults it was; the path goes to the proxy log
    instead, because the message reaches whoever sent the request and the operator who can
    act on the path is reading the log anyway. The path is redacted on the way there too, so
    a credential misconfigured into this field does not become a log entry.
    """
    match failure:
        case VertexCredentialsFileUnreadable(path=path, reason=reason):
            verbose_logger.error("Vertex: cannot read the credentials file at %s: %s", redact_string(path), reason)
            raise ValueError(
                f"Unable to read the vertex credentials file: {reason}. The proxy log names the path. "
                "Set `vertex_credentials` to a readable file path, or to the credentials JSON itself."
            )
        case VertexCredentialsFileNotJson(path=path, detail=detail):
            verbose_logger.error("Vertex: credentials file at %s is not valid JSON: %s", redact_string(path), detail)
            raise ValueError(
                f"The vertex credentials file is not valid JSON: {detail}. The proxy log names the path. "
                "Check for unescaped newlines in private_key."
            )
        case VertexCredentialsInlineNotJson(detail=detail):
            raise ValueError(
                f"The inline `vertex_credentials` value is not valid JSON: {detail}. "
                "Check for unescaped newlines in private_key."
            )
        case _:  # pragma: no cover - exhaustiveness guard, unreachable while the union holds
            assert_never(failure)  # pragma: no cover

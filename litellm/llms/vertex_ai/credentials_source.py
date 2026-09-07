"""
Resolve the `vertex_credentials` config value into the JSON object google-auth needs.

The value is either the credentials JSON itself or a path to a file holding it. Which one
it is decides what a failure means, so the two are told apart by the shape of the value and
each failure is returned as its own case instead of collapsing into one parse error.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

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


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parse_json_object(raw: str) -> Mapping[str, object] | _NotAJsonObject:
    """Parse *raw*, describing any failure without echoing it: the input can be key material."""
    try:
        return _JSON_OBJECT_ADAPTER.validate_json(raw)
    except ValidationError as e:
        # Only "msg" is reported. Pydantic keeps the offending value under "input", and that
        # value is the credential.
        return _NotAJsonObject("; ".join(detail["msg"] for detail in e.errors()))


def is_inline_credentials_json(credentials: str) -> bool:
    """Whether *credentials* carries the JSON itself rather than a path to a file holding it."""
    return credentials.lstrip().startswith("{")


def load_vertex_credentials_source(credentials: str) -> VertexCredentialsSource:
    """Read *credentials* as the credentials JSON when it is shaped like one, else as a path to it.

    Telling the two apart by shape rather than by `os.path.exists()` is what keeps a read
    failure recognisable: `os.path.exists()` answers False for an unreadable path as well as
    an absent one, so both used to reach the inline branch and be reported as malformed JSON.
    """
    if is_inline_credentials_json(credentials):
        inline: Final = _parse_json_object(credentials)
        if isinstance(inline, _NotAJsonObject):
            return VertexCredentialsInlineNotJson(inline.detail)
        return VertexCredentialsJson(inline)

    try:
        with open(credentials, encoding="utf-8") as f:
            contents: Final = f.read()
    except OSError as e:
        return VertexCredentialsFileUnreadable(credentials, f"{e.strerror or e} ({type(e).__name__})")
    except UnicodeDecodeError:
        return VertexCredentialsFileNotJson(credentials, "file is not UTF-8 text")

    from_file: Final = _parse_json_object(contents)
    if isinstance(from_file, _NotAJsonObject):
        return VertexCredentialsFileNotJson(credentials, from_file.detail)
    return VertexCredentialsJson(from_file)


def raise_vertex_credentials_failure(failure: VertexCredentialsFailure) -> NoReturn:
    """Map a load failure onto the ValueError the auth flow already surfaces as a 500.

    A path names itself in the message so the operator's log says which file failed. The
    proxy scrubs filesystem paths out of what it hands back to the API caller, and the value
    is redacted first so a non-path value misconfigured here cannot leak instead.
    """
    match failure:
        case VertexCredentialsFileUnreadable(path=path, reason=reason):
            raise ValueError(
                f"Unable to read the vertex credentials file at {redact_string(path)}: {reason}. "
                "Set `vertex_credentials` to a readable file path, or to the credentials JSON itself."
            )
        case VertexCredentialsFileNotJson(path=path, detail=detail):
            raise ValueError(
                f"The vertex credentials file at {redact_string(path)} is not valid JSON: {detail}. "
                "Check for unescaped newlines in private_key."
            )
        case VertexCredentialsInlineNotJson(detail=detail):
            raise ValueError(
                f"The inline `vertex_credentials` value is not valid JSON: {detail}. "
                "Check for unescaped newlines in private_key."
            )
        case _:
            assert_never(failure)

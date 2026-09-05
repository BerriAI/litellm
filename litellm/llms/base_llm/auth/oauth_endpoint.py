"""OAuth 2.0 token endpoint hygiene shared by every grant LiteLLM posts itself: HTTPS pinning,
RFC 6749 5.2 error redaction with the sent credential scrubbed out of an echoing response, and the
response guard a raw ``httpx`` poster needs. Providers map the resulting typed values onto their
own public exception contract."""

import re
from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError

from litellm.llms.base_llm.auth.types import InsecureTokenUrl, TokenEndpointError

MAX_RESPONSE_BYTES: Final = 1024 * 1024

_REDACTION_CAP: Final = 256
_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
_OAUTH_ERROR_FIELDS: Final = ("error", "error_description", "error_uri")
_NESTED_ERROR_FIELDS: Final = ("type", "message")
_OVERSIZED_BODY_MESSAGE: Final = "oversized error response omitted"
_NON_OBJECT_BODY_MESSAGE: Final = "non-object error response omitted"
_NO_OAUTH_FIELDS_MESSAGE: Final = "error response carried no RFC 6749 fields"
_UNSTRUCTURED_BODY_MESSAGE: Final = "non-JSON error response omitted"
_REFLECTED_VALUE_MESSAGE: Final = "<redacted: response echoed the request>"
# A credential fragment shorter than this is not worth the false positives; longer, and a run
# shared with the assertion is reflection rather than coincidence.
_REFLECTION_MIN_RUN: Final = 8
# Everything a base64url credential is NOT made of, stripped so a fragment split by delimiters
# still lines up against the assertion.
_CREDENTIAL_CHARS: Final = re.compile(r"[^A-Za-z0-9._~+/=-]")
_SENTINEL_BODY_MESSAGES: Final = frozenset({_OVERSIZED_BODY_MESSAGE, _NON_OBJECT_BODY_MESSAGE})


_RedactableBody: TypeAlias = Mapping[str, object] | list[object] | str | int | float | bool | None
_REDACTABLE_BODY_ADAPTER: Final = TypeAdapter[_RedactableBody](_RedactableBody)


def endpoint_url_for_error_message(url: str) -> str:
    """``url`` reduced to scheme, host and path for operator-facing errors.

    A token endpoint is configuration, not a secret, and naming it is what makes these errors
    actionable. But nothing stops an operator writing a credential into it, as a query parameter
    or as userinfo, and these errors reach model callers, so neither part is echoed.
    """
    parsed: Final = urlsplit(url)
    host: Final = parsed.hostname or ""
    authority: Final = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def validate_token_endpoint_url(url: str) -> str | InsecureTokenUrl:
    parsed: Final = urlsplit(url)
    if parsed.scheme == "https":
        return url
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOCAL_HOSTS:
        return url
    return InsecureTokenUrl(host=parsed.hostname or "")


def redact_oauth_error_body(
    status_code: int,
    body_text: str,
    assertion: SecretStr | Sequence[SecretStr] | None = None,
) -> TokenEndpointError:
    """``assertion`` may be every form of the credential that went out on the wire.

    A grant that encodes its credential before sending it (``client_secret_basic`` base64s
    ``id:secret``) can have that encoded form echoed back, and it decodes straight to the secret,
    so checking only the raw value lets reversible material through.
    """
    rendered: Final = _redact_body_text(body_text)
    secrets: Final = () if assertion is None else (assertion,) if isinstance(assertion, SecretStr) else tuple(assertion)
    redacted: Final = next(
        (
            _REFLECTED_VALUE_MESSAGE
            for secret in secrets
            if drop_reflected_credential(rendered, secret) is _REFLECTED_VALUE_MESSAGE
        ),
        rendered,
    )
    return TokenEndpointError(status_code=status_code, redacted_body=redacted)


def drop_reflected_credential(rendered: str, assertion: SecretStr | None) -> str:
    """Catches an endpoint that echoes the submitted credential back, verbatim or in fragments,
    however it split or percent-encoded it.

    Both sides are reduced to the characters a credential is made of before comparison. Stripping
    only the rendered side would stop matching a secret that carries spaces or punctuation of its
    own, which is exactly the hand-set passphrase most at risk of being echoed.

    This stops an accidental or naive echo. It cannot stop an endpoint that deliberately re-encodes
    or interleaves the credential, and it is not what keeps the credential from the endpoint, which
    already holds it. What it protects is blast radius: keeping the value out of the caller's error
    and out of third-party log sinks.
    """
    if assertion is None:
        return rendered
    secret: Final = assertion.get_secret_value()
    if not secret:
        return rendered
    if secret in rendered:
        return _REFLECTED_VALUE_MESSAGE
    compacted_secret: Final = _CREDENTIAL_CHARS.sub("", secret)
    if not compacted_secret:
        return rendered
    return _REFLECTED_VALUE_MESSAGE if _shares_a_credential_run(rendered, compacted_secret) else rendered


def _shares_a_credential_run(rendered: str, compacted_secret: str) -> bool:
    """``unquote`` covers a credential sent form-encoded, without every caller enumerating that
    shape for itself: percent-escaping is reversible and applies to any field, query string
    included.

    A secret shorter than the probe run is compared whole: a window longer than the secret can
    never be found inside it, which would leave a short client secret unprotected in every shape
    but the verbatim one.
    """
    # unquote covers %XX; unquote_plus additionally covers the "+" a form-encoded body uses for a
    # space. Both are kept rather than only the wider one, because "+" is a base64 character and
    # decoding it away would lose a run that the undecoded candidate still matches on.
    run: Final = min(_REFLECTION_MIN_RUN, len(compacted_secret))
    compacted_candidates: Final = tuple(
        _CREDENTIAL_CHARS.sub("", candidate) for candidate in (rendered, unquote(rendered), unquote_plus(rendered))
    )
    return any(
        compacted[start : start + run] in compacted_secret
        for compacted in compacted_candidates
        for start in range(len(compacted) - run + 1)
    )


def _redact_body_text(body_text: str) -> str:
    if body_text in _SENTINEL_BODY_MESSAGES:
        return body_text
    if len(body_text) > MAX_RESPONSE_BYTES:
        return _OVERSIZED_BODY_MESSAGE
    try:
        parsed: Final = _REDACTABLE_BODY_ADAPTER.validate_json(body_text)
    except ValidationError:
        return _UNSTRUCTURED_BODY_MESSAGE
    match parsed:
        case Mapping():
            return _format_oauth_error_fields(parsed)
        case _:
            return _NON_OBJECT_BODY_MESSAGE


def _format_oauth_error_fields(body: Mapping[str, object]) -> str:
    fields: Final = tuple(
        f"{name}: {_format_oauth_error_value(value)}"
        for name in _OAUTH_ERROR_FIELDS
        for value in (body.get(name),)
        if value is not None
    )
    return "; ".join(fields) if fields else _NO_OAUTH_FIELDS_MESSAGE


def _format_oauth_error_value(value: object) -> str:
    """RFC 6749 types ``error`` as a string, but Anthropic (and other providers) nest their
    own ``{"type": ..., "message": ...}`` envelope there; render that rather than a dict repr."""
    if isinstance(value, Mapping):
        nested: Final = tuple(
            f"{str(part)[:_REDACTION_CAP]}"
            for key in _NESTED_ERROR_FIELDS
            for part in (value.get(key),)
            if part is not None
        )
        if nested:
            return " - ".join(nested)
    return str(value)[:_REDACTION_CAP]


def require_posted_response(response: httpx.Response | None, endpoint_label: str) -> httpx.Response:
    """The legacy ``HTTPHandler`` carries no return annotation, so a patched or stubbed client can
    hand a poster ``None`` back; a transport error beats dereferencing it."""
    if response is None:
        raise httpx.TransportError(f"{endpoint_label} returned no response")
    return response

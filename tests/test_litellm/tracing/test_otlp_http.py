import gzip
from typing import Final
from unittest.mock import patch

import pytest

from litellm.tracing import otlp_http
from litellm.tracing.otlp_http import (
    InvalidOTLPPayloadError,
    TracingPayloadTooLargeError,
    decompress,
    encode_otlp_response,
)

BODY: Final = b'{"resourceSpans": []}'


@pytest.mark.parametrize("encoding", (None, "identity", "IDENTITY"))
def test_identity_body_is_unchanged(encoding: str | None) -> None:
    assert decompress(BODY, encoding) == BODY


def test_gzip_body_is_decompressed_by_header() -> None:
    assert decompress(gzip.compress(BODY), "gzip") == BODY


def test_concatenated_gzip_members_are_decoded() -> None:
    midpoint: Final = len(BODY) // 2
    assert decompress(gzip.compress(BODY[:midpoint]) + gzip.compress(BODY[midpoint:]), "gzip") == BODY


@pytest.mark.parametrize(("body", "encoding"), ((b"not gzip", "gzip"), (BODY, "br"), (BODY, "gzip, identity")))
def test_invalid_or_unsupported_encoding_is_rejected(body: bytes, encoding: str) -> None:
    with pytest.raises(InvalidOTLPPayloadError):
        decompress(body, encoding)


@pytest.mark.parametrize(
    ("body", "encoding"),
    ((b" " * 2048, None), (gzip.compress(b" " * 16384, mtime=0), "gzip")),
)
def test_body_and_expansion_respect_the_body_limit(body: bytes, encoding: str | None) -> None:
    with patch.object(otlp_http, "OTLP_MAX_BODY_BYTES", 1024):
        with pytest.raises(TracingPayloadTooLargeError):
            decompress(body, encoding)


def test_response_matches_request_encoding() -> None:
    assert encode_otlp_response("application/json") == (b"{}", "application/json")
    assert encode_otlp_response("application/json; charset=utf-8", "bad") == (
        b'{"message": "bad"}',
        "application/json",
    )
    assert encode_otlp_response("application/x-protobuf") == (b"", "application/x-protobuf")
    assert encode_otlp_response(None) == (b"", "application/x-protobuf")


@pytest.mark.requires_rust_extension
def test_protobuf_error_is_an_rpc_status() -> None:
    from google.rpc.status_pb2 import Status

    body, media_type = encode_otlp_response("application/x-protobuf", "invalid trace")
    assert media_type == "application/x-protobuf"
    assert Status.FromString(body).message == "invalid trace"

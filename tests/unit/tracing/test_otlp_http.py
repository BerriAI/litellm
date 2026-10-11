import pytest

from litellm.tracing.otlp_http import encode_otlp_response


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

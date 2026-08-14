import httpx
from httpx._decoders import SUPPORTED_DECODERS

from litellm.passthrough.utils import BasePassthroughUtils

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _build_upstream_request(forwarded_headers: dict) -> httpx.Request:
    with httpx.Client(headers={"user-agent": "litellm/test"}) as client:
        return client.build_request("POST", ANTHROPIC_MESSAGES_URL, headers=forwarded_headers)


def test_client_accept_encoding_is_not_forwarded_upstream():
    headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers={
            "accept-encoding": "br",
            "anthropic-version": "2023-06-01",
            "host": "localhost:4000",
            "content-length": "123",
        },
        headers={"x-api-key": "sk-anthropic"},
        forward_headers=True,
    )

    assert "accept-encoding" not in {name.lower() for name in headers}
    assert headers["anthropic-version"] == "2023-06-01"
    assert "host" not in headers
    assert "content-length" not in headers


def test_client_accept_encoding_is_not_forwarded_via_x_pass_prefix():
    headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers={"x-pass-accept-encoding": "br"},
        headers={},
        forward_headers=False,
    )

    assert "accept-encoding" not in headers


def test_upstream_request_only_advertises_decodable_encodings():
    """A content coding httpx cannot decode would reach the client still compressed,
    with Content-Encoding stripped by get_response_headers (LIT-5613)."""
    forwarded_headers = BasePassthroughUtils.forward_headers_from_request(
        request_headers={"accept-encoding": "br, zstd, exotic"},
        headers={"x-api-key": "sk-anthropic"},
        forward_headers=True,
    )

    advertised = {
        value.strip().lower()
        for value in _build_upstream_request(forwarded_headers).headers["accept-encoding"].split(",")
    }

    assert advertised
    assert advertised <= set(SUPPORTED_DECODERS)

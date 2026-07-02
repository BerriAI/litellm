"""Tests for the cache codec: encrypt on encode, drop the refresh_token, round-trip the bearer."""

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)


def _wrapping_codec():
    # A reversible stand-in for NaCl: proves encode() encrypts (output is wrapped) and decode()
    # decrypts, without needing a salt key.
    return OAuthTokenCacheCodec(
        encrypt=lambda s: f"enc:{s}",
        decrypt=lambda b: b[4:] if b.startswith("enc:") else None,
    )


def test_round_trips_the_access_token():
    codec = _wrapping_codec()
    token = codec.decode(
        codec.encode(OAuthToken(access_token="at-123", expires_at=1234.5))
    )
    assert token is not None
    assert token.access_token == "at-123"


def test_encode_encrypts_and_omits_the_refresh_token():
    codec = _wrapping_codec()
    blob = codec.encode(OAuthToken(access_token="at", refresh_token="super-secret-rt"))
    assert blob.startswith("enc:")  # encryption was applied
    assert (
        "super-secret-rt" not in blob
    )  # the long-lived secret never reaches the cache
    decoded = codec.decode(blob)
    assert decoded is not None and decoded.refresh_token is None


def test_decoded_token_defers_expiry_to_the_cache_ttl():
    codec = _wrapping_codec()
    token = codec.decode(codec.encode(OAuthToken(access_token="at", expires_at=999.0)))
    # The value carries no expiry; the cache entry's TTL bounds its life instead.
    assert token is not None and token.expires_at is None


def test_undecryptable_blob_is_a_miss():
    # e.g. master-key rotation makes an old entry unreadable -> treat as a miss, not a crash.
    assert _wrapping_codec().decode("not-our-prefix") is None


def test_empty_plaintext_is_a_miss():
    codec = OAuthTokenCacheCodec(encrypt=lambda s: s, decrypt=lambda b: b)
    assert codec.decode("") is None

"""Parity tests: Rust auth must produce identical results to Python.

These tests verify that the Rust bridge implementation matches the Python
implementation for token hashing.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from litellm.proxy.utils import hash_token as python_hash_token
from litellm.rust_bridge.auth import try_rust_hash_token


@pytest.fixture(autouse=True)
def enable_rust_auth(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST_AUTH", "1")


class TestRustMatchesPythonHash:
    @pytest.mark.parametrize(
        "token",
        [
            "sk-test-key-123",
            "sk-1234567890abcdef",
            "sk-" + "a" * 100,
            "sk-!@#$%^&*()",
            "sk-unicode-áéíóú-日本語",
        ],
    )
    def test_rust_matches_python_hash(self, token):
        python_hash = python_hash_token(token)
        rust_hash = try_rust_hash_token(token)

        if rust_hash is None:
            pytest.skip("Rust bridge unavailable")

        assert rust_hash == python_hash, (
            f"Rust ({rust_hash}) != Python ({python_hash}) for token={token[:20]!r}"
        )

    def test_hash_is_64_char_hex(self):
        rust_hash = try_rust_hash_token("sk-test")
        if rust_hash is None:
            pytest.skip("Rust bridge unavailable")

        assert len(rust_hash) == 64
        assert all(c in "0123456789abcdef" for c in rust_hash)

    def test_hash_is_deterministic(self):
        hash1 = try_rust_hash_token("sk-test-key")
        hash2 = try_rust_hash_token("sk-test-key")

        if hash1 is None or hash2 is None:
            pytest.skip("Rust bridge unavailable")

        assert hash1 == hash2

    def test_different_tokens_different_hashes(self):
        hash1 = try_rust_hash_token("sk-key-1")
        hash2 = try_rust_hash_token("sk-key-2")

        if hash1 is None or hash2 is None:
            pytest.skip("Rust bridge unavailable")

        assert hash1 != hash2

    def test_matches_hashlib_sha256_directly(self):
        token = "sk-direct-test"
        rust_hash = try_rust_hash_token(token)

        if rust_hash is None:
            pytest.skip("Rust bridge unavailable")

        expected = hashlib.sha256(token.encode()).hexdigest()
        assert rust_hash == expected


class TestRustFallback:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST_AUTH", raising=False)
        result = try_rust_hash_token("sk-test")
        assert result is None, "Rust should be disabled without env var"

    def test_python_always_works(self):
        result = python_hash_token("sk-test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

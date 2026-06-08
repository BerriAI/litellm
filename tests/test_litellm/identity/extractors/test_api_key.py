import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.identity.extractors.api_key import (
    extract_api_key_principal,
    hash_principal_token,
)
from litellm.proxy._types import UserAPIKeyAuth


def test_returns_none_for_empty_input():
    assert extract_api_key_principal(None) is None
    assert extract_api_key_principal("") is None


def test_sk_key_is_hashed_with_legacy_helper():
    raw = "sk-abc123"
    principal = extract_api_key_principal(raw)
    assert principal is not None
    assert principal.token_hash == UserAPIKeyAuth._safe_hash_litellm_api_key(raw)
    assert principal.token_hash != raw


def test_bearer_prefix_normalized_then_hashed():
    raw = "Bearer sk-abc123"
    principal = extract_api_key_principal(raw)
    assert principal is not None
    assert principal.token_hash == UserAPIKeyAuth._safe_hash_litellm_api_key(raw)
    sk_only_hash = UserAPIKeyAuth._safe_hash_litellm_api_key("sk-abc123")
    assert principal.token_hash == sk_only_hash


def test_jwt_shaped_token_gets_hashed_jwt_prefix():
    fake_jwt = "aaaa.bbbb.cccc"
    principal = extract_api_key_principal(fake_jwt)
    assert principal is not None
    assert principal.token_hash.startswith("hashed-jwt-")


def test_non_sk_non_jwt_returned_unhashed():
    raw = "custom-key-without-prefix"
    principal = extract_api_key_principal(raw)
    assert principal is not None
    assert principal.token_hash == raw


def test_hash_helper_delegates_to_legacy_path():
    assert hash_principal_token("sk-x") == UserAPIKeyAuth._safe_hash_litellm_api_key(
        "sk-x"
    )

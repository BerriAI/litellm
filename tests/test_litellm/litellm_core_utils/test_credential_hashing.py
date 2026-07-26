import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.litellm_core_utils.credential_hashing import hash_credential, sanitize_key_hash


def test_raw_virtual_key_is_hashed():
    assert hash_credential("sk-1234") == hashlib.sha256(b"sk-1234").hexdigest()


def test_bearer_prefixed_virtual_key_is_hashed_without_prefix():
    assert hash_credential("Bearer sk-1234") == hashlib.sha256(b"sk-1234").hexdigest()


def test_jwt_is_hashed_and_tagged():
    token = "header.payload.signature"
    assert hash_credential(token) == f"hashed-jwt-{hashlib.sha256(token.encode()).hexdigest()}"


def test_existing_hash_passes_through():
    digest = hashlib.sha256(b"sk-1234").hexdigest()
    assert hash_credential(digest) == digest


def test_master_key_alias_passes_through():
    assert hash_credential(LITELLM_PROXY_MASTER_KEY_ALIAS) == LITELLM_PROXY_MASTER_KEY_ALIAS


def test_non_credential_identifier_passes_through():
    assert hash_credential("test_hash") == "test_hash"


def test_sanitize_key_hash_leaves_non_strings_alone():
    assert sanitize_key_hash(None) is None


def test_user_api_key_auth_shares_the_hashing_rule():
    from litellm.proxy._types import UserAPIKeyAuth

    raw_key = "sk-1234"

    assert UserAPIKeyAuth(api_key=raw_key).api_key == sanitize_key_hash(raw_key)

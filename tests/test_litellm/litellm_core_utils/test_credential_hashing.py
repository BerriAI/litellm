import base64
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.litellm_core_utils.credential_hashing import sanitize_key_hash


def _jwt(header: dict) -> str:
    def segment(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()

    return f"{segment(header)}.{segment({'sub': 'user-1'})}.signature"


def test_raw_virtual_key_is_hashed():
    assert sanitize_key_hash("sk-1234") == hashlib.sha256(b"sk-1234").hexdigest()


def test_bearer_prefixed_virtual_key_is_hashed_without_prefix():
    assert sanitize_key_hash("Bearer sk-1234") == hashlib.sha256(b"sk-1234").hexdigest()


def test_jwt_is_hashed():
    token = _jwt({"alg": "RS256", "typ": "JWT"})
    assert sanitize_key_hash(token) == hashlib.sha256(token.encode()).hexdigest()


def test_existing_hash_passes_through():
    digest = hashlib.sha256(b"sk-1234").hexdigest()
    assert sanitize_key_hash(digest) == digest


def test_master_key_alias_passes_through():
    assert sanitize_key_hash(LITELLM_PROXY_MASTER_KEY_ALIAS) == LITELLM_PROXY_MASTER_KEY_ALIAS


def test_non_credential_identifier_passes_through():
    assert sanitize_key_hash("test_hash") == "test_hash"


def test_dotted_identifier_is_not_treated_as_jwt():
    assert sanitize_key_hash("team.project.key") == "team.project.key"


def test_none_passes_through():
    assert sanitize_key_hash(None) is None

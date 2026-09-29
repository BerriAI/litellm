"""Live e2e: UI/CLI session tokens are accepted only while valid and only when minted as session tokens.

The runner mints its own session tokens under the proxy's salt key, so the valid and expired cases run in
seconds instead of waiting out a real login's expiry.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from e2e_config import MASTER_KEY, unique_marker
from e2e_http import UnauthorizedError, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody, KeyLoggingCallback, KeyLoggingCallbackVars, KeyMetadata
from other_client import OtherClient

pytestmark = pytest.mark.e2e

SALT_KEY: Final = os.environ.get("LITELLM_SALT_KEY") or MASTER_KEY
SESSION_TOKEN_AAD: Final = b"litellm-session-token"
AES_GCM_PREFIX: Final = "v2:gcm:"
ENCRYPTED_PREFIX: Final = "litellm_enc::"


def _admin_session_token(expires_at: datetime) -> str:
    claims: Final = json.dumps(
        {
            "token": f"ui-token-{unique_marker()}",
            "user_id": f"e2e-session-{unique_marker()}",
            "user_role": "proxy_admin",
            "team_id": "litellm-dashboard",
            "expires": expires_at.isoformat(),
        }
    )
    nonce: Final = os.urandom(12)
    sealed: Final = AESGCM(hashlib.sha256(SALT_KEY.encode()).digest()).encrypt(
        nonce, claims.encode(), SESSION_TOKEN_AAD
    )
    return AES_GCM_PREFIX + base64.urlsafe_b64encode(nonce + sealed).decode()


class TestSessionToken:
    @pytest.mark.covers("other.auth.session_token.valid_allows")
    def test_unexpired_session_token_reaches_admin_route(self, client: OtherClient) -> None:
        token: Final = _admin_session_token(datetime.now(timezone.utc) + timedelta(minutes=10))
        listing: Final = unwrap(client.list_users_as(token))
        assert listing.total >= 0, f"an unexpired admin session token did not reach /user/list: {listing}"

    @pytest.mark.covers("other.auth.session_token.expired_denied")
    def test_expired_session_token_is_denied(self, client: OtherClient) -> None:
        token: Final = _admin_session_token(datetime.now(timezone.utc) - timedelta(minutes=1))
        result: Final = client.list_users_as(token)
        assert isinstance(result, UnauthorizedError), f"an expired session token must get 401, got {result}"
        assert "expired" in result.body.lower(), f"expected the expired-key error, got {result.body[:300]}"

    @pytest.mark.covers("other.auth.session_token.encrypted_value_denied")
    def test_encrypted_stored_value_is_not_a_bearer_token(
        self, client: OtherClient, resources: ResourceManager
    ) -> None:
        stored_value: Final = f'{{"token": "{unique_marker()}", "user_role": "proxy_admin"}}'
        key: Final = client.proxy.generate_key(
            KeyGenerateBody(
                key_alias=f"e2e-session-{unique_marker()}",
                metadata=KeyMetadata(
                    logging=[
                        KeyLoggingCallback(
                            callback_name="langfuse",
                            callback_vars=KeyLoggingCallbackVars(langfuse_secret_key=stored_value),
                        )
                    ]
                ),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        metadata: Final = client.proxy.key_info(key).metadata
        assert metadata is not None and metadata.logging, f"/key/info dropped the logging metadata: {metadata}"
        encrypted: Final = metadata.logging[0].callback_vars.langfuse_secret_key
        assert encrypted is not None and encrypted.startswith(ENCRYPTED_PREFIX), (
            f"expected /key/info to return the stored secret encrypted, got {encrypted!r}"
        )

        for bearer in (encrypted.removeprefix(ENCRYPTED_PREFIX), encrypted):
            result = client.list_users_as(bearer)
            assert isinstance(result, UnauthorizedError), f"an encrypted stored value must get 401, got {result}"

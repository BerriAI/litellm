"""Live e2e: master key rotation re-encrypts team and key callback_vars.

Logging callback credentials (Langfuse / Langsmith secrets) live encrypted at
rest in LiteLLM_TeamTable.metadata and LiteLLM_VerificationToken.metadata. Master
key rotation used to re-encrypt models, config env vars, MCP credentials, and the
credentials table but skipped these callback_vars, so after a rotation they stayed
encrypted under the old key and produced recurring decryption errors.

This drives the real /key/regenerate rotation against a live proxy: it creates a
team and a key carrying encrypted callback_vars, rotates the master key to a fresh
value, and asserts the stored ciphertext for both rows was re-encrypted (changed)
while staying encrypted at rest and never leaking the plaintext secret. The
non-sensitive langfuse_host is left untouched. Reverting the fix leaves the
ciphertext byte-for-byte identical after rotation, so the change assertion fails.

The rotation re-encrypts every at-rest secret under the new key while the running
proxy keeps the old key in memory (a restart with the new key is the operator's
next step), so the teardown rotates the master key back to the suite's key to
leave the shared stack decryptable for other tests.
"""

from __future__ import annotations

import pytest

from e2e_config import MASTER_KEY, unique_marker
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    CallbackMetadata,
    CallbackVars,
    KeyGenerateBody,
    LoggingCallbackEntry,
    TeamNewBody,
)

pytestmark = pytest.mark.e2e

ENCRYPTED_PREFIX = "litellm_enc::"
LANGFUSE_HOST = "https://cloud.langfuse.com"


def _callback_metadata(marker: str, secret: str) -> CallbackMetadata:
    return CallbackMetadata(
        logging=[
            LoggingCallbackEntry(
                callback_name="langfuse",
                callback_type="success",
                callback_vars=CallbackVars(
                    langfuse_public_key=f"pk-lf-{marker}",
                    langfuse_secret_key=secret,
                    langfuse_host=LANGFUSE_HOST,
                ),
            )
        ]
    )


def _callback_vars(metadata: CallbackMetadata | None) -> CallbackVars:
    assert metadata is not None and metadata.logging, (
        f"expected callback logging metadata to be persisted, got {metadata!r}"
    )
    return metadata.logging[0].callback_vars


class TestMasterKeyRotationCallbackVars:
    @pytest.mark.covers("other.key_mgmt.master_rotation.reencrypts_callback_vars")
    def test_rotation_reencrypts_team_and_key_callback_vars(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        team_secret = f"sk-lf-team-secret-{marker}"
        key_secret = f"sk-lf-key-secret-{marker}"

        team_id = client.create_team(
            TeamNewBody(team_alias=f"e2e-rot-team-{marker}", metadata=_callback_metadata(marker, team_secret))
        )
        resources.defer(lambda: client.delete_team(team_id))
        key = client.gateway.generate_key(
            KeyGenerateBody(key_alias=f"e2e-rot-key-{marker}", metadata=_callback_metadata(marker, key_secret))
        )
        resources.defer(lambda: client.gateway.delete_key(key))

        team_before = _callback_vars(client.team_info(team_id).metadata)
        key_before = _callback_vars(client.gateway.key_info(key).metadata)

        for label, plaintext, ciphertext in (
            ("team", team_secret, team_before.langfuse_secret_key),
            ("key", key_secret, key_before.langfuse_secret_key),
        ):
            assert ciphertext is not None and ciphertext.startswith(ENCRYPTED_PREFIX), (
                f"{label} langfuse_secret_key must be encrypted at rest, got {ciphertext!r}"
            )
            assert plaintext not in ciphertext, f"{label} secret leaked as plaintext at rest: {ciphertext!r}"

        new_master_key = f"sk-rotated-{marker}"
        client.rotate_master_key(MASTER_KEY, new_master_key)
        resources.defer(lambda: client.rotate_master_key(MASTER_KEY, MASTER_KEY))

        team_after = _callback_vars(client.team_info(team_id).metadata)
        key_after = _callback_vars(client.gateway.key_info(key).metadata)

        for label, plaintext, before, after in (
            ("team", team_secret, team_before.langfuse_secret_key, team_after.langfuse_secret_key),
            ("key", key_secret, key_before.langfuse_secret_key, key_after.langfuse_secret_key),
        ):
            assert after is not None and after.startswith(ENCRYPTED_PREFIX), (
                f"{label} langfuse_secret_key must stay encrypted after rotation, got {after!r}"
            )
            assert after != before, (
                f"{label} langfuse_secret_key was not re-encrypted on master key rotation; "
                f"it is still the old-key ciphertext {after!r}"
            )
            assert plaintext not in after, f"{label} secret leaked as plaintext after rotation: {after!r}"

        assert team_after.langfuse_host == LANGFUSE_HOST, (
            f"non-sensitive langfuse_host must be untouched by rotation, got {team_after.langfuse_host!r}"
        )
        assert key_after.langfuse_host == LANGFUSE_HOST, (
            f"non-sensitive langfuse_host must be untouched by rotation, got {key_after.langfuse_host!r}"
        )

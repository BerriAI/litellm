from typing import Final
from unittest.mock import patch

import litellm
from litellm.litellm_core_utils.credential_accessor import (
    CredentialAccessor,
    expand_credential_env_refs,
)
from litellm.models.credentials import CredentialItem

CREDENTIAL_NAME: Final = "bedrock-account-b"


def _credential(values: dict[str, object]) -> CredentialItem:
    return CredentialItem(
        credential_name=CREDENTIAL_NAME,
        credential_values=values,
        credential_info={"custom_llm_provider": "bedrock"},
    )


def test_upsert_resolves_env_refs_so_deployments_read_effective_values(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID_CRED_TEST", "ak-account-b")
    with patch.object(litellm, "credential_list", []):  # test-quality-ok: isolate the global list
        CredentialAccessor.upsert_credentials(
            [
                _credential(
                    {
                        "aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID_CRED_TEST",
                        "aws_region_name": "us-east-1",
                    }
                )
            ]
        )

        assert CredentialAccessor.get_credential_values(CREDENTIAL_NAME) == {
            "aws_access_key_id": "ak-account-b",
            "aws_region_name": "us-east-1",
        }


def test_upsert_keeps_the_reference_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY_CRED_TEST", raising=False)
    with patch.object(litellm, "credential_list", []):  # test-quality-ok: isolate the global list
        CredentialAccessor.upsert_credentials(
            [_credential({"aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY_CRED_TEST"})]
        )

        assert CredentialAccessor.get_credential_values(CREDENTIAL_NAME) == {
            "aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY_CRED_TEST"
        }


def test_expand_leaves_non_reference_values_alone():
    values: Final = {
        "aws_region_name": "us-east-1",
        "aws_session_tags": ["team", "prod"],
        "max_retries": 4,
    }

    assert expand_credential_env_refs(CREDENTIAL_NAME, values) == values


def test_expand_leaves_federation_pointer_fields_as_written(monkeypatch):
    monkeypatch.setenv("JWKS_CRED_TEST_SIGNING_KEY", "pem-bytes")
    values: Final = {
        "anthropic_identity_source": "internal_issuer",
        "anthropic_issuer_signing_key_ref": "os.environ/JWKS_CRED_TEST_SIGNING_KEY",
    }

    assert expand_credential_env_refs(CREDENTIAL_NAME, values) == values

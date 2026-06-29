from unittest.mock import patch

from litellm.proxy.credential_endpoints.endpoints import (
    _preserve_masked_credential_values,
    update_db_credential,
)
from litellm.types.utils import CredentialItem


def test_preserve_masked_credential_value_keeps_existing_secret():
    db_credential = CredentialItem(
        credential_name="openai-prod",
        credential_values={
            "api_key": "encrypted-stored-api-key",
            "api_base": "https://old.example",
        },
        credential_info={},
    )
    updated_patch = CredentialItem(
        credential_name="openai-prod",
        credential_values={
            "api_key": "sk****34",
            "api_base": "https://new.example",
        },
        credential_info={},
    )

    with (
        patch(
            "litellm.proxy.credential_endpoints.endpoints.decrypt_value_helper",
            return_value="sk-live-secret-1234",
        ),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key=None: f"encrypted:{value}",
        ),
    ):
        sanitized_patch = _preserve_masked_credential_values(
            db_credential, updated_patch
        )
        merged_credential = update_db_credential(db_credential, sanitized_patch)

    assert sanitized_patch.credential_values == {"api_base": "https://new.example"}
    assert merged_credential.credential_values["api_key"] == "encrypted-stored-api-key"
    assert merged_credential.credential_values["api_base"] == (
        "encrypted:https://new.example"
    )


def test_preserve_masked_credential_value_allows_real_secret_rotation():
    db_credential = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "encrypted-stored-api-key"},
        credential_info={},
    )
    updated_patch = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "sk-new-secret-5678"},
        credential_info={},
    )

    with (
        patch(
            "litellm.proxy.credential_endpoints.endpoints.decrypt_value_helper",
            return_value="sk-live-secret-1234",
        ),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key=None: f"encrypted:{value}",
        ),
    ):
        sanitized_patch = _preserve_masked_credential_values(
            db_credential, updated_patch
        )
        merged_credential = update_db_credential(db_credential, sanitized_patch)

    assert sanitized_patch.credential_values == {"api_key": "sk-new-secret-5678"}
    assert merged_credential.credential_values["api_key"] == (
        "encrypted:sk-new-secret-5678"
    )


def test_preserve_masked_credential_value_on_decryption_failure():
    db_credential = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "encrypted-with-old-master-key"},
        credential_info={},
    )
    updated_patch = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": "sk****34"},
        credential_info={},
    )

    with (
        patch(
            "litellm.proxy.credential_endpoints.endpoints.decrypt_value_helper",
            return_value=None,
        ),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key=None: f"encrypted:{value}",
        ),
    ):
        sanitized_patch = _preserve_masked_credential_values(
            db_credential, updated_patch
        )
        merged_credential = update_db_credential(db_credential, sanitized_patch)

    assert sanitized_patch.credential_values == {}
    assert (
        merged_credential.credential_values["api_key"]
        == "encrypted-with-old-master-key"
    )


def test_preserve_short_masked_credential_value_on_decryption_failure():
    db_credential = CredentialItem(
        credential_name="short-secret",
        credential_values={"api_key": "encrypted-short-secret"},
        credential_info={},
    )
    updated_patch = CredentialItem(
        credential_name="short-secret",
        credential_values={"api_key": "*****", "api_base": "https://new.example"},
        credential_info={},
    )

    with (
        patch(
            "litellm.proxy.credential_endpoints.endpoints.decrypt_value_helper",
            return_value=None,
        ),
        patch(
            "litellm.proxy.credential_endpoints.endpoints.encrypt_value_helper",
            side_effect=lambda value, new_encryption_key=None: f"encrypted:{value}",
        ),
    ):
        sanitized_patch = _preserve_masked_credential_values(
            db_credential, updated_patch
        )
        merged_credential = update_db_credential(db_credential, sanitized_patch)

    assert sanitized_patch.credential_values == {"api_base": "https://new.example"}
    assert merged_credential.credential_values["api_key"] == "encrypted-short-secret"
    assert merged_credential.credential_values["api_base"] == (
        "encrypted:https://new.example"
    )

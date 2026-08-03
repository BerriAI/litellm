from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.credential_endpoints import endpoints as credential_endpoints
from litellm.proxy.credential_endpoints.endpoints import (
    CredentialHelperUtils,
    router,
    update_credential,
    update_db_credential,
)
from litellm.types.utils import CredentialItem


def _auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN)


def _app() -> TestClient:
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[user_api_key_auth] = _auth
    return TestClient(test_app)


def test_update_db_credential_merges_api_base_without_dropping_other_values() -> None:
    db_credential = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_key": "enc-key", "api_base": "https://api.openai.com/v1"},
        credential_info={"custom_llm_provider": "openai"},
    )
    update_patch = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_base": "https://proxy.e2e.example.com/v1"},
        credential_info={},
    )

    with patch.object(
        CredentialHelperUtils,
        "encrypt_credential_values",
        side_effect=lambda cred, new_encryption_key=None: cred,
    ):
        merged = update_db_credential(db_credential, update_patch)

    assert merged.credential_values["api_base"] == "https://proxy.e2e.example.com/v1"
    assert merged.credential_values["api_key"] == "enc-key"


@pytest.mark.asyncio
async def test_update_credential_always_upserts_in_memory_even_when_absent() -> None:
    db_credential = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_key": "enc-key", "api_base": "https://api.openai.com/v1"},
        credential_info={},
    )
    update_patch = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_base": "https://proxy.e2e.example.com/v1"},
        credential_info={},
    )
    repo = MagicMock()
    repo.find_by_name = AsyncMock(return_value=db_credential)
    repo.update_by_name = AsyncMock()
    prisma = MagicMock()
    previous = list(litellm.credential_list)
    litellm.credential_list = []

    try:
        with (
            patch.object(credential_endpoints, "CredentialsRepository", return_value=repo),
            patch("litellm.proxy.proxy_server.prisma_client", prisma),
            patch.object(
                CredentialHelperUtils,
                "encrypt_credential_values",
                side_effect=lambda cred, new_encryption_key=None: cred,
            ),
            patch.object(
                CredentialHelperUtils,
                "decrypt_credential_values",
                return_value=CredentialItem(
                    credential_name="e2e-cred",
                    credential_values={"api_key": "sk-plain", "api_base": "https://api.openai.com/v1"},
                    credential_info={},
                ),
            ),
            patch.object(credential_endpoints, "jsonify_object", side_effect=lambda x: x),
            patch.object(credential_endpoints.CredentialAccessor, "upsert_credentials") as upsert,
        ):
            result = await update_credential(
                request=MagicMock(),
                fastapi_response=MagicMock(),
                credential=update_patch,
                credential_name="e2e-cred",
                user_api_key_dict=_auth(),
            )

        assert result["success"] is True
        upsert.assert_called_once()
        written = upsert.call_args.args[0][0]
        assert written.credential_values["api_base"] == "https://proxy.e2e.example.com/v1"
        assert written.credential_values["api_key"] == "sk-plain"
    finally:
        litellm.credential_list = previous


def test_get_credential_by_name_prefers_db_over_stale_memory() -> None:
    client = _app()
    db_credential = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_key": "enc-key", "api_base": "https://proxy.e2e.example.com/v1"},
        credential_info={},
    )
    stale = CredentialItem(
        credential_name="e2e-cred",
        credential_values={"api_key": "sk-plain", "api_base": "https://api.openai.com/v1"},
        credential_info={},
    )
    previous = list(litellm.credential_list)
    litellm.credential_list = [stale]
    repo = MagicMock()
    repo.find_by_name = AsyncMock(return_value=db_credential)

    try:
        with (
            patch.object(credential_endpoints, "CredentialsRepository", return_value=repo),
            patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
            patch.object(
                CredentialHelperUtils,
                "decrypt_credential_values",
                return_value=CredentialItem(
                    credential_name="e2e-cred",
                    credential_values={
                        "api_key": "sk-plain",
                        "api_base": "https://proxy.e2e.example.com/v1",
                    },
                    credential_info={},
                ),
            ),
        ):
            response = client.get("/credentials/by_name/e2e-cred")
    finally:
        litellm.credential_list = previous

    assert response.status_code == 200
    body = response.json()
    assert body["credential_values"]["api_base"] == "https://proxy.e2e.example.com/v1"


@pytest.mark.asyncio
async def test_update_credential_raises_proxy_exception_instead_of_returning_200() -> None:
    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch.object(
            credential_endpoints,
            "CredentialsRepository",
            side_effect=HTTPException(status_code=403, detail="forbidden"),
        ),
        patch.object(
            credential_endpoints,
            "handle_exception_on_proxy",
            side_effect=lambda e: e,
        ),
    ):
        with pytest.raises(HTTPException):
            await update_credential(
                request=MagicMock(),
                fastapi_response=MagicMock(),
                credential=CredentialItem(
                    credential_name="e2e-cred",
                    credential_values={"api_base": "https://x"},
                    credential_info={},
                ),
                credential_name="e2e-cred",
                user_api_key_dict=_auth(),
            )

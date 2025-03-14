import json
import os
import sys
import pytest
from fastapi.testclient import TestClient
from litellm.proxy._types import LiteLLM_VerificationToken, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

class MockPrismaClient:
    def __init__(self):
        self.db = self
        self.litellm_verificationtoken = self

    async def find_unique(self, where):
        return LiteLLM_VerificationToken(
            token="sk-existing",
            user_id="user-123",
            team_id=None,
            key_name="test-key",
            key_alias="test-alias"
        )

    async def find_first(self, where):
        # Used by _enforce_unique_key_alias to check for duplicate key aliases
        return None

    async def update(self, where, data):
        self.last_update_data = data
        return LiteLLM_VerificationToken(
            token="sk-existing",
            user_id=data.get("user_id", "user-123"),
            team_id=None,
            key_name="test-key",
            key_alias=data.get("key_alias", "test-alias")
        )

    async def get_data(self, token, table_name, query_type="find_unique"):
        return await self.find_unique({"token": token})

    async def update_data(self, token, data):
        updated_token = await self.update({"token": token}, data)
        # Return in the format expected by the update_key_fn
        return {
            "data": {
                "token": updated_token.token,
                "user_id": updated_token.user_id,
                "team_id": updated_token.team_id,
                "key_name": updated_token.key_name,
                "key_alias": updated_token.key_alias
            }
        }

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_prisma():
    return MockPrismaClient()

@pytest.fixture(autouse=True)
def mock_user_auth(mocker):
    return mocker.patch(
        "litellm.proxy.auth.user_api_key_auth",
        return_value=UserAPIKeyAuth(
            api_key="sk-auth",
            user_id="user-123",
            team_id=None,
            user_role=LitellmUserRoles.PROXY_ADMIN.value  # Use the correct enum value
        )
    )

def test_user_id_not_reset_on_key_update(test_client, mock_prisma, mocker):
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = test_client.post(
        "/key/update",
        headers={"Authorization": "Bearer sk-auth"},
        json={
            "key": "sk-existing",
            "key_alias": "new-alias"
        }
    )

    assert response.status_code == 200
    assert mock_prisma.last_update_data["user_id"] == "user-123"

def test_user_id_explicit_none_prevented(test_client, mock_prisma, mocker):
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = test_client.post(
        "/key/update",
        headers={"Authorization": "Bearer sk-auth"},
        json={
            "key": "sk-existing",
            "key_alias": "new-alias",
            "user_id": None
        }
    )

    assert response.status_code == 200
    assert mock_prisma.last_update_data["user_id"] == "user-123"

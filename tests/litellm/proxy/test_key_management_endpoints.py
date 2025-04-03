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
            key_name="test-key"
        )

    async def find_first(self, where):
        return None

    async def get_data(self, token, table_name, query_type="find_unique"):
        return await self.find_unique({"token": token})

    async def update_data(self, token, data):
        self.last_update_data = data  # Store the update data for test verification
        return {"data": data}

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
            user_role=LitellmUserRoles.INTERNAL_USER.value
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

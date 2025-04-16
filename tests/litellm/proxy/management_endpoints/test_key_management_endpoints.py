import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock

from litellm.proxy.management_endpoints.key_management_endpoints import _list_key_helper
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_list_keys():
    mock_prisma_client = AsyncMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many
    args = {
        "prisma_client": mock_prisma_client,
        "page": 1,
        "size": 50,
        "user_id": "cda88cb4-cc2c-4e8c-b871-dc71ca111b00",
        "team_id": None,
        "organization_id": None,
        "key_alias": None,
        "exclude_team_id": None,
        "return_full_object": True,
        "admin_team_ids": ["28bd3181-02c5-48f2-b408-ce790fb3d5ba"],
    }
    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    mock_find_many.assert_called_once()

    where_condition = mock_find_many.call_args.kwargs["where"]
    print(f"where_condition: {where_condition}")
    assert json.dumps({"team_id": {"not": "litellm-dashboard"}}) in json.dumps(
        where_condition
    )

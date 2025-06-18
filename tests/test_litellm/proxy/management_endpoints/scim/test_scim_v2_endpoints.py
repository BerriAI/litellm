from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.scim.scim_errors import ScimUserAlreadyExists
from litellm.proxy.management_endpoints.scim.scim_v2 import create_user
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMUser,
    SCIMUserEmail,
    SCIMUserName,
)


@pytest.mark.asyncio
async def test_create_user_existing_user_conflict(mocker):
    """If a user already exists, create_user should raise ScimUserAlreadyExists"""

    scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        userName="existing-user",
        name=SCIMUserName(familyName="User", givenName="Existing"),
        emails=[SCIMUserEmail(value="existing@example.com")],
    )

    # Create a properly structured mock for the prisma client
    mock_prisma_client = mocker.MagicMock()
    mock_prisma_client.db = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable = mocker.MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value={"user_id": "existing-user"})

    # Mock the _get_prisma_client_or_raise_exception to return our mock
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._get_prisma_client_or_raise_exception",
        AsyncMock(return_value=mock_prisma_client),
    )
    
    mocked_new_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_user(user=scim_user)

    # Check that it's an HTTPException with status 409 
    assert exc_info.value.status_code == 409
    assert "existing-user" in str(exc_info.value.detail)
    mocked_new_user.assert_not_called()

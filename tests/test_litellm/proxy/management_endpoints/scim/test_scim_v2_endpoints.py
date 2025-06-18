from unittest.mock import AsyncMock

import pytest

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

    mock_prisma = mocker.MagicMock()

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2._check_user_exists",
        AsyncMock(return_value=True),
    )
    mocked_new_user = mocker.patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.new_user",
        AsyncMock(),
    )

    with pytest.raises(ScimUserAlreadyExists) as exc_info:
        await create_user(user=scim_user)

    assert exc_info.value.status_code == 409
    assert "existing-user" in exc_info.value.message
    mocked_new_user.assert_not_called()

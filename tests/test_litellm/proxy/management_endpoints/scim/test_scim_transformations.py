import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.proxy.management_endpoints.scim.scim_transformations import (
    ScimTransformations,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOperation,
)


# Mock data
@pytest.fixture
def mock_user():
    return LiteLLM_UserTable(
        user_id="user-123",
        user_email="test@example.com",
        user_alias="Test User",
        teams=["team-1", "team-2"],
        created_at=None,
        updated_at=None,
        metadata={},
    )


@pytest.fixture
def mock_user_with_scim_metadata():
    return LiteLLM_UserTable(
        user_id="user-456",
        user_email="test2@example.com",
        user_alias="Test User 2",
        teams=["team-1"],
        created_at=None,
        updated_at=None,
        metadata={"scim_metadata": {"givenName": "Test", "familyName": "User"}},
    )


@pytest.fixture
def mock_user_minimal():
    return LiteLLM_UserTable(
        user_id="user-789",
        user_email=None,
        user_alias=None,
        teams=[],
        created_at=None,
        updated_at=None,
        metadata={},
    )


@pytest.fixture
def mock_team():
    return LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Test Team",
        members_with_roles=[
            Member(user_id="user-123", user_email="test@example.com", role="admin"),
            Member(user_id="user-456", user_email="test2@example.com", role="user"),
        ],
        created_at=None,
        updated_at=None,
    )


@pytest.fixture
def mock_team_minimal():
    return LiteLLM_TeamTable(
        team_id="team-2",
        team_alias="Test Team 2",
        members_with_roles=[Member(user_id="user-789", user_email=None, role="user")],
        created_at=None,
        updated_at=None,
    )


@pytest.fixture
def mock_prisma_client():
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db

    mock_find_unique = AsyncMock()
    mock_db.litellm_teamtable.find_unique = mock_find_unique

    return mock_client, mock_find_unique


class TestScimTransformations:
    @pytest.mark.asyncio
    async def test_transform_litellm_user_to_scim_user(
        self, mock_user, mock_prisma_client
    ):
        mock_client, mock_find_unique = mock_prisma_client

        # Mock the team lookup
        team1 = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Team One", members_with_roles=[]
        )
        team2 = LiteLLM_TeamTable(
            team_id="team-2", team_alias="Team Two", members_with_roles=[]
        )

        mock_find_unique.side_effect = [team1, team2]

        with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
                mock_user
            )

            assert scim_user.id == mock_user.user_id
            assert scim_user.userName == mock_user.user_email
            assert scim_user.displayName == mock_user.user_email
            assert scim_user.name.familyName == mock_user.user_alias
            assert scim_user.name.givenName == mock_user.user_alias
            assert len(scim_user.emails) == 1
            assert scim_user.emails[0].value == mock_user.user_email
            assert len(scim_user.groups) == 2
            assert scim_user.groups[0].value == "team-1"
            assert scim_user.groups[0].display == "Team One"
            assert scim_user.groups[1].value == "team-2"
            assert scim_user.groups[1].display == "Team Two"

    @pytest.mark.asyncio
    async def test_transform_user_with_scim_metadata(
        self, mock_user_with_scim_metadata, mock_prisma_client
    ):
        mock_client, mock_find_unique = mock_prisma_client

        # Mock the team lookup
        team1 = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Team One", members_with_roles=[]
        )
        mock_find_unique.return_value = team1

        with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
                mock_user_with_scim_metadata
            )

            assert scim_user.name.givenName == "Test"
            assert scim_user.name.familyName == "User"

    @pytest.mark.asyncio
    async def test_transform_litellm_team_to_scim_group(
        self, mock_team, mock_prisma_client
    ):
        mock_client, _ = mock_prisma_client

        with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
            scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
                mock_team
            )

            assert scim_group.id == mock_team.team_id
            assert scim_group.displayName == mock_team.team_alias
            assert len(scim_group.members) == 2
            assert scim_group.members[0].value == "test@example.com"
            assert scim_group.members[0].display == "test@example.com"
            assert scim_group.members[1].value == "test2@example.com"
            assert scim_group.members[1].display == "test2@example.com"

    def test_get_scim_user_name(self, mock_user, mock_user_minimal):
        # User with email
        result = ScimTransformations._get_scim_user_name(mock_user)
        assert result == mock_user.user_email

        # User without email
        result = ScimTransformations._get_scim_user_name(mock_user_minimal)
        assert result == ScimTransformations.DEFAULT_SCIM_DISPLAY_NAME

    def test_get_scim_family_name(
        self, mock_user, mock_user_with_scim_metadata, mock_user_minimal
    ):
        # User with alias
        result = ScimTransformations._get_scim_family_name(mock_user)
        assert result == mock_user.user_alias

        # User with SCIM metadata
        result = ScimTransformations._get_scim_family_name(mock_user_with_scim_metadata)
        assert result == "User"

        # User without alias or metadata
        result = ScimTransformations._get_scim_family_name(mock_user_minimal)
        assert result == ScimTransformations.DEFAULT_SCIM_FAMILY_NAME

    def test_get_scim_given_name(
        self, mock_user, mock_user_with_scim_metadata, mock_user_minimal
    ):
        # User with alias
        result = ScimTransformations._get_scim_given_name(mock_user)
        assert result == mock_user.user_alias

        # User with SCIM metadata
        result = ScimTransformations._get_scim_given_name(mock_user_with_scim_metadata)
        assert result == "Test"

        # User without alias or metadata
        result = ScimTransformations._get_scim_given_name(mock_user_minimal)
        assert result == ScimTransformations.DEFAULT_SCIM_NAME

    def test_get_scim_member_value(self):
        # Member with email
        member_with_email = Member(
            user_id="user-123", user_email="test@example.com", role="admin"
        )
        result = ScimTransformations._get_scim_member_value(member_with_email)
        assert result == member_with_email.user_email

        # Member without email should fall back to user_id
        member_without_email = Member(user_id="user-456", user_email=None, role="user")
        result = ScimTransformations._get_scim_member_value(member_without_email)
        assert result == member_without_email.user_id

    @pytest.mark.asyncio
    async def test_transform_user_with_uuid_as_email(self, mock_prisma_client):
        """
        Test that users with UUID in user_email don't cause validation errors.
        This tests the defensive fix that validates email contains '@' before creating SCIMUserEmail.
        """
        mock_client, mock_find_unique = mock_prisma_client

        user_with_uuid_email = LiteLLM_UserTable(
            user_id="21df4e37-2f38-4f2e-a21b-c33cb939ff5b",
            user_email="21df4e37-2f38-4f2e-a21b-c33cb939ff5b",  # UUID as email (bug scenario)
            user_alias=None,
            teams=[],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        mock_find_unique.return_value = None

        with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
                user_with_uuid_email
            )

            assert scim_user.id == user_with_uuid_email.user_id
            assert scim_user.emails is None or len(scim_user.emails) == 0

    @pytest.mark.asyncio
    async def test_transform_user_with_none_email(self, mock_prisma_client):
        """
        Test that users with user_email=None are transformed correctly.
        This tests the root cause fix.
        """
        mock_client, mock_find_unique = mock_prisma_client

        user_with_none_email = LiteLLM_UserTable(
            user_id="user-from-group",
            user_email=None,
            user_alias=None,
            teams=[],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        mock_find_unique.return_value = None

        with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
                user_with_none_email
            )

            assert scim_user.id == user_with_none_email.user_id
            assert scim_user.emails is None or len(scim_user.emails) == 0



class TestSCIMPatchOperations:
    """Test SCIM PATCH operation validation and case-insensitive handling"""

    def test_scim_patch_operation_lowercase(self):
        """Test that lowercase operations are accepted"""
        op = SCIMPatchOperation(op="add", path="members", value=[{"value": "user123"}])
        assert op.op == "add"

        op = SCIMPatchOperation(op="remove", path='members[value eq "user123"]')
        assert op.op == "remove"

        op = SCIMPatchOperation(op="replace", path="displayName", value="New Name")
        assert op.op == "replace"

    def test_scim_patch_operation_uppercase(self):
        """Test that uppercase operations are normalized to lowercase"""
        op = SCIMPatchOperation(op="ADD", path="members", value=[{"value": "user123"}])
        assert op.op == "add"

        op = SCIMPatchOperation(op="REMOVE", path='members[value eq "user123"]')
        assert op.op == "remove"

        op = SCIMPatchOperation(op="REPLACE", path="displayName", value="New Name")
        assert op.op == "replace"

    def test_scim_patch_operation_mixed_case(self):
        """Test that mixed case operations are normalized to lowercase"""
        op = SCIMPatchOperation(op="Add", path="members", value=[{"value": "user123"}])
        assert op.op == "add"

        op = SCIMPatchOperation(op="Remove", path='members[value eq "user123"]')
        assert op.op == "remove"

        op = SCIMPatchOperation(op="Replace", path="displayName", value="New Name")
        assert op.op == "replace"

    def test_scim_patch_operation_with_optional_fields(self):
        """Test SCIMPatchOperation with and without optional fields"""
        # Operation with all fields
        op_full = SCIMPatchOperation(
            op="Add",
            path="members",
            value=[{"value": "user123", "display": "User 123"}],
        )
        assert op_full.op == "add"
        assert op_full.path == "members"
        assert op_full.value == [{"value": "user123", "display": "User 123"}]

        # Operation with minimal fields (only op is required)
        op_minimal = SCIMPatchOperation(op="Remove")
        assert op_minimal.op == "remove"
        assert op_minimal.path is None
        assert op_minimal.value is None

import asyncio
import json
import os
import sys
import uuid
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, Member
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMGroup,
    SCIMListResponse,
    SCIMMember,
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserEmail,
    SCIMUserGroup,
    SCIMUserName,
)


@pytest.fixture
def mock_prisma_client():
    """Mock prisma client for testing"""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    return mock_client


@pytest.fixture
def mock_user():
    """Mock user for testing"""
    return LiteLLM_UserTable(
        user_id="user-123",
        user_email="test@example.com",
        user_alias="Test User",
        teams=["team-1"],
        created_at=None,
        updated_at=None,
        metadata={},
    )


@pytest.fixture
def mock_team():
    """Mock team for testing"""
    return LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="Test Team",
        members=["user-123"],
        members_with_roles=[
            Member(user_id="user-123", user_email="test@example.com", role="user")
        ],
        created_at=None,
        updated_at=None,
    )


@pytest.fixture
def scim_user():
    """Sample SCIM user for testing"""
    return SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id="user-123",
        userName="test@example.com",
        displayName="Test User",
        name=SCIMUserName(givenName="Test", familyName="User"),
        emails=[SCIMUserEmail(value="test@example.com", primary=True)],
        groups=[SCIMUserGroup(value="team-1", display="Test Team")],
        active=True,
    )


@pytest.fixture
def scim_group():
    """Sample SCIM group for testing"""
    return SCIMGroup(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
        id="team-1",
        displayName="Test Team",
        members=[SCIMMember(value="user-123", display="test@example.com")],
    )


class TestSCIMUserEndpoints:
    """Test SCIM User endpoints"""

    @pytest.mark.asyncio
    async def test_create_user_basic(self, mock_prisma_client, scim_user):
        """Test basic user creation"""
        # Mock database responses
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = None
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Test Team", members=[], members_with_roles=[]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            with patch(
                "litellm.proxy.management_endpoints.scim.scim_v2.new_user"
            ) as mock_new_user:
                mock_new_user.return_value = LiteLLM_UserTable(
                    user_id="user-123",
                    user_email="test@example.com",
                    user_alias="Test",
                    teams=["team-1"],
                    created_at=None,
                    updated_at=None,
                    metadata={
                        "scim_metadata": {"givenName": "Test", "familyName": "User"}
                    },
                )

                from litellm.proxy.management_endpoints.scim.scim_v2 import create_user

                result = await create_user(scim_user)

                assert result.id == "user-123"
                assert result.userName == "test@example.com"
                assert result.name.givenName == "Test"
                assert result.name.familyName == "User"

    @pytest.mark.asyncio
    async def test_update_user_teams(self, mock_prisma_client, scim_user, mock_user):
        """Test updating user team memberships"""
        # Setup mocks
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = mock_user
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Test Team", members=["user-123"]
        )

        updated_user = LiteLLM_UserTable(**mock_user.model_dump())
        updated_user.teams = ["team-1"]
        mock_prisma_client.db.litellm_usertable.update.return_value = updated_user

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import update_user

            result = await update_user("user-123", scim_user)

            assert result.id == "user-123"
            # Verify update was called
            mock_prisma_client.db.litellm_usertable.update.assert_called()

    @pytest.mark.asyncio
    async def test_patch_user_add_group(self, mock_prisma_client, mock_user):
        """Test adding user to group via PATCH"""
        # Setup mocks
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = mock_user
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = LiteLLM_TeamTable(
            team_id="team-2", team_alias="Another Team", members=[]
        )

        updated_user = LiteLLM_UserTable(**mock_user.model_dump())
        updated_user.teams = ["team-1", "team-2"]
        mock_prisma_client.db.litellm_usertable.update.return_value = updated_user
        mock_prisma_client.db.litellm_teamtable.update.return_value = LiteLLM_TeamTable(
            team_id="team-2", team_alias="Another Team", members=["user-123"]
        )

        patch_ops = SCIMPatchOp(
            Operations=[
                SCIMPatchOperation(
                    op="add",
                    path="groups",
                    value=[{"value": "team-2", "display": "Another Team"}],
                )
            ]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import patch_user

            result = await patch_user("user-123", patch_ops)

            assert result.id == "user-123"
            # Verify team was added to user
            mock_prisma_client.db.litellm_usertable.update.assert_called()
            # Verify user was added to team
            mock_prisma_client.db.litellm_teamtable.update.assert_called()

    @pytest.mark.asyncio
    async def test_patch_user_remove_group(self, mock_prisma_client, mock_user):
        """Test removing user from group via PATCH"""
        # Setup mocks
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = mock_user
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Test Team", members=["user-123"]
        )

        updated_user = LiteLLM_UserTable(**mock_user.model_dump())
        updated_user.teams = []
        mock_prisma_client.db.litellm_usertable.update.return_value = updated_user
        mock_prisma_client.db.litellm_teamtable.update.return_value = LiteLLM_TeamTable(
            team_id="team-1", team_alias="Test Team", members=[]
        )

        patch_ops = SCIMPatchOp(
            Operations=[
                SCIMPatchOperation(
                    op="remove", path='groups[value eq "team-1"]', value=None
                )
            ]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import patch_user

            result = await patch_user("user-123", patch_ops)

            assert result.id == "user-123"
            # Verify team was removed from user
            mock_prisma_client.db.litellm_usertable.update.assert_called()
            # Verify user was removed from team
            mock_prisma_client.db.litellm_teamtable.update.assert_called()


class TestSCIMGroupEndpoints:
    """Test SCIM Group endpoints"""

    @pytest.mark.asyncio
    async def test_create_group_with_members(self, mock_prisma_client, scim_group):
        """Test creating group with members"""
        # Setup mocks
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = None
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            user_alias="Test User",
            teams=[],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        created_team = LiteLLM_TeamTable(
            team_id="team-1",
            team_alias="Test Team",
            members=["user-123"],
            members_with_roles=[
                Member(user_id="user-123", user_email="test@example.com", role="user")
            ],
            created_at=None,
            updated_at=None,
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            with patch(
                "litellm.proxy.management_endpoints.scim.scim_v2.new_team"
            ) as mock_new_team:
                mock_new_team.return_value = created_team

                from litellm.proxy.management_endpoints.scim.scim_v2 import create_group

                result = await create_group(scim_group)

                assert result.id == "team-1"
                assert result.displayName == "Test Team"
                assert result.members is not None
                assert len(result.members) == 1
                assert result.members[0].value == "test@example.com"

                # Verify new_team was called with correct members
                mock_new_team.assert_called_once()
                call_args = mock_new_team.call_args[1]["data"]
                assert len(call_args.members_with_roles) == 1
                assert call_args.members_with_roles[0].user_id == "user-123"

    @pytest.mark.asyncio
    async def test_create_group_member_lookup_by_email(
        self, mock_prisma_client, scim_group
    ):
        """Test creating group where member is looked up by email"""
        # Setup mocks - first call returns None (lookup by user_id), second call returns user (lookup by email)
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = None
        mock_prisma_client.db.litellm_usertable.find_unique.side_effect = [
            None,  # First call (lookup by user_id) returns None
            LiteLLM_UserTable(  # Second call (lookup by email) returns user
                user_id="user-456",
                user_email="user-123",  # This is actually an email in the member value
                user_alias="Test User",
                teams=[],
                created_at=None,
                updated_at=None,
                metadata={},
            ),
        ]

        created_team = LiteLLM_TeamTable(
            team_id="team-1",
            team_alias="Test Team",
            members=["user-456"],
            members_with_roles=[
                Member(user_id="user-456", user_email="user-123", role="user")
            ],
            created_at=None,
            updated_at=None,
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            with patch(
                "litellm.proxy.management_endpoints.scim.scim_v2.new_team"
            ) as mock_new_team:
                mock_new_team.return_value = created_team

                from litellm.proxy.management_endpoints.scim.scim_v2 import create_group

                result = await create_group(scim_group)

                assert result.id == "team-1"
                # Verify both lookup calls were made
                assert mock_prisma_client.db.litellm_usertable.find_unique.call_count == 2

    @pytest.mark.asyncio
    async def test_update_group_members(self, mock_prisma_client, scim_group, mock_team):
        """Test updating group members"""
        # Setup mocks
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = mock_team
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            user_alias="Test User",
            teams=["team-1"],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        updated_team = LiteLLM_TeamTable(**mock_team.model_dump())
        updated_team.members = ["user-123"]
        mock_prisma_client.db.litellm_teamtable.update.return_value = updated_team
        mock_prisma_client.db.litellm_usertable.update.return_value = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            user_alias="Test User",
            teams=["team-1"],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import update_group

            result = await update_group("team-1", scim_group)

            assert result.id == "team-1"
            assert result.displayName == "Test Team"
            # Verify update was called
            mock_prisma_client.db.litellm_teamtable.update.assert_called()

    @pytest.mark.asyncio
    async def test_patch_group_add_member(self, mock_prisma_client, mock_team):
        """Test adding member to group via PATCH"""
        # Setup mocks
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = mock_team
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
            user_id="user-456",
            user_email="new@example.com",
            user_alias="New User",
            teams=[],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        updated_team = LiteLLM_TeamTable(**mock_team.model_dump())
        updated_team.members = ["user-123", "user-456"]
        mock_prisma_client.db.litellm_teamtable.update.return_value = updated_team
        mock_prisma_client.db.litellm_usertable.update.return_value = LiteLLM_UserTable(
            user_id="user-456",
            user_email="new@example.com",
            user_alias="New User",
            teams=["team-1"],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        patch_ops = SCIMPatchOp(
            Operations=[
                SCIMPatchOperation(
                    op="add",
                    path="members",
                    value=[{"value": "user-456", "display": "new@example.com"}],
                )
            ]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import patch_group

            result = await patch_group("team-1", patch_ops)

            assert result.id == "team-1"
            # Verify team was updated
            mock_prisma_client.db.litellm_teamtable.update.assert_called()
            # Verify user was updated
            mock_prisma_client.db.litellm_usertable.update.assert_called()

    @pytest.mark.asyncio
    async def test_patch_group_remove_member(self, mock_prisma_client, mock_team):
        """Test removing member from group via PATCH"""
        # Setup mocks
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = mock_team
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            user_alias="Test User",
            teams=["team-1"],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        updated_team = LiteLLM_TeamTable(**mock_team.model_dump())
        updated_team.members = []
        mock_prisma_client.db.litellm_teamtable.update.return_value = updated_team
        mock_prisma_client.db.litellm_usertable.update.return_value = LiteLLM_UserTable(
            user_id="user-123",
            user_email="test@example.com",
            user_alias="Test User",
            teams=[],
            created_at=None,
            updated_at=None,
            metadata={},
        )

        patch_ops = SCIMPatchOp(
            Operations=[
                SCIMPatchOperation(
                    op="remove", path='members[value eq "user-123"]', value=None
                )
            ]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import patch_group

            result = await patch_group("team-1", patch_ops)

            assert result.id == "team-1"
            # Verify team was updated
            mock_prisma_client.db.litellm_teamtable.update.assert_called()
            # Verify user was updated
            mock_prisma_client.db.litellm_usertable.update.assert_called()

    @pytest.mark.asyncio
    async def test_patch_group_replace_members(self, mock_prisma_client, mock_team):
        """Test replacing all members in a group via PATCH"""
        # Setup mocks for existing user and new user
        mock_prisma_client.db.litellm_teamtable.find_unique.return_value = mock_team

        def user_lookup_side_effect(*args, **kwargs):
            where_clause = kwargs.get("where", {})
            if "user_id" in where_clause:
                if where_clause["user_id"] == "user-456":
                    return LiteLLM_UserTable(
                        user_id="user-456",
                        user_email="new@example.com",
                        user_alias="New User",
                        teams=[],
                        created_at=None,
                        updated_at=None,
                        metadata={},
                    )
                elif where_clause["user_id"] == "user-123":
                    return LiteLLM_UserTable(
                        user_id="user-123",
                        user_email="test@example.com",
                        user_alias="Test User",
                        teams=["team-1"],
                        created_at=None,
                        updated_at=None,
                        metadata={},
                    )
            return None

        mock_prisma_client.db.litellm_usertable.find_unique.side_effect = (
            user_lookup_side_effect
        )

        updated_team = LiteLLM_TeamTable(**mock_team.model_dump())
        updated_team.members = ["user-456"]
        mock_prisma_client.db.litellm_teamtable.update.return_value = updated_team

        patch_ops = SCIMPatchOp(
            Operations=[
                SCIMPatchOperation(
                    op="replace",
                    path="members",
                    value=[{"value": "user-456", "display": "new@example.com"}],
                )
            ]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import patch_group

            result = await patch_group("team-1", patch_ops)

            assert result.id == "team-1"
            # Verify team was updated
            mock_prisma_client.db.litellm_teamtable.update.assert_called()
            # Verify user updates were called (both removal and addition)
            assert mock_prisma_client.db.litellm_usertable.update.call_count >= 1


class TestSCIMErrorHandling:
    """Test SCIM error handling"""

    @pytest.mark.asyncio
    async def test_create_user_duplicate_error(self, mock_prisma_client, scim_user):
        """Test creating user with duplicate username"""
        # Mock existing user
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = (
            LiteLLM_UserTable(
                user_id="user-123",
                user_email="test@example.com",
                user_alias="Test User",
                teams=[],
                created_at=None,
                updated_at=None,
                metadata={},
            )
        )

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import create_user

            with pytest.raises(HTTPException) as exc_info:
                await create_user(scim_user)

            assert exc_info.value.status_code == 409
            assert "already exists" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, mock_prisma_client):
        """Test getting non-existent user"""
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = None

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import get_user

            with pytest.raises(HTTPException) as exc_info:
                await get_user("non-existent-user")

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, mock_prisma_client, scim_user):
        """Test updating non-existent user"""
        mock_prisma_client.db.litellm_usertable.find_unique.return_value = None

        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
            from litellm.proxy.management_endpoints.scim.scim_v2 import update_user

            with pytest.raises(HTTPException) as exc_info:
                await update_user("non-existent-user", scim_user)

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
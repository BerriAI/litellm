import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, Mock, patch

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.proxy_server import app

client = TestClient(app)

SAMPLE_SKILL_MD = """---
name: Frontend Design
description: Create production-grade frontend interfaces
version: "1.0"
---

# Frontend Design

Best practices for building modern frontend applications.
"""

SAMPLE_SKILL_MD_UPDATED = """---
name: Frontend Design
description: Updated frontend design guidelines
version: "2.0"
---

# Frontend Design v2

Updated best practices for building modern frontend applications.
"""


def _make_object_permission(skills=None):
    """Helper to create a mock LiteLLM_ObjectPermissionTable with skills."""
    return LiteLLM_ObjectPermissionTable(
        object_permission_id="test-perm-id",
        skills=skills or [],
    )


@pytest.mark.asyncio
async def test_create_skill():
    """
    Test creation of a new skill with valid SKILL.md content.
    Verifies frontmatter is parsed and stored correctly.
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Skill doesn't exist yet
            mock_db.litellm_skilltable.find_unique = AsyncMock(return_value=None)

            # Mock create
            created_skill = Mock()
            created_skill.skill_name = "frontend-design"
            created_skill.description = "Create production-grade frontend interfaces"
            created_skill.content = SAMPLE_SKILL_MD
            created_skill.metadata = json.dumps(
                {
                    "name": "Frontend Design",
                    "description": "Create production-grade frontend interfaces",
                    "version": "1.0",
                }
            )
            created_skill.created_at = datetime.now()
            created_skill.updated_at = datetime.now()
            created_skill.created_by = "test-user-123"
            mock_db.litellm_skilltable.create = AsyncMock(return_value=created_skill)

            skill_data = {
                "name": "frontend-design",
                "content": SAMPLE_SKILL_MD,
            }
            headers = {"Authorization": "Bearer sk-1234"}

            response = client.post("/skill/new", json=skill_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Skill 'frontend-design' created successfully"
            assert result["skill"]["name"] == "frontend-design"
            assert (
                result["skill"]["description"]
                == "Create production-grade frontend interfaces"
            )
            assert result["skill"]["metadata"]["version"] == "1.0"

            # Verify create was called with parsed frontmatter
            create_call = mock_db.litellm_skilltable.create.call_args
            assert create_call[1]["data"]["skill_name"] == "frontend-design"
            assert (
                create_call[1]["data"]["description"]
                == "Create production-grade frontend interfaces"
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_skill_duplicate():
    """
    Test that creating a skill with an existing name returns 400.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Skill already exists
            existing_skill = Mock()
            existing_skill.skill_name = "frontend-design"
            mock_db.litellm_skilltable.find_unique = AsyncMock(
                return_value=existing_skill
            )

            skill_data = {
                "name": "frontend-design",
                "content": SAMPLE_SKILL_MD,
            }
            headers = {"Authorization": "Bearer sk-1234"}

            response = client.post("/skill/new", json=skill_data, headers=headers)
            assert response.status_code == 400
            assert "already exists" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_skills_all():
    """
    Test listing all skills when object_permission has no skill restrictions.
    SkillPermissionHandler returns [] (no restrictions) -> all skills returned.
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        object_permission=None,  # no object_permission = unrestricted
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            skill_a = Mock()
            skill_a.skill_name = "skill-a"
            skill_a.description = "Skill A"
            skill_a.metadata = json.dumps({"name": "Skill A"})
            skill_a.created_at = datetime.now()
            skill_a.updated_at = datetime.now()
            skill_a.created_by = "admin"

            skill_b = Mock()
            skill_b.skill_name = "skill-b"
            skill_b.description = "Skill B"
            skill_b.metadata = json.dumps({"name": "Skill B"})
            skill_b.created_at = datetime.now()
            skill_b.updated_at = datetime.now()
            skill_b.created_by = "admin"

            mock_db.litellm_skilltable.find_many = AsyncMock(
                return_value=[skill_a, skill_b]
            )

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/skill/list", headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 2
            names = [s["name"] for s in result]
            assert "skill-a" in names
            assert "skill-b" in names

            # Verify content is NOT included in list response
            for skill in result:
                assert "content" not in skill
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_skills_scoped():
    """
    Test listing skills with object_permission.skills restriction.
    Only allowed skills should be returned.
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-456",
        object_permission=_make_object_permission(skills=["skill-a"]),
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            skill_a = Mock()
            skill_a.skill_name = "skill-a"
            skill_a.description = "Skill A"
            skill_a.metadata = json.dumps({"name": "Skill A"})
            skill_a.created_at = datetime.now()
            skill_a.updated_at = datetime.now()
            skill_a.created_by = "admin"

            mock_db.litellm_skilltable.find_many = AsyncMock(return_value=[skill_a])

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/skill/list", headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            assert result[0]["name"] == "skill-a"

            # Verify prisma was queried with the filter
            find_many_call = mock_db.litellm_skilltable.find_many.call_args
            assert find_many_call[1]["where"]["skill_name"]["in"] == ["skill-a"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_skills_empty_means_all():
    """
    Test listing skills when object_permission.skills is empty list
    (all skills allowed, matching how agents/MCP works in LiteLLM).
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-789",
        object_permission=_make_object_permission(skills=[]),  # empty = all allowed
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            skill_a = Mock()
            skill_a.skill_name = "skill-a"
            skill_a.description = "Skill A"
            skill_a.metadata = "{}"
            skill_a.created_at = datetime.now()
            skill_a.updated_at = datetime.now()
            skill_a.created_by = "admin"

            mock_db.litellm_skilltable.find_many = AsyncMock(return_value=[skill_a])

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/skill/list", headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1

            # Verify prisma was queried without filter (all skills)
            mock_db.litellm_skilltable.find_many.assert_called_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_skill_content():
    """
    Test getting full skill content by name (unrestricted key).
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        object_permission=None,  # unrestricted
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            skill_record = Mock()
            skill_record.skill_name = "frontend-design"
            skill_record.description = "Create production-grade frontend interfaces"
            skill_record.content = SAMPLE_SKILL_MD
            skill_record.metadata = json.dumps(
                {
                    "name": "Frontend Design",
                    "description": "Create production-grade frontend interfaces",
                }
            )
            skill_record.created_at = datetime.now()
            skill_record.updated_at = datetime.now()
            skill_record.created_by = "admin"

            mock_db.litellm_skilltable.find_unique = AsyncMock(
                return_value=skill_record
            )

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/skill/frontend-design", headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["name"] == "frontend-design"
            assert result["content"] == SAMPLE_SKILL_MD
            assert (
                result["description"] == "Create production-grade frontend interfaces"
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_skill_forbidden():
    """
    Test that accessing a skill not in object_permission.skills returns 403.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-456",
        object_permission=_make_object_permission(
            skills=["skill-a"]
        ),  # only skill-a allowed
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/skill/skill-b", headers=headers)
            assert response.status_code == 403
            assert "not allowed" in response.json()["detail"]

            # Verify prisma was NOT queried (rejected before DB call)
            mock_db.litellm_skilltable.find_unique.assert_not_called()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_update_skill():
    """
    Test updating an existing skill's content.
    """
    from datetime import datetime

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Skill exists
            existing_skill = Mock()
            existing_skill.skill_name = "frontend-design"
            mock_db.litellm_skilltable.find_unique = AsyncMock(
                return_value=existing_skill
            )

            # Mock update
            updated_skill = Mock()
            updated_skill.skill_name = "frontend-design"
            updated_skill.description = "Updated frontend design guidelines"
            updated_skill.content = SAMPLE_SKILL_MD_UPDATED
            updated_skill.metadata = json.dumps(
                {
                    "name": "Frontend Design",
                    "description": "Updated frontend design guidelines",
                    "version": "2.0",
                }
            )
            updated_skill.created_at = datetime.now()
            updated_skill.updated_at = datetime.now()
            updated_skill.created_by = "admin"
            mock_db.litellm_skilltable.update = AsyncMock(return_value=updated_skill)

            update_data = {
                "name": "frontend-design",
                "content": SAMPLE_SKILL_MD_UPDATED,
            }
            headers = {"Authorization": "Bearer sk-1234"}

            response = client.post("/skill/update", json=update_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Skill 'frontend-design' updated successfully"
            assert (
                result["skill"]["description"] == "Updated frontend design guidelines"
            )
            assert result["skill"]["metadata"]["version"] == "2.0"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_skill():
    """
    Test deleting a skill.
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Skill exists
            existing_skill = Mock()
            existing_skill.skill_name = "frontend-design"
            mock_db.litellm_skilltable.find_unique = AsyncMock(
                return_value=existing_skill
            )
            mock_db.litellm_skilltable.delete = AsyncMock(return_value=existing_skill)

            delete_data = {"name": "frontend-design"}
            headers = {"Authorization": "Bearer sk-1234"}

            response = client.post("/skill/delete", json=delete_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Skill 'frontend-design' deleted successfully"

            mock_db.litellm_skilltable.delete.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_parse_frontmatter():
    """
    Unit test for the _parse_frontmatter helper function.
    """
    from litellm.proxy.management_endpoints.skill_management_endpoints import (
        _parse_frontmatter,
    )

    # Valid frontmatter
    result = _parse_frontmatter(SAMPLE_SKILL_MD)
    assert result["name"] == "Frontend Design"
    assert result["description"] == "Create production-grade frontend interfaces"
    assert result["version"] == "1.0"

    # No frontmatter
    result = _parse_frontmatter("# Just a markdown file\n\nNo frontmatter here.")
    assert result == {}

    # Empty frontmatter
    result = _parse_frontmatter("---\n---\n# Content")
    assert result == {}

    # Frontmatter with only name
    result = _parse_frontmatter("---\nname: My Skill\n---\n# Body")
    assert result["name"] == "My Skill"
    assert result.get("description") is None


def test_skill_permission_handler():
    """
    Unit test for SkillPermissionHandler - verify it reads from object_permission.skills.
    """
    from litellm.proxy.management_endpoints.skill_permission_handler import (
        SkillPermissionHandler,
    )

    # No object_permission -> empty list (unrestricted)
    auth_none = UserAPIKeyAuth(user_id="test", object_permission=None)
    result = SkillPermissionHandler._get_allowed_skills_for_key(auth_none)
    assert result == []

    # object_permission with empty skills -> empty list (unrestricted)
    auth_empty = UserAPIKeyAuth(
        user_id="test",
        object_permission=_make_object_permission(skills=[]),
    )
    result = SkillPermissionHandler._get_allowed_skills_for_key(auth_empty)
    assert result == []

    # object_permission with specific skills -> those skills
    auth_scoped = UserAPIKeyAuth(
        user_id="test",
        object_permission=_make_object_permission(skills=["skill-a", "skill-b"]),
    )
    result = SkillPermissionHandler._get_allowed_skills_for_key(auth_scoped)
    assert sorted(result) == ["skill-a", "skill-b"]

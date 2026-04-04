"""
Tests for LiteLLMSkillsHandler - database CRUD operations.

Tests skill creation, listing, retrieval, and deletion through mocked Prisma client.
"""

import base64
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_SkillsTable, NewSkillRequest


def _make_prisma_skill(
    skill_id: str = "litellm_skill_test123",
    display_title: str = "Test Skill",
    description: str = "A test skill",
    instructions: str = "Do testing.",
    source: str = "custom",
    file_content: bytes = b"fake-zip",
    file_name: str = "skill.zip",
    file_type: str = "application/zip",
):
    """Create a mock Prisma skill record."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "skill_id": skill_id,
        "display_title": display_title,
        "description": description,
        "instructions": instructions,
        "source": source,
        "latest_version": None,
        "file_content": base64.b64encode(file_content).decode("utf-8"),
        "file_name": file_name,
        "file_type": file_type,
        "metadata": None,
        "created_at": datetime(2026, 3, 21),
        "created_by": "user1",
        "updated_at": datetime(2026, 3, 21),
        "updated_by": "user1",
    }
    return mock


class TestCreateSkill:
    """Tests for LiteLLMSkillsHandler.create_skill."""

    @pytest.mark.asyncio
    async def test_create_skill_success(self):
        """Test successful skill creation stores data in DB."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_first = AsyncMock(return_value=None)
        mock_prisma.db.litellm_skillstable.create = AsyncMock(
            return_value=_make_prisma_skill()
        )

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            request = NewSkillRequest(
                display_title="Test Skill",
                description="A test skill",
                instructions="Do testing.",
                file_content=b"fake-zip",
                file_name="skill.zip",
                file_type="application/zip",
            )

            result = await LiteLLMSkillsHandler.create_skill(
                data=request, user_id="user1"
            )

            assert isinstance(result, LiteLLM_SkillsTable)
            assert result.display_title == "Test Skill"
            assert result.description == "A test skill"
            mock_prisma.db.litellm_skillstable.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_skill_generates_litellm_prefix_id(self):
        """Test that created skill IDs have litellm_skill_ prefix."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_first = AsyncMock(return_value=None)
        mock_prisma.db.litellm_skillstable.create = AsyncMock(
            return_value=_make_prisma_skill()
        )

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            request = NewSkillRequest(
                display_title="Test",
                instructions="test",
            )

            await LiteLLMSkillsHandler.create_skill(data=request)

            call_args = mock_prisma.db.litellm_skillstable.create.call_args
            skill_data = call_args[1]["data"]
            assert skill_data["skill_id"].startswith("litellm_skill_")

    @pytest.mark.asyncio
    async def test_create_skill_no_prisma_raises(self):
        """Test that creating a skill without Prisma client raises ValueError."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        with patch(
            "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler._get_prisma_client",
            new_callable=AsyncMock,
            side_effect=ValueError("Prisma client is not initialized"),
        ):
            request = NewSkillRequest(display_title="Test", instructions="test")

            with pytest.raises(ValueError, match="Prisma client"):
                await LiteLLMSkillsHandler.create_skill(data=request)

    @pytest.mark.asyncio
    async def test_create_skill_duplicate_title_raises(self):
        """Test that creating a skill with a duplicate display_title raises ValueError."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        existing_skill = MagicMock()
        existing_skill.skill_id = "litellm_skill_existing"

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_first = AsyncMock(
            return_value=existing_skill
        )

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            request = NewSkillRequest(
                display_title="Duplicate Name",
                instructions="test",
            )

            with pytest.raises(ValueError, match="already exists"):
                await LiteLLMSkillsHandler.create_skill(data=request)

            # Should never reach create
            mock_prisma.db.litellm_skillstable.create.assert_not_called()


class TestListSkills:
    """Tests for LiteLLMSkillsHandler.list_skills."""

    @pytest.mark.asyncio
    async def test_list_skills_returns_records(self):
        """Test listing skills returns all records."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_many = AsyncMock(
            return_value=[
                _make_prisma_skill(skill_id="litellm_skill_1", display_title="Skill 1"),
                _make_prisma_skill(skill_id="litellm_skill_2", display_title="Skill 2"),
            ]
        )

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            results = await LiteLLMSkillsHandler.list_skills(limit=10)

            assert len(results) == 2
            assert all(isinstance(r, LiteLLM_SkillsTable) for r in results)

    @pytest.mark.asyncio
    async def test_list_skills_empty(self):
        """Test listing skills returns empty list when none exist."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_many = AsyncMock(return_value=[])

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            results = await LiteLLMSkillsHandler.list_skills()
            assert results == []

    @pytest.mark.asyncio
    async def test_list_skills_with_pagination(self):
        """Test listing skills passes limit and offset to Prisma."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_many = AsyncMock(return_value=[])

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            await LiteLLMSkillsHandler.list_skills(limit=5, offset=10)

            call_args = mock_prisma.db.litellm_skillstable.find_many.call_args
            assert call_args[1]["take"] == 5
            assert call_args[1]["skip"] == 10


class TestGetSkill:
    """Tests for LiteLLMSkillsHandler.get_skill."""

    @pytest.mark.asyncio
    async def test_get_skill_found(self):
        """Test getting a skill that exists."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(
            return_value=_make_prisma_skill(skill_id="litellm_skill_abc")
        )

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            result = await LiteLLMSkillsHandler.get_skill("litellm_skill_abc")

            assert isinstance(result, LiteLLM_SkillsTable)
            assert result.skill_id == "litellm_skill_abc"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self):
        """Test getting a skill that doesn't exist raises ValueError."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(return_value=None)

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            with pytest.raises(ValueError, match="Skill not found"):
                await LiteLLMSkillsHandler.get_skill("nonexistent")


class TestDeleteSkill:
    """Tests for LiteLLMSkillsHandler.delete_skill."""

    @pytest.mark.asyncio
    async def test_delete_skill_success(self):
        """Test deleting an existing skill."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(
            return_value=_make_prisma_skill(skill_id="litellm_skill_del")
        )
        mock_prisma.db.litellm_skillstable.delete = AsyncMock()

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            result = await LiteLLMSkillsHandler.delete_skill("litellm_skill_del")

            assert result["id"] == "litellm_skill_del"
            assert result["type"] == "skill_deleted"
            mock_prisma.db.litellm_skillstable.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_skill_not_found(self):
        """Test deleting a nonexistent skill raises ValueError."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(return_value=None)

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            with pytest.raises(ValueError, match="Skill not found"):
                await LiteLLMSkillsHandler.delete_skill("nonexistent")


class TestFetchSkillFromDb:
    """Tests for LiteLLMSkillsHandler.fetch_skill_from_db (convenience method)."""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_not_found(self):
        """Test that fetch_skill_from_db returns None instead of raising."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(return_value=None)

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            result = await LiteLLMSkillsHandler.fetch_skill_from_db("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_error(self):
        """Test that fetch_skill_from_db returns None on unexpected errors."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB connection lost"),
        ):
            result = await LiteLLMSkillsHandler.fetch_skill_from_db("any_id")
            assert result is None


class TestProviderSkillIds:
    """Tests for provider skill ID save/retrieve."""

    @pytest.mark.asyncio
    async def test_save_provider_skill_id(self):
        """Test saving an Anthropic skill ID for a LiteLLM skill."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_skill = MagicMock()
        mock_skill.metadata = {}

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(
            return_value=mock_skill
        )
        mock_prisma.db.litellm_skillstable.update = AsyncMock()

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            await LiteLLMSkillsHandler.save_provider_skill_id(
                skill_id="litellm_skill_abc",
                provider="anthropic",
                provider_skill_id="sk_ant_123",
            )

            call_args = mock_prisma.db.litellm_skillstable.update.call_args
            assert call_args[1]["where"] == {"skill_id": "litellm_skill_abc"}
            import json

            saved_metadata = json.loads(call_args[1]["data"]["metadata"])
            assert saved_metadata["_provider_skill_ids"]["anthropic"] == "sk_ant_123"

    @pytest.mark.asyncio
    async def test_save_provider_skill_id_preserves_existing_metadata(self):
        """Test that saving a provider ID doesn't clobber existing metadata."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_skill = MagicMock()
        mock_skill.metadata = {"custom_key": "custom_value"}

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(
            return_value=mock_skill
        )
        mock_prisma.db.litellm_skillstable.update = AsyncMock()

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            await LiteLLMSkillsHandler.save_provider_skill_id(
                skill_id="litellm_skill_abc",
                provider="anthropic",
                provider_skill_id="sk_ant_456",
            )

            call_args = mock_prisma.db.litellm_skillstable.update.call_args
            import json

            saved_metadata = json.loads(call_args[1]["data"]["metadata"])
            assert saved_metadata["custom_key"] == "custom_value"
            assert saved_metadata["_provider_skill_ids"]["anthropic"] == "sk_ant_456"

    @pytest.mark.asyncio
    async def test_save_provider_skill_id_skill_not_found(self):
        """Test that saving provider ID for missing skill doesn't raise."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_skillstable.find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_skillstable.update = AsyncMock()

        with patch.object(
            LiteLLMSkillsHandler,
            "_get_prisma_client",
            new_callable=AsyncMock,
            return_value=mock_prisma,
        ):
            # Should not raise
            await LiteLLMSkillsHandler.save_provider_skill_id(
                skill_id="nonexistent",
                provider="anthropic",
                provider_skill_id="sk_ant_789",
            )

            mock_prisma.db.litellm_skillstable.update.assert_not_called()

    def test_get_provider_skill_id_found(self):
        """Test retrieving a saved Anthropic skill ID."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        skill = LiteLLM_SkillsTable(
            skill_id="litellm_skill_abc",
            metadata={"_provider_skill_ids": {"anthropic": "sk_ant_123"}},
        )

        result = LiteLLMSkillsHandler.get_provider_skill_id(skill, "anthropic")
        assert result == "sk_ant_123"

    def test_get_provider_skill_id_not_found(self):
        """Test retrieving provider ID when none saved."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        skill = LiteLLM_SkillsTable(
            skill_id="litellm_skill_abc",
            metadata={},
        )

        result = LiteLLMSkillsHandler.get_provider_skill_id(skill, "anthropic")
        assert result is None

    def test_get_provider_skill_id_no_metadata(self):
        """Test retrieving provider ID when metadata is None."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        skill = LiteLLM_SkillsTable(
            skill_id="litellm_skill_abc",
            metadata=None,
        )

        result = LiteLLMSkillsHandler.get_provider_skill_id(skill, "anthropic")
        assert result is None

    def test_get_provider_skill_id_different_provider(self):
        """Test that provider IDs are namespaced per provider."""
        from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

        skill = LiteLLM_SkillsTable(
            skill_id="litellm_skill_abc",
            metadata={"_provider_skill_ids": {"anthropic": "sk_ant_123"}},
        )

        assert LiteLLMSkillsHandler.get_provider_skill_id(skill, "anthropic") == "sk_ant_123"
        assert LiteLLMSkillsHandler.get_provider_skill_id(skill, "openai") is None

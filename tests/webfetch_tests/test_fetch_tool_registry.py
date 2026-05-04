"""Unit tests for fetch_tool_registry.

These tests use pure mocks without real DB access.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from litellm.proxy.fetch_endpoints.fetch_tool_registry import FetchToolRegistry


class TestConvertPrismaToDict:
    """Test _convert_prisma_to_dict static method."""

    def test_empty_object(self):
        """Test with empty object."""
        obj = MagicMock()
        obj.__iter__ = MagicMock(return_value=iter([]))
        result = FetchToolRegistry._convert_prisma_to_dict(obj)
        assert result == {}

    def test_with_datetimes(self):
        """Test conversion of datetime fields."""
        from datetime import datetime, timezone

        obj = MagicMock()
        now = datetime.now(timezone.utc)
        obj.__dict__ = {
            "fetch_tool_id": "123",
            "fetch_tool_name": "test",
            "created_at": now,
            "updated_at": now,
        }
        obj.__iter__ = MagicMock(return_value=iter(obj.__dict__.items()))

        result = FetchToolRegistry._convert_prisma_to_dict(obj)

        assert result["fetch_tool_id"] == "123"
        assert result["fetch_tool_name"] == "test"
        assert result["created_at"] == now.isoformat()
        assert result["updated_at"] == now.isoformat()

    def test_with_none_datetimes(self):
        """Test with None datetime fields."""
        from datetime import datetime, timezone

        obj = MagicMock()
        now = datetime.now(timezone.utc)
        obj.__dict__ = {
            "created_at": now,
            "updated_at": None,
        }
        obj.__iter__ = MagicMock(return_value=iter(obj.__dict__.items()))

        result = FetchToolRegistry._convert_prisma_to_dict(obj)

        assert result["created_at"] == now.isoformat()
        assert result["updated_at"] is None

    def test_no_datetime_fields(self):
        """Test without any datetime fields."""
        obj = MagicMock()
        obj.__dict__ = {"foo": "bar"}
        obj.__iter__ = MagicMock(return_value=iter(obj.__dict__.items()))

        result = FetchToolRegistry._convert_prisma_to_dict(obj)

        assert result == {"foo": "bar"}


class TestAddFetchTool:
    """Test add_fetch_tool_to_db."""

    @pytest.fixture
    def registry(self):
        return FetchToolRegistry()

    @pytest.fixture
    def mock_prisma(self):
        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_fetchtoolstable = MagicMock()
        return prisma

    @pytest.mark.asyncio
    async def test_success(self, registry, mock_prisma):
        """Test successful creation."""
        mock_tool = MagicMock()
        mock_tool.fetch_tool_id = "abc-123"
        mock_tool.created_at = MagicMock()
        mock_tool.created_at.isoformat.return_value = "2026-05-03T12:00:00"
        mock_tool.updated_at = MagicMock()
        mock_tool.updated_at.isoformat.return_value = "2026-05-03T12:00:00"

        mock_prisma.db.litellm_fetchtoolstable.create = AsyncMock(return_value=mock_tool)

        fetch_tool = {
            "fetch_tool_name": "my-tool",
            "litellm_params": {"provider": "firecrawl"},
            "fetch_tool_info": {"description": "test"},
        }

        result = await registry.add_fetch_tool_to_db(fetch_tool, mock_prisma)

        assert result["fetch_tool_id"] == "abc-123"
        assert result["fetch_tool_name"] == "my-tool"
        mock_prisma.db.litellm_fetchtoolstable.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, registry, mock_prisma):
        """Test error handling."""
        mock_prisma.db.litellm_fetchtoolstable.create = AsyncMock(
            side_effect=Exception("DB error")
        )

        fetch_tool = {"fetch_tool_name": "fail"}

        with pytest.raises(Exception, match="DB error"):
            await registry.add_fetch_tool_to_db(fetch_tool, mock_prisma)


class TestDeleteFetchTool:
    """Test delete_fetch_tool_from_db."""

    @pytest.fixture
    def registry(self):
        return FetchToolRegistry()

    @pytest.fixture
    def mock_prisma(self):
        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_fetchtoolstable = MagicMock()
        return prisma

    @pytest.mark.asyncio
    async def test_success(self, registry, mock_prisma):
        """Test successful deletion."""
        mock_prisma.db.litellm_fetchtoolstable.delete = AsyncMock(return_value=None)

        result = await registry.delete_fetch_tool_from_db("tool-123", mock_prisma)

        assert result["message"] == "Fetch tool tool-123 deleted successfully"
        mock_prisma.db.litellm_fetchtoolstable.delete.assert_awaited_once_with(
            where={"fetch_tool_id": "tool-123"}
        )

    @pytest.mark.asyncio
    async def test_error(self, registry, mock_prisma):
        """Test deletion error."""
        mock_prisma.db.litellm_fetchtoolstable.delete = AsyncMock(
            side_effect=Exception("not found")
        )

        with pytest.raises(Exception, match="not found"):
            await registry.delete_fetch_tool_from_db("bad-id", mock_prisma)


class TestGetFetchTool:
    """Test get_fetch_tool_from_db."""

    @pytest.fixture
    def registry(self):
        return FetchToolRegistry()

    @pytest.fixture
    def mock_prisma(self):
        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_fetchtoolstable = MagicMock()
        return prisma

    @pytest.mark.asyncio
    async def test_found(self, registry, mock_prisma):
        """Test found tool."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        mock_tool = MagicMock()
        mock_tool.__iter__ = MagicMock(
            return_value=iter(
                {
                    "fetch_tool_id": "123",
                    "fetch_tool_name": "test",
                    "created_at": now,
                    "updated_at": now,
                }.items()
            )
        )

        mock_prisma.db.litellm_fetchtoolstable.find_unique = AsyncMock(return_value=mock_tool)

        result = await registry.get_fetch_tool_from_db("123", mock_prisma)

        assert result["fetch_tool_id"] == "123"
        assert result["fetch_tool_name"] == "test"
        assert "created_at" in result

    @pytest.mark.asyncio
    async def test_not_found(self, registry, mock_prisma):
        """Test not found."""
        mock_prisma.db.litellm_fetchtoolstable.find_unique = AsyncMock(return_value=None)

        result = await registry.get_fetch_tool_from_db("missing", mock_prisma)

        assert result is None

    @pytest.mark.asyncio
    async def test_error(self, registry, mock_prisma):
        """Test error handling."""
        mock_prisma.db.litellm_fetchtoolstable.find_unique = AsyncMock(
            side_effect=Exception("db error")
        )

        with pytest.raises(Exception, match="db error"):
            await registry.get_fetch_tool_from_db("bad", mock_prisma)


class TestGetAllFetchTools:
    """Test get_all_fetch_tools_from_db."""

    @pytest.fixture
    def registry(self):
        return FetchToolRegistry()

    @pytest.fixture
    def mock_prisma(self):
        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_fetchtoolstable = MagicMock()
        return prisma

    @pytest.mark.asyncio
    async def test_success(self, registry, mock_prisma):
        """Test listing all tools."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        mock_tool1 = MagicMock()
        mock_tool1.__iter__ = MagicMock(
            return_value=iter(
                {"fetch_tool_id": "1", "fetch_tool_name": "tool1", "created_at": now}.items()
            )
        )
        mock_tool2 = MagicMock()
        mock_tool2.__iter__ = MagicMock(
            return_value=iter(
                {"fetch_tool_id": "2", "fetch_tool_name": "tool2", "created_at": now}.items()
            )
        )

        mock_prisma.db.litellm_fetchtoolstable.find_many = AsyncMock(
            return_value=[mock_tool1, mock_tool2]
        )

        result = await registry.get_all_fetch_tools_from_db(mock_prisma)

        assert len(result) == 2
        assert result[0]["fetch_tool_name"] == "tool1"
        assert result[1]["fetch_tool_name"] == "tool2"

    @pytest.mark.asyncio
    async def test_empty(self, registry, mock_prisma):
        """Test empty list."""
        mock_prisma.db.litellm_fetchtoolstable.find_many = AsyncMock(return_value=[])

        result = await registry.get_all_fetch_tools_from_db(mock_prisma)

        assert result == []


class TestUpdateFetchTool:
    """Test update_fetch_tool_in_db."""

    @pytest.fixture
    def registry(self):
        return FetchToolRegistry()

    @pytest.fixture
    def mock_prisma(self):
        prisma = MagicMock()
        prisma.db = MagicMock()
        prisma.db.litellm_fetchtoolstable = MagicMock()
        return prisma

    @pytest.mark.asyncio
    async def test_success(self, registry, mock_prisma):
        """Test successful update."""
        mock_tool = MagicMock()
        mock_tool.fetch_tool_id = "123"
        mock_tool.created_at = MagicMock()
        mock_tool.created_at.isoformat.return_value = "2026-05-03T12:00:00"
        mock_tool.updated_at = MagicMock()
        mock_tool.updated_at.isoformat.return_value = "2026-05-03T13:00:00"

        mock_prisma.db.litellm_fetchtoolstable.update = AsyncMock(return_value=mock_tool)

        fetch_tool = {
            "fetch_tool_name": "updated-tool",
            "litellm_params": {"provider": "new"},
        }

        result = await registry.update_fetch_tool_in_db("123", fetch_tool, mock_prisma)

        assert result["fetch_tool_id"] == "123"
        assert result["fetch_tool_name"] == "updated-tool"
        mock_prisma.db.litellm_fetchtoolstable.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_error(self, registry, mock_prisma):
        """Test update error."""
        mock_prisma.db.litellm_fetchtoolstable.update = AsyncMock(
            side_effect=Exception("update failed")
        )

        with pytest.raises(Exception, match="update failed"):
            await registry.update_fetch_tool_in_db("123", {}, mock_prisma)

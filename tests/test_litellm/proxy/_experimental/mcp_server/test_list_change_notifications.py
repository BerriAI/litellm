from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.session import ServerSession

from litellm.proxy._experimental.mcp_server.list_change_notifications import (
    MCPListChangeNotifier,
)


@pytest.mark.asyncio
async def test_notify_lists_changed_fans_out_and_discards_failed_sessions():
    notifier = MCPListChangeNotifier()
    tool_session = MagicMock(spec=ServerSession)
    tool_session.send_tool_list_changed = AsyncMock()
    failed_tool_session = MagicMock(spec=ServerSession)
    failed_tool_session.send_tool_list_changed = AsyncMock(side_effect=RuntimeError("closed"))
    resource_session = MagicMock(spec=ServerSession)
    resource_session.send_resource_list_changed = AsyncMock()

    notifier.remember_tool_list_session(tool_session)
    notifier.remember_tool_list_session(failed_tool_session)
    notifier.remember_resource_list_session(resource_session)

    await notifier.notify_lists_changed()
    await notifier.notify_lists_changed()

    assert tool_session.send_tool_list_changed.await_count == 2
    failed_tool_session.send_tool_list_changed.assert_awaited_once()
    assert resource_session.send_resource_list_changed.await_count == 2

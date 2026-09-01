import logging

import pytest
from mcp.types import ListToolsResult, PaginatedRequestParams
from mcp.types import Tool as MCPTool

import litellm.experimental_mcp_client.pagination as pagination_module
from litellm.experimental_mcp_client.pagination import collect_pages


def _tool(index: int) -> MCPTool:
    return MCPTool(name=f"tool_{index:02d}", inputSchema={"type": "object", "properties": {}})


class _PagedTools:
    """A tools/list upstream serving ``total`` tools ``page_size`` at a time, cursors being offsets."""

    def __init__(self, total: int, page_size: int):
        self._tools = tuple(_tool(i) for i in range(total))
        self._page_size = page_size
        self.cursors_seen: list[str | None] = []

    async def fetch(self, params: PaginatedRequestParams | None) -> ListToolsResult:
        cursor = params.cursor if params is not None else None
        self.cursors_seen.append(cursor)
        start = int(cursor) if cursor else 0
        end = start + self._page_size
        return ListToolsResult(
            tools=list(self._tools[start:end]),
            nextCursor=str(end) if end < len(self._tools) else None,
        )


@pytest.mark.asyncio
async def test_collect_pages_follows_next_cursor_until_exhausted():
    upstream = _PagedTools(total=72, page_size=30)

    tools = await collect_pages(upstream.fetch, lambda page: page.tools, method="tools/list", server="s")

    assert [t.name for t in tools] == [f"tool_{i:02d}" for i in range(72)]
    assert upstream.cursors_seen == [None, "30", "60"], (
        "each page must be requested with the cursor the previous one returned"
    )


@pytest.mark.asyncio
async def test_collect_pages_single_page_makes_one_request():
    upstream = _PagedTools(total=5, page_size=30)

    tools = await collect_pages(upstream.fetch, lambda page: page.tools, method="tools/list", server="s")

    assert len(tools) == 5
    assert upstream.cursors_seen == [None]


@pytest.mark.asyncio
async def test_collect_pages_stops_on_a_repeated_cursor_and_keeps_what_it_read(caplog):
    calls: list[str | None] = []

    async def fetch(params: PaginatedRequestParams | None) -> ListToolsResult:
        calls.append(params.cursor if params else None)
        return ListToolsResult(tools=[_tool(len(calls))], nextCursor="same")

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tools = await collect_pages(fetch, lambda page: page.tools, method="tools/list", server="s")

    assert calls == [None, "same"], "the cursor must be followed once and refused the second time it comes back"
    assert len(tools) == 2
    assert any("repeated cursor" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_collect_pages_honors_the_page_cap(monkeypatch, caplog):
    monkeypatch.setattr(pagination_module, "MCP_LIST_MAX_PAGES", 3)
    upstream = _PagedTools(total=1000, page_size=10)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        tools = await collect_pages(upstream.fetch, lambda page: page.tools, method="tools/list", server="s")

    assert len(upstream.cursors_seen) == 3
    assert len(tools) == 30
    assert any("MCP_LIST_MAX_PAGES" in record.getMessage() for record in caplog.records)

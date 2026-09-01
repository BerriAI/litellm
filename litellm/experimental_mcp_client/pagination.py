"""
Follows ``nextCursor`` on the paginated MCP list operations so a multi-page catalog is read in full.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Final, TypeVar

from mcp import ClientSession, Resource
from mcp.types import PaginatedRequestParams, PaginatedResult, Prompt, ResourceTemplate
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_logger
from litellm.constants import MCP_LIST_MAX_PAGES

TPage = TypeVar("TPage", bound=PaginatedResult)
TItem = TypeVar("TItem")


async def collect_pages(
    fetch_page: Callable[[PaginatedRequestParams | None], Awaitable[TPage]],
    items_of: Callable[[TPage], Sequence[TItem]],
    *,
    method: str,
    server: str,
    cursor: str | None = None,
    seen_cursors: frozenset[str] = frozenset(),
) -> tuple[TItem, ...]:
    page: Final = await fetch_page(None if cursor is None else PaginatedRequestParams(cursor=cursor))
    items: Final = tuple(items_of(page))
    next_cursor: Final = page.nextCursor
    pages_read: Final = len(seen_cursors) + 1
    if next_cursor is None:
        return items
    if next_cursor in seen_cursors:
        verbose_logger.warning(
            "MCP %s from %s repeated cursor %r; returning the %s page(s) read so far",
            method,
            server,
            next_cursor,
            pages_read,
        )
        return items
    if pages_read >= MCP_LIST_MAX_PAGES:
        verbose_logger.warning(
            "MCP %s from %s still paginating after %s pages (LITELLM_MCP_LIST_MAX_PAGES); returning what was read",
            method,
            server,
            pages_read,
        )
        return items
    rest: Final = await collect_pages(
        fetch_page,
        items_of,
        method=method,
        server=server,
        cursor=next_cursor,
        seen_cursors=seen_cursors | frozenset((next_cursor,)),
    )
    return items + rest


async def list_all_tools(session: ClientSession, server: str) -> tuple[MCPTool, ...]:
    return await collect_pages(
        lambda params: session.list_tools(params=params), lambda page: page.tools, method="tools/list", server=server
    )


async def list_all_prompts(session: ClientSession, server: str) -> tuple[Prompt, ...]:
    return await collect_pages(
        lambda params: session.list_prompts(params=params),
        lambda page: page.prompts,
        method="prompts/list",
        server=server,
    )


async def list_all_resources(session: ClientSession, server: str) -> tuple[Resource, ...]:
    return await collect_pages(
        lambda params: session.list_resources(params=params),
        lambda page: page.resources,
        method="resources/list",
        server=server,
    )


async def list_all_resource_templates(session: ClientSession, server: str) -> tuple[ResourceTemplate, ...]:
    return await collect_pages(
        lambda params: session.list_resource_templates(params=params),
        lambda page: page.resourceTemplates,
        method="resources/templates/list",
        server=server,
    )

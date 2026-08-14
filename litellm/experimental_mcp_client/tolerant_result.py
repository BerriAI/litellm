"""
Tolerant parsing of ``tools/call`` results from non-spec-compliant upstream MCP servers.

Some upstream MCP servers emit content blocks that fail SDK-side validation, e.g. an
``EmbeddedResource`` whose ``uri`` is a relative path rather than a URI, or ``text/*``
content shipped as a base64 ``blob`` instead of ``text``. The stock ``ClientSession``
lets a single such block fail the entire ``tools/call`` result with a ``ValidationError``,
discarding content the caller could otherwise use (litellm's own ``MCPClient.call_tool``
degrades that into an opaque ``isError`` text block containing the pydantic traceback).
Blocks that fail validation are degraded to text instead, so the rest of the result
survives.
"""

import base64
import json
from datetime import timedelta
from typing import Any, Final, Mapping, Sequence

from mcp import ClientSession, types
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult as MCPCallToolResult
from pydantic import ValidationError, model_validator


def _block_is_valid(block: object) -> bool:
    try:
        MCPCallToolResult.model_validate({"content": [block]})
    except ValidationError:
        return False
    return True


def _resource_text(resource: Mapping[str, Any]) -> str | None:
    text: Final = resource.get("text")
    if isinstance(text, str):
        return text
    blob: Final = resource.get("blob")
    mime_type: Final = resource.get("mimeType")
    if not isinstance(blob, str) or not isinstance(mime_type, str) or not mime_type.startswith("text/"):
        return None
    try:
        return base64.b64decode(blob, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _as_text_block(block: object) -> dict[str, Any]:
    if isinstance(block, Mapping):
        resource: Final = block.get("resource")
        if isinstance(resource, Mapping):
            text: Final = _resource_text(resource)
            if text is not None:
                return {"type": "text", "text": text}
    return {"type": "text", "text": json.dumps(block, default=str)}


class TolerantCallToolResult(MCPCallToolResult):
    """``CallToolResult`` that degrades invalid content blocks to text instead of raising."""

    @model_validator(mode="before")
    @classmethod
    def _degrade_invalid_content_blocks(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        content: Final = data.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            return data
        if all(_block_is_valid(block) for block in content):
            return data
        return {
            **data,
            "content": [block if _block_is_valid(block) else _as_text_block(block) for block in content],
        }


class TolerantClientSession(ClientSession):
    """``ClientSession`` that parses ``tools/call`` results with ``TolerantCallToolResult``.

    ``ClientSession.call_tool`` hardcodes ``types.CallToolResult`` as the response type passed to
    ``send_request`` with no seam to override just that argument, so this mirrors the mcp==1.28.1
    method body verbatim (including the private ``_validate_tool_result`` call) rather than calling
    ``super().call_tool()``. A future mcp release changing that method's signature or behavior would
    not automatically propagate here.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        request_meta: Final = types.RequestParams.Meta(**meta) if meta is not None else None
        result: Final = await self.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(name=name, arguments=arguments, _meta=request_meta),
                )
            ),
            TolerantCallToolResult,
            request_read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
        )
        if not result.isError:
            await self._validate_tool_result(name, result)
        return result

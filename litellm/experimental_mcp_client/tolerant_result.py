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
from collections.abc import Mapping, Sequence
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Final, Literal, TypedDict  # noqa: TID251  # matches ClientSession.call_tool's real signature

from mcp import ClientSession, types
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult as MCPCallToolResult
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import ReadOnly


def _block_is_valid(block: object) -> bool:
    try:
        MCPCallToolResult.model_validate(MappingProxyType({"content": (block,)}))
    except ValidationError:
        return False
    return True


class _ResourcePayload(BaseModel):
    """Lenient shape of an ``EmbeddedResource.resource``: every field a non-compliant upstream
    might have gotten wrong is optional, so this never itself fails to validate."""

    model_config = ConfigDict(extra="allow")

    text: str | None = None
    blob: str | None = None
    mimeType: str | None = None


class _TextContentBlock(TypedDict):
    type: ReadOnly[Literal["text"]]
    text: ReadOnly[str]


def _resource_text(resource: _ResourcePayload) -> str | None:
    if resource.text is not None:
        return resource.text
    if resource.blob is None or resource.mimeType is None or not resource.mimeType.startswith("text/"):
        return None
    try:
        return base64.b64decode(resource.blob, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _validate_resource(raw_resource: Mapping[str, object]) -> _ResourcePayload | None:
    try:
        return _ResourcePayload.model_validate(raw_resource)
    except ValidationError:
        return None


def _as_text_block(block: object) -> _TextContentBlock:
    if isinstance(block, Mapping):
        raw_resource: Final = block.get("resource")
        if isinstance(raw_resource, Mapping):
            resource: Final = _validate_resource(raw_resource)
            if resource is not None:
                text: Final = _resource_text(resource)
                if text is not None:
                    return _TextContentBlock(type="text", text=text)
    return _TextContentBlock(type="text", text=json.dumps(block, default=str))


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
        return MappingProxyType(
            {
                **data,
                "content": tuple(block if _block_is_valid(block) else _as_text_block(block) for block in content),
            }
        )


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

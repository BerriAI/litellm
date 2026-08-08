"""
Utilities for handling OpenAI Responses API 'namespace' tools when bridging to Chat
Completions providers.

A namespace tool is a grouping container: it carries no callable schema of its own and
holds its callable tools under ``tools``. Chat Completions has no equivalent container,
so the bridge replaces each namespace with the tools it contains, which then go through
the same conversion as any top level tool. Namespaces are flat in practice, and a
namespace that somehow contains another one keeps the inner container, which the
conversion then drops as an unsupported type.
"""

from collections.abc import Sequence
from typing import TypeAlias

from openai.types.responses.tool_param import FunctionToolParam

from litellm.types.llms.openai import OpenAIMcpServerTool

ResponsesAPITool: TypeAlias = FunctionToolParam | OpenAIMcpServerTool


def flatten_namespace_tools(tools: Sequence[ResponsesAPITool]) -> tuple[ResponsesAPITool, ...]:
    """Replace every namespace tool with the tools it contains."""
    return tuple(
        nested for tool in tools for nested in (tool.get("tools") or () if tool.get("type") == "namespace" else (tool,))
    )

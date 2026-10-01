import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

_LC_ROLES: Final = MappingProxyType({"human": "user", "ai": "assistant", "system": "system", "tool": "tool"})


class _ContentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    type: str = ""
    text: str | None = None


_CONTENT_BLOCKS: Final = TypeAdapter(tuple[_ContentBlock, ...])


def content_text(content: object) -> str:
    """Message content as display text: Responses-style block lists keep only their text blocks."""
    if isinstance(content, str):
        return content
    try:
        blocks: Final = _CONTENT_BLOCKS.validate_python(content)
    except ValidationError:
        return json.dumps(content)
    texts: Final = tuple(block.text for block in blocks if block.text is not None)
    if texts or all(block.type == "reasoning" for block in blocks):
        return "\n\n".join(texts)
    return json.dumps(content)


def lc_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """LangChain serialized message (or plain {role, content}) -> {role, content, tool_calls?}."""
    kwargs: Final = message.get("kwargs", message)
    role: Final = _LC_ROLES.get(
        kwargs.get("type") or kwargs.get("role"), kwargs.get("role") or kwargs.get("type") or ""
    )
    out: Final[dict[str, Any]] = {  # mutable-ok: the framework message is built for JSON serialization
        "role": role,
        "content": content_text(kwargs.get("content", "")),
    }
    if kwargs.get("tool_calls"):
        out["tool_calls"] = tuple(
            {"name": t.get("name"), "args": t.get("args")}  # mutable-ok: JSON tool calls need object payloads
            for t in kwargs["tool_calls"]
        )
    if role == "tool" and kwargs.get("name"):
        out["name"] = kwargs["name"]
    return out

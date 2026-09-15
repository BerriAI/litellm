"""Codex CLI wire-format quirks shared by the Responses API providers that need them.

Codex sends history item types that api.openai.com accepts but other Responses
backends reject with ``400 Invalid 'input': value did not match any expected
variant``. Both Amazon Bedrock endpoints reject them:

- ``bedrock-mantle.{region}.api.aws`` (verified against ``openai.gpt-5.6-sol``)
- ``bedrock-runtime.{region}.amazonaws.com/openai/v1`` (same, verified separately)

They are *history* items, so they only appear from the second turn of a session
onward -- a first-turn request succeeds and hides the problem entirely.

The normalizer is a pure transform that reports which types it rewrote; callers do
their own logging, so each provider keeps its own wording.
"""

import json
from collections.abc import Mapping
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.openai import ResponseInputParam

AGENT_MESSAGE_INPUT_ITEM_TYPE: Final = "agent_message"
CONTEXT_COMPACTION_INPUT_ITEM_TYPE: Final = "context_compaction"
LOCAL_SHELL_CALL_INPUT_ITEM_TYPE: Final = "local_shell_call"


class _RewrittenOutputTextBlock(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]


class _RewrittenAssistantMessageItem(TypedDict):
    type: ReadOnly[str]
    role: ReadOnly[str]
    content: ReadOnly[tuple[_RewrittenOutputTextBlock, ...]]


class _RewrittenCompactionItem(TypedDict):
    type: ReadOnly[str]
    encrypted_content: ReadOnly[str]


class _RewrittenFunctionCallItem(TypedDict):
    type: ReadOnly[str]
    call_id: ReadOnly[str]
    name: ReadOnly[str]
    arguments: ReadOnly[str]


def _agent_message_text(item: "Mapping[str, object]") -> str:
    content: Final = item.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or block.get("encrypted_content") or "") for block in content if isinstance(block, dict)
    )


def _normalize_agent_message_item(item: "Mapping[str, object]") -> "_RewrittenAssistantMessageItem | None":
    text: Final = _agent_message_text(item)
    if not text:
        return None
    rewritten: Final[_RewrittenAssistantMessageItem] = {
        "type": "message",
        "role": "assistant",
        "content": ({"type": "output_text", "text": text},),
    }
    return rewritten


def _normalize_context_compaction_item(item: "Mapping[str, object]") -> "_RewrittenCompactionItem | None":
    encrypted_content: Final = item.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    rewritten: Final[_RewrittenCompactionItem] = {"type": "compaction", "encrypted_content": encrypted_content}
    return rewritten


def _normalize_local_shell_call_item(item: "Mapping[str, object]") -> "_RewrittenFunctionCallItem | None":
    call_id: Final = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        return None
    action: Final = item.get("action")
    rewritten: Final[_RewrittenFunctionCallItem] = {
        "type": "function_call",
        "call_id": call_id,
        "name": "local_shell",
        "arguments": json.dumps(action) if isinstance(action, dict) else "{}",
    }
    return rewritten


def _normalize_input_item(item: object) -> "tuple[object, str | None]":
    """Returns (normalized item, or None to drop it; original type when rewritten)."""
    if not isinstance(item, dict):
        return item, None
    item_type: Final = item.get("type")
    if item_type == AGENT_MESSAGE_INPUT_ITEM_TYPE:
        return _normalize_agent_message_item(item), item_type
    if item_type == CONTEXT_COMPACTION_INPUT_ITEM_TYPE:
        return _normalize_context_compaction_item(item), item_type
    if item_type == LOCAL_SHELL_CALL_INPUT_ITEM_TYPE:
        return _normalize_local_shell_call_item(item), item_type
    return item, None


def normalize_codex_input_items(
    input: "str | ResponseInputParam",
) -> "tuple[str | ResponseInputParam, tuple[str, ...]]":
    """Rewrite the Codex history item types a Responses backend rejects.

    ``agent_message`` (Codex multi-agent traffic; its ``encrypted_content`` slot
    carries the plaintext payload when the model never issued encrypted args)
    becomes an assistant message, ``context_compaction`` becomes the ``compaction``
    spelling these backends accept, and ``local_shell_call`` becomes the
    ``function_call`` its recorded ``function_call_output`` already pairs with.

    Returns the normalized input and the sorted set of types that were rewritten,
    so the caller can log in its own words. Non-list input is returned untouched.
    """
    if not isinstance(input, list):
        return input, ()
    normalized: Final = tuple(_normalize_input_item(item) for item in input)
    rewritten_types: Final = tuple(sorted(frozenset(item_type for _, item_type in normalized if item_type is not None)))
    kept: Final = [i for i, _ in normalized if i is not None]  # mutable-ok: downstream narrows on isinstance(list)
    # Codex passthrough items sit outside the OpenAI input union.
    return kept, rewritten_types  # pyright: ignore[reportReturnType]  # see above

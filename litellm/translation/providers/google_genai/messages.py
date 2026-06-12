"""IR messages -> generateContent ``contents`` (and ``system_instruction``).

Mirrors v1's ``_gemini_convert_messages_with_history`` for the IR surface:
user/system turns merge (the inbound parser already merged them), assistant
turns become role ``model``, tool results become their own ``user`` turn of
``function_response`` parts flushed before the next regular turn (an IR user
message therefore splits into maximal runs of tool-result vs other blocks),
thinking blocks with a signature become ``thoughtSignature`` parts (v1
json.loads-spreads the thinking string when it parses to an object),
signature-less and redacted thinking emit nothing, and tool invokes are
``function_call`` parts deduped against identical parts. Shapes whose v1
path performs I/O (http(s) downloads on AI Studio, gs:// metadata) or is
ambiguous post-IR (empty user text) fail closed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from expression import Nothing
from expression.collections import Block

from ...errors import TranslationError
from ...ir import (
    THOUGHT_SIGNATURE_SEPARATOR,
    ContentBlock,
    Image,
    Message,
    PlainJson,
    SystemText,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
)
from .params import (
    DUMMY_THOUGHT_SIGNATURE,
    GoogleTarget,
    forwards_function_call_id,
    is_gemini_3_or_newer,
)

_Parts = list[PlainJson]
_ContentsResult = list[PlainJson] | TranslationError

# v1's known-extension mime sniff for https URLs
# (_get_image_mime_type_from_url); anything else would force a download.
_URL_MIME_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/mov",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpg",
        ".avi": "video/avi",
        ".wmv": "video/wmv",
        ".mpegps": "video/mpegps",
        ".flv": "video/flv",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".mpga": "audio/mpeg",
        ".pdf": "application/pdf",
    }
)

_GEMINI_FILES_PREFIX = "https://generativelanguage.googleapis.com/v1beta/files/"


def serialize_system(system: Block[SystemText]) -> PlainJson | None:
    if len(system) == 0:
        return None
    return {"parts": [{"text": entry.text} for entry in system]}


def serialize_contents(
    messages: Block[Message], model: str, target: GoogleTarget
) -> _ContentsResult:
    contents: list[PlainJson] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        if message.role == "assistant":
            converted = _assistant_parts(message.content, model, target)
            if isinstance(converted, TranslationError):
                return converted
            parts, names = converted
            if names:
                tool_names = names  # v1: the LAST message with tool calls wins
            if parts:
                contents = [*contents, {"role": "model", "parts": parts}]
            continue
        runs = _user_runs(message.content)
        for is_tool_run, blocks in runs:
            built = (
                _tool_result_parts(blocks, model, target, tool_names)
                if is_tool_run
                else _user_parts(blocks, model, target)
            )
            if isinstance(built, TranslationError):
                return built
            if built:
                contents = [*contents, {"role": "user", "parts": built}]
    if not contents:
        contents = [{"role": "user", "parts": [{"text": " "}]}]
    return contents


def _user_runs(
    content: Block[ContentBlock],
) -> list[tuple[bool, list[ContentBlock]]]:
    # runs is a local build-then-freeze accumulator; it never escapes
    runs: list[tuple[bool, list[ContentBlock]]] = []
    for block in content:
        is_tool = block.tag == "tool_result"
        if runs and runs[-1][0] == is_tool:
            runs[-1][1].append(block)  # nosemgrep: translation-no-mutation
        else:
            runs.append((is_tool, [block]))  # nosemgrep: translation-no-mutation
    return runs


def _user_parts(
    blocks: list[ContentBlock], model: str, target: GoogleTarget
) -> _Parts | TranslationError:
    parts: _Parts = []
    for block in blocks:
        if block.tag == "text":
            part = _user_text_part(block.text)
        elif block.tag == "image":
            part = _media_part(block.image, model, target)
        else:
            return TranslationError.of_unsupported(
                f"{block.tag} block in a user turn; v1 handles it"
            )
        if isinstance(part, TranslationError):
            return part
        if part is not None:
            parts = [*parts, part]
    if parts and not any(isinstance(p, dict) and "text" in p for p in parts):
        parts = [*parts, {"text": " "}]  # v1's no-text-in-content guard
    return parts


def _user_text_part(text: Text) -> PlainJson | None | TranslationError:
    if len(text.text) == 0:
        # v1 keeps an empty STRING content as {"text": ""} but skips an empty
        # list part; the IR cannot tell them apart (same ambiguity converse
        # declared) so the shape stays on v1.
        return TranslationError.of_unsupported(
            "empty user text block is ambiguous post-IR (string vs list part); v1 handles it"
        )
    return {"text": text.text}


def _media_part(
    image: Image, model: str, target: GoogleTarget
) -> PlainJson | TranslationError:
    source = image.source
    if source.tag == "base64":
        return {
            "inline_data": {
                "data": source.base64.data,
                "mime_type": source.base64.media_type,
            }
        }
    url = source.url.url
    if target == "gemini":
        return TranslationError.of_unsupported(
            "https media on google ai studio is downloaded to base64 by v1 (network)"
        )
    fmt = source.url.format.default_value(None)
    if url.startswith(_GEMINI_FILES_PREFIX):
        file_data: dict[str, PlainJson] = {"file_uri": url}
        if fmt:
            file_data = {"mime_type": fmt, "file_uri": url}
        return {"file_data": file_data}
    mime = fmt if fmt else _mime_from_url(url)
    if mime is None:
        return TranslationError.of_unsupported(
            "https media without a recognizable extension is downloaded by v1 (network)"
        )
    return {"file_data": {"file_uri": url, "mime_type": mime}}


def _mime_from_url(url: str) -> str | None:
    from urllib.parse import urlparse

    path = urlparse(url.lower()).path
    for extension, mime in _URL_MIME_TYPES.items():
        if path.endswith(extension):
            return mime
    return None


_AssistantResult = tuple[_Parts, dict[str, str]] | TranslationError


def _assistant_parts(
    content: Block[ContentBlock], model: str, target: GoogleTarget
) -> _AssistantResult:
    parts: _Parts = []
    names: dict[str, str] = {}
    seen_non_thinking = False
    for block in content:
        if block.tag in ("thinking", "redacted_thinking"):
            if seen_non_thinking:
                return TranslationError.of_unsupported(
                    "inline thinking after text/tool content; v1's gemini path ignores inline thinking"
                )
            if block.tag == "thinking":
                part = _thinking_part(block.thinking)
                if part is not None:
                    parts = [*parts, part]
            continue
        seen_non_thinking = True
        if block.tag == "text":
            parts = [*parts, {"text": block.text.text}]
        elif block.tag == "tool_use":
            part = _function_call_part(block.tool_use, model, target)
            if isinstance(part, TranslationError):
                return part
            names = {**names, block.tool_use.id: block.tool_use.name}
            if not _part_exists(parts, part):
                parts = [*parts, part]
        else:
            return TranslationError.of_unsupported(
                f"{block.tag} block in an assistant turn; v1 handles it"
            )
    return parts, names


def _thinking_part(thinking: Thinking) -> PlainJson | None:
    signature = thinking.signature.default_value(None)
    if signature is None:
        return None  # v1 only forwards thinking blocks that carry a signature
    try:
        decoded: PlainJson = json.loads(thinking.thinking)
    except ValueError:
        decoded = None
    if isinstance(decoded, dict):
        return {"thoughtSignature": signature, **decoded}
    return {"thoughtSignature": signature, "text": thinking.thinking}


def _function_call_part(
    tool_use: ToolUse, model: str, target: GoogleTarget
) -> PlainJson | TranslationError:
    arguments = tool_use.arguments.value
    if not isinstance(arguments, dict):
        return TranslationError.of_unsupported(
            "non-object tool_call arguments; v1 forwards them to json.loads verbatim"
        )
    args_copy: dict[str, PlainJson] = dict(arguments)
    function_call: dict[str, PlainJson] = {"name": tool_use.name, "args": args_copy}
    clean_id, _, signature_suffix = tool_use.id.partition(THOUGHT_SIGNATURE_SEPARATOR)
    if forwards_function_call_id(model, target) and clean_id:
        function_call = {**function_call, "id": clean_id}
    part: dict[str, PlainJson] = {"function_call": function_call}
    signature = signature_suffix or None
    if signature is None and is_gemini_3_or_newer(model):
        signature = DUMMY_THOUGHT_SIGNATURE
    if signature:
        part = {**part, "thoughtSignature": signature}
    return part


def _part_exists(parts: _Parts, part: PlainJson) -> bool:
    """v1 ``check_if_part_exists_in_parts`` (thoughtSignature excluded)."""
    if not isinstance(part, dict):
        return False
    candidate = {k: v for k, v in part.items() if k != "thoughtSignature"}
    for existing in parts:
        if not isinstance(existing, dict):
            continue
        stripped = {k: v for k, v in existing.items() if k != "thoughtSignature"}
        if stripped == candidate:
            return True
    return False


def _tool_result_parts(
    blocks: list[ContentBlock],
    model: str,
    target: GoogleTarget,
    tool_names: dict[str, str],
) -> _Parts | TranslationError:
    parts: _Parts = []
    for block in blocks:
        part = _function_response_part(block.tool_result, model, target, tool_names)
        if isinstance(part, TranslationError):
            return part
        parts = [*parts, part]
    return parts


def _function_response_part(
    result: ToolResult,
    model: str,
    target: GoogleTarget,
    tool_names: dict[str, str],
) -> PlainJson | TranslationError:
    content_str = _tool_result_text(result)
    if isinstance(content_str, TranslationError):
        return content_str
    name = tool_names.get(result.tool_use_id, "")
    if not name:
        return TranslationError.of_unsupported(
            "tool result without a matching tool call name; v1 raises"
        )
    response_data = _response_payload(content_str)
    function_response: dict[str, PlainJson] = {"name": name, "response": response_data}
    if forwards_function_call_id(model, target):
        clean_id = result.tool_use_id.partition(THOUGHT_SIGNATURE_SEPARATOR)[0]
        if clean_id:
            function_response = {**function_response, "id": clean_id}
    return {"function_response": function_response}


def _tool_result_text(result: ToolResult) -> str | TranslationError:
    if result.content.tag == "text":
        text = result.content.text
    else:
        text = "".join(part.text for part in result.content.parts)
    if text[:5].lower() == "data:" and ";base64," in text:
        return TranslationError.of_unsupported(
            "data:-URI tool results take v1's functionResponse.parts media path"
        )
    return text


def _response_payload(content_str: str) -> PlainJson:
    stripped = content_str.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed: PlainJson = json.loads(content_str)
        except ValueError:
            return {"content": content_str}
        if isinstance(parsed, dict):
            return parsed
        return {"content": content_str}
    return {"content": content_str}


def with_schema_prompt(messages: Block[Message], prompt: str) -> Block[Message]:
    """v1 appends the response-schema prompt as a NEW user message before
    conversion, which merges into a trailing user turn when one exists."""
    prompt_block = ContentBlock.of_text(Text(text=prompt, cache=Nothing))
    if len(messages) > 0 and messages[len(messages) - 1].role == "user":
        last = messages[len(messages) - 1]
        merged = Message(
            role="user", content=Block.of_seq([*last.content, prompt_block])
        )
        return Block.of_seq([*messages.take(len(messages) - 1), merged])
    return Block.of_seq(
        [*messages, Message(role="user", content=Block.of_seq([prompt_block]))]
    )

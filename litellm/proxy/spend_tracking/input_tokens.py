"""Input-token counting for the budget reservation path.

Tokenizing is the reservation path's dominant CPU cost and is O(prompt), so
counting a large prompt inline stalls every other request on the worker.
Models whose tokenizer the Rust bridge ports (Anthropic, tiktoken cl100k_base
and o200k_base) are counted from the raw body by the bridge, once per distinct
tokenizer, which parses and tokenizes with the GIL released. Everything it
declines, and every model with no Rust tokenizer, is counted in Python, large
prompts in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.responses.litellm_completion_transformation.compaction_items import (
    billable_input_since_newest_compaction,
    holds_compaction_summary,
)
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.token_counter import (
    TOKEN_COUNTER,
    RustTokenCounterFactory,
    RustTokenizer,
    native_count,
    rust_tokenizer,
)

TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS: Final = 30_000

_GEMINI_SYSTEM_INSTRUCTION_FIELDS: Final = ("systemInstruction", "system_instruction")
_GEMINI_CACHED_CONTENT_FIELDS: Final = ("cachedContent", "cached_content")
_GEMINI_FUNCTION_RESPONSE_FIELDS: Final = ("functionResponse", "function_response")
_GEMINI_TEXT_PART_FIELDS: Final = frozenset({"text", "thought", "thoughtSignature", "thought_signature"})
_GEMINI_COUNTED_PART_FIELDS: Final = _GEMINI_TEXT_PART_FIELDS | {
    "functionCall",
    "function_call",
    *_GEMINI_FUNCTION_RESPONSE_FIELDS,
}
_GEMINI_COUNTED_ROLES: Final = frozenset({"user", "model"})
_GEMINI_COUNTED_TOOL_FIELDS: Final = frozenset({"functionDeclarations"})
_GeminiContents: TypeAlias = list[dict[str, object]] | dict[str, object]
_GEMINI_CONTENTS: Final = TypeAdapter[_GeminiContents](_GeminiContents)
_INPUT_SIZE_FIELDS: Final = (
    "messages",
    "prompt",
    "input",
    "query",
    "documents",
    "contents",
    *_GEMINI_SYSTEM_INSTRUCTION_FIELDS,
    "tools",
    "tool_choice",
)


def _approximate_input_size(request_body: Mapping[str, object]) -> int:
    """Length of the request's input text, a cheap stand-in for tokenizing cost.

    Every field count_input_tokens_for_model hands the tokenizer is sized here,
    and rendering rather than walking keeps mapping keys in the total, which a
    tool schema's property names are."""
    return sum(len(str(request_body.get(field, ""))) for field in _INPUT_SIZE_FIELDS)


async def count_input_tokens(
    request_body: dict[str, object],
    raw_body: bytes | None,
    models: Sequence[str],
) -> Mapping[str, int]:
    """Input-token count per model, sharing one native count across models that
    select the same tokenizer."""
    tokenizers: Final[tuple[tuple[str, RustTokenizer | None], ...]] = tuple(
        (model, rust_tokenizer(model)) for model in models
    )
    groups: Final[tuple[RustTokenizer | None, ...]] = tuple(dict.fromkeys(tokenizer for _, tokenizer in tokenizers))
    counted_raw_body: Final = None if holds_compaction_summary(request_body.get("input")) else raw_body
    group_counts: Final = [
        await _count_group(
            request_body=request_body,
            raw_body=counted_raw_body,
            tokenizer=tokenizer,
            models=tuple(model for model, selected in tokenizers if selected == tokenizer),
        )
        for tokenizer in groups
    ]
    counts: Final = MappingProxyType({model: tokens for group in group_counts for model, tokens in group.items()})
    verbose_proxy_logger.debug("input token counts: %s", dict(counts))
    return counts


async def _count_group(
    request_body: dict[str, object],
    raw_body: bytes | None,
    tokenizer: RustTokenizer | None,
    models: tuple[str, ...],
) -> Mapping[str, int]:
    async def python() -> Mapping[str, int]:
        if _approximate_input_size(request_body) < TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS:
            return _count_input_tokens_for_models(request_body=request_body, models=models)
        return await asyncio.to_thread(
            _count_input_tokens_for_models,
            request_body=request_body,
            models=models,
        )

    if tokenizer is None or raw_body is None:
        return await python()
    try:
        return await runtime.arun(
            RouteContext(Route.TOKEN_COUNTER, provider=tokenizer),
            binding=TOKEN_COUNTER,
            native=lambda factory: _native_counts(factory, tokenizer, raw_body, models),
            python=python,
        )
    except (RuntimeError, ValueError) as error:
        from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted, ProcessReservedForForking

        if isinstance(error, (ForkedAfterNativeRuntimeStarted, ProcessReservedForForking)):
            raise
        verbose_proxy_logger.debug("Rust token counter (%s) failed, counting in Python: %s", tokenizer, error)
        return await python()


async def _native_counts(
    factory: RustTokenCounterFactory,
    tokenizer: RustTokenizer,
    raw_body: bytes,
    models: tuple[str, ...],
) -> Mapping[str, int]:
    count: Final = await native_count(factory, tokenizer, raw_body)
    verbose_proxy_logger.debug("Rust token counter (%s) counted %d input tokens", tokenizer, count.input_tokens)
    return MappingProxyType({model: count.input_tokens for model in models})


def _count_input_tokens_for_models(
    request_body: dict[str, object],
    models: Sequence[str],
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            model: tokens
            for model in models
            if (tokens := count_input_tokens_for_model(request_body=request_body, model=model)) is not None
        }
    )


def count_input_tokens_for_model(request_body: dict, model: str) -> int | None:
    try:
        if "messages" in request_body:
            try:
                return litellm.token_counter(
                    model=model,
                    messages=request_body.get("messages") or (),
                    tools=request_body.get("tools"),
                    tool_choice=request_body.get("tool_choice"),
                )
            except ValueError:
                return _count_text_tokens(model=model, text=request_body.get("messages"))
        if "contents" in request_body:
            return _count_gemini_input_tokens(request_body=request_body, model=model)
        if "prompt" in request_body:
            return _count_text_tokens(model=model, text=request_body.get("prompt"))
        if "input" in request_body:
            return _count_text_tokens(
                model=model, text=billable_input_since_newest_compaction(request_body.get("input"))
            )
        if "query" in request_body or "documents" in request_body:
            query_tokens: Final = _count_text_tokens(model=model, text=request_body.get("query"))
            document_tokens: Final = _count_text_tokens(
                model=model,
                text=request_body.get("documents"),
            )
            return query_tokens + document_tokens
    except Exception:
        verbose_proxy_logger.debug("Unable to count input tokens for budget reservation", exc_info=True)
    return None


def _count_gemini_input_tokens(request_body: Mapping[str, object], model: str) -> int | None:
    system_instruction: Final = next(
        (request_body[field] for field in _GEMINI_SYSTEM_INSTRUCTION_FIELDS if field in request_body), None
    )
    contents: Final = _GEMINI_CONTENTS.validate_python(request_body["contents"])
    tools: Final = request_body.get("tools")
    if not _gemini_body_is_countable(request_body, contents, system_instruction, tools):
        return None
    translate: Final = GoogleGenAIAdapter().translate_generate_content_to_completion  # pyright: ignore[reportUnknownMemberType]  # adapter's **kwargs is untyped
    chat_request: Final = translate(model=model, contents=contents, systemInstruction=system_instruction, tools=tools)
    return count_input_tokens_for_model(request_body=chat_request, model=model)


def _gemini_body_is_countable(
    request_body: Mapping[str, object],
    contents: _GeminiContents,
    system_instruction: object,
    tools: object,
) -> bool:
    content_items: Final = contents if isinstance(contents, list) else (contents,)
    system: Final = as_str_mapping(system_instruction)
    return (
        not any(field in request_body for field in _GEMINI_CACHED_CONTENT_FIELDS)
        and all(_gemini_content_is_countable(content) for content in content_items)
        and (
            system_instruction is None
            or (system is not None and _gemini_parts_are_countable(system.get("parts", ()), _GEMINI_TEXT_PART_FIELDS))
        )
        and all(_gemini_tool_is_countable(tool) for tool in iter_items(tools or ()))
    )


def _gemini_content_is_countable(content: Mapping[str, object]) -> bool:
    return content.get("role", "user") in _GEMINI_COUNTED_ROLES and _gemini_parts_are_countable(
        content.get("parts", ()), _GEMINI_COUNTED_PART_FIELDS
    )


def _gemini_parts_are_countable(parts: object, fields: frozenset[str]) -> bool:
    return all(_gemini_part_is_countable(part, fields) for part in iter_items(parts))


def _gemini_part_is_countable(part: object, fields: frozenset[str]) -> bool:
    if isinstance(part, str):
        return True
    mapping: Final = as_str_mapping(part)
    return (
        mapping is not None
        and mapping.keys() <= fields
        and not any("parts" in (as_str_mapping(mapping.get(field)) or ()) for field in _GEMINI_FUNCTION_RESPONSE_FIELDS)
    )


def _gemini_tool_is_countable(tool: object) -> bool:
    mapping: Final = as_str_mapping(tool)
    return mapping is not None and mapping.keys() <= _GEMINI_COUNTED_TOOL_FIELDS


def _count_text_tokens(model: str, text: object) -> int:
    if text is None:
        return 0

    token_count = 0
    stack: Final = [text]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, list):
            stack.extend(item)
            continue
        if isinstance(item, dict):
            token_count += litellm.token_counter(model=model, text=json.dumps(item))
            continue
        token_count += litellm.token_counter(model=model, text=str(item))
    return token_count

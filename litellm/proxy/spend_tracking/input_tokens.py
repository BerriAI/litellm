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
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS, PDF_DATA_URL_PREFIX
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.litellm_core_utils.core_helpers import as_str_mapping, iter_items
from litellm.litellm_core_utils.deployment_pricing import deployment_pricing_models, str_keyed
from litellm.litellm_core_utils.token_counter import (
    deployment_openai_model,
    pdf_page_pricing,
    reads_pdf_pages,
    unresolved_tokenizer_model,
)
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

if TYPE_CHECKING:
    from litellm.router import Router

TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS: Final = 30_000

_OPENAI_PREFIX: Final = "openai/"

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
    gemini_route: bool = False,
) -> Mapping[str, int]:
    """Input-token count per model, sharing one native count across models that
    select the same tokenizer."""
    tokenizers: Final[tuple[tuple[str, RustTokenizer | None], ...]] = tuple(
        (model, rust_tokenizer(model)) for model in models
    )
    groups: Final[tuple[RustTokenizer | None, ...]] = tuple(dict.fromkeys(tokenizer for _, tokenizer in tokenizers))
    counted_raw_body: Final = None if gemini_route or holds_compaction_summary(request_body.get("input")) else raw_body
    group_counts: Final = [
        await _count_group(
            request_body=request_body,
            raw_body=counted_raw_body,
            tokenizer=tokenizer,
            models=tuple(model for model, selected in tokenizers if selected == tokenizer),
            gemini_route=gemini_route,
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
    gemini_route: bool,
) -> Mapping[str, int]:
    async def python() -> Mapping[str, int]:
        if _approximate_input_size(request_body) < TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS:
            return _count_input_tokens_for_models(request_body=request_body, models=models, gemini_route=gemini_route)
        return await asyncio.to_thread(
            _count_input_tokens_for_models,
            request_body=request_body,
            models=models,
            gemini_route=gemini_route,
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
    gemini_route: bool,
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            model: tokens
            for model in models
            if (
                tokens := count_input_tokens_for_model(
                    request_body=request_body, model=model, gemini_route=gemini_route
                )
            )
            is not None
        }
    )


def models_counted_for(model: str, llm_router: Router | None, team_id: str | None) -> tuple[str, ...]:
    """The names to count a request for `model` from `team_id` with, keeping the largest count, one per deployment
    routing would serve it from: the team's or else the shared ones named `model`, its alias target, or its routing
    group's members, else those of the one wildcard route that serves it.

    An OpenAI deployment counts as its OpenAI model, so it gets that model's tokenizer and PDF page pricing, prefixed
    `openai/` when OpenAI itself serves it, so `openai_chat_pdf_file_tokens` can add what its chat completions bills. A
    deployment that reads PDF pages and prices them unlike `model` counts as its model minus the provider prefix, the
    name `/v1/messages/count_tokens` counts with. Any other deployment counts as the requested name with the encoding
    tiktoken gives that name as written, without loading its vendor's tokenizer. A name no deployment serves counts as
    written."""
    deployments: Final = _routed_deployments(model, llm_router, team_id) if llm_router else ()
    return tuple(dict.fromkeys(_counted_name(model, deployment) for deployment in deployments)) or (model,)


def _routed_deployments(model: str, llm_router: Router, team_id: str | None) -> Sequence[object]:
    registry: Final = llm_router.registry()
    resolved: Final = registry.resolve(model, kind="alias") or model
    routing_group: Final = registry.routing_group(resolved)
    members: Final = routing_group.models if routing_group is not None else (resolved,)
    named: Final = tuple(
        chain.from_iterable(registry.deployments(member, source="all", team_id=team_id) for member in members)
    )
    return named or registry.deployments(resolved, source="wildcard", team_id=team_id)


def openai_chat_pdf_file_tokens(counted_model: str, pdf_files: int) -> int:
    """What OpenAI chat completions bills on top of the counted pages of `pdf_files` inline PDF files, when
    `counted_model`, a name from `models_counted_for`, is served by OpenAI itself."""
    return OPENAI_CHAT_PDF_FILE_TOKENS * pdf_files if counted_model.startswith(_OPENAI_PREFIX) else 0


def inline_pdf_file_count(messages: object) -> int:
    return sum(1 for part in _content_parts(messages) if _is_inline_pdf_file(part))


def _content_parts(messages: object) -> Iterator[Mapping[str, object]]:
    contents: Final = (_as_list(str_keyed(message).get("content")) for message in _as_list(messages))
    return (str_keyed(part) for part in chain.from_iterable(contents))


def _as_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()


def _is_inline_pdf_file(part: Mapping[str, object]) -> bool:
    file_data: Final = str_keyed(part.get("file")).get("file_data")
    return part.get("type") == "file" and isinstance(file_data, str) and file_data.startswith(PDF_DATA_URL_PREFIX)


def _counted_name(model: str, deployment: object) -> str:
    openai_model: Final = deployment_openai_model(deployment)
    if openai_model is not None and _served_by_openai(deployment):
        return _OPENAI_PREFIX + openai_model.removeprefix(_OPENAI_PREFIX)
    if openai_model is not None:
        return openai_model
    served: Final = next(iter(deployment_pricing_models(deployment)), model)
    if pdf_page_pricing(served) != pdf_page_pricing(model) and reads_pdf_pages(served):
        return served.partition("/")[2] or served
    return unresolved_tokenizer_model(model)


def _served_by_openai(deployment: object) -> bool:
    params: Final = str_keyed(str_keyed(deployment).get("litellm_params"))
    model: Final = params.get("model")
    provider: Final = params.get("custom_llm_provider")
    if not isinstance(model, str):
        return False
    try:
        _, served_provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=provider if isinstance(provider, str) else None
        )
    except litellm.exceptions.BadRequestError:
        return False
    return served_provider == "openai"


def count_input_tokens_for_model(request_body: dict, model: str, gemini_route: bool = False) -> int | None:
    try:
        if gemini_route:
            return (
                _count_gemini_input_tokens(request_body=request_body, model=model)
                if "contents" in request_body
                else None
            )
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
    single_part: Final = as_str_mapping(parts)
    items: Final = (single_part,) if single_part is not None else iter_items(parts)
    return all(_gemini_part_is_countable(part, fields) for part in items)


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

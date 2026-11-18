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
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS, PDF_DATA_URL_PREFIX
from litellm.google_genai.adapters.transformation import genai_media_part, google_genai_request_config
from litellm.litellm_core_utils.audio_utils.utils import audio_duration_tokens
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.deployment_pricing import deployment_pricing_models, str_keyed
from litellm.litellm_core_utils.prompt_templates.tpm_media_utils import split_media_for_tpm
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

REQUEST_BODY_FIELDS: Final = TypeAdapter(Mapping[str, object])

_INPUT_SIZE_FIELDS: Final = (
    "messages",
    "prompt",
    "input",
    "query",
    "documents",
    "tools",
    "tool_choice",
    "contents",
    "systemInstruction",
    "system_instruction",
)


def _approximate_input_size(request_body: Mapping[str, object]) -> int:
    """Length of the request's input text, a cheap stand-in for tokenizing cost.

    Every field count_input_tokens_for_model hands the tokenizer is sized here,
    and rendering rather than walking keeps mapping keys in the total, which a
    tool schema's property names are."""
    return sum(len(str(request_body.get(field, ""))) for field in _INPUT_SIZE_FIELDS)


async def count_input_tokens(
    request_body: dict,
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
    request_body: dict,
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
    request_body: dict,
    models: Sequence[str],
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            model: tokens
            for model in models
            if (tokens := count_input_tokens_for_model(request_body=request_body, model=model)) is not None
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
        if "contents" in request_body:
            return _count_gemini_tokens(request_body=REQUEST_BODY_FIELDS.validate_python(request_body), model=model)
    except Exception:
        verbose_proxy_logger.debug("Unable to count input tokens for budget reservation", exc_info=True)
    return None


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


_GEMINI_SYSTEM_INSTRUCTION_FIELDS: Final = ("systemInstruction", "system_instruction")
_GEMINI_FUNCTION_DECLARATION_FIELDS: Final = frozenset({"functionDeclarations", "function_declarations"})
_GEMINI_INLINE_DATA_FIELDS: Final = ("inlineData", "inline_data")
_GEMINI_FILE_DATA_FIELDS: Final = ("fileData", "file_data")
_GEMINI_FUNCTION_RESPONSE_FIELDS: Final = ("functionResponse", "function_response")
_GEMINI_UNBILLED_PART_FIELDS: Final = frozenset({"thoughtSignature", "thought_signature"})
_GEMINI_RESPONSE_SCHEMA_FIELDS: Final = (
    "responseSchema",
    "response_schema",
    "responseJsonSchema",
    "response_json_schema",
)


def _count_gemini_tokens(request_body: Mapping[str, object], model: str) -> int | None:
    system_parts: Final = (_gemini_parts(request_body.get(field)) for field in _GEMINI_SYSTEM_INSTRUCTION_FIELDS)
    parts: Final = chain(_gemini_parts(request_body.get("contents")), *system_parts)
    return _sum_if_all_sized(
        (
            _gemini_function_declaration_tokens(request_body.get("tools"), model),
            _gemini_response_schema_tokens(request_body, model),
            *(_gemini_part_tokens(part, model) for part in parts),
        )
    )


def _gemini_response_schema_tokens(request_body: Mapping[str, object], model: str) -> int:
    config: Final = as_str_mapping(google_genai_request_config(request_body))
    if config is None:
        return 0
    return sum(_count_text_tokens(model=model, text=config.get(field)) for field in _GEMINI_RESPONSE_SCHEMA_FIELDS)


def _sum_if_all_sized(counts: Sequence[int | None]) -> int | None:
    sized: Final = tuple(count for count in counts if count is not None)
    return sum(sized) if len(sized) == len(counts) else None


def _gemini_items(value: object) -> Sequence[object]:
    if value is None:
        return ()
    items: Final = as_object_list(value)
    return (value,) if items is None else items


def _gemini_parts(contents: object) -> Iterator[object]:
    return chain.from_iterable(map(_gemini_content_parts, _gemini_items(contents)))


def _gemini_content_parts(content: object) -> Sequence[object]:
    fields: Final = as_str_mapping(content)
    if fields is None or "parts" not in fields:
        return (content,)
    return _gemini_items(fields.get("parts"))


def _gemini_function_declaration_tokens(tools: object, model: str) -> int | None:
    entries: Final = tuple(map(as_str_mapping, _gemini_items(tools)))
    if any(entry is None or not entry.keys() <= _GEMINI_FUNCTION_DECLARATION_FIELDS for entry in entries):
        return None
    return _count_text_tokens(model=model, text=tools)


def _gemini_part_tokens(part: object, model: str) -> int | None:
    fields: Final = as_str_mapping(part)
    if fields is None:
        return _count_text_tokens(model=model, text=part)
    if any(field in fields for field in _GEMINI_FILE_DATA_FIELDS):
        return None
    if any(field in fields for field in _GEMINI_INLINE_DATA_FIELDS):
        return _gemini_inline_audio_tokens(fields, model)
    text: Final = fields.get("text")
    if isinstance(text, str):
        return _count_text_tokens(model=model, text=text)
    return _gemini_structured_part_tokens(fields, model)


def _gemini_inline_audio_tokens(part: Mapping[str, object], model: str) -> int | None:
    block: Final = genai_media_part(part)
    if block is None:
        return None
    _, media = split_media_for_tpm((block,))
    audio: Final = tuple(map(_input_audio_data, media))
    if not audio:
        return None
    return _sum_if_all_sized(tuple(audio_duration_tokens(data, (model,), model) if data else None for data in audio))


def _input_audio_data(block: Mapping[str, object]) -> str | None:
    audio: Final = as_str_mapping(block.get("input_audio")) if block.get("type") == "input_audio" else None
    data: Final = audio.get("data") if audio is not None else None
    return data if isinstance(data, str) else None


def _gemini_structured_part_tokens(part: Mapping[str, object], model: str) -> int | None:
    response_field: Final = next((field for field in _GEMINI_FUNCTION_RESPONSE_FIELDS if field in part), None)
    response: Final = as_str_mapping(part.get(response_field)) if response_field is not None else None
    billed: Final = {key: value for key, value in part.items() if key not in _GEMINI_UNBILLED_PART_FIELDS}
    if response_field is None or response is None:
        return _count_text_tokens(model=model, text=billed)
    response_without_parts: Final = {key: value for key, value in response.items() if key != "parts"}
    return _sum_if_all_sized(
        (
            _count_text_tokens(model=model, text={**billed, response_field: response_without_parts}),
            *(_gemini_part_tokens(nested, model) for nested in _gemini_items(response.get("parts"))),
        )
    )

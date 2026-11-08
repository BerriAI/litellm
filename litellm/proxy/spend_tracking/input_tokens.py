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

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS, PDF_DATA_URL_PREFIX
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

_INPUT_SIZE_FIELDS: Final = ("messages", "prompt", "input", "query", "documents", "tools", "tool_choice")


def _cheap_payload_size(val: object) -> int:
    if isinstance(val, str):
        return len(val)
    if isinstance(val, (int, float, bool)):
        return 10
    if isinstance(val, Mapping):
        return sum(len(str(k)) + _cheap_payload_size(v) for k, v in val.items())
    if isinstance(val, (list, tuple, set)):
        return sum(_cheap_payload_size(item) for item in val)
    return len(str(val))


def _approximate_gemini_part_size(part: Mapping[str, object]) -> int:
    if "inlineData" in part or "inline_data" in part or "fileData" in part or "file_data" in part:
        return 0
    text = part.get("text")
    if isinstance(text, str):
        return len(text)
    for key in (
        "functionResponse",
        "function_response",
        "functionCall",
        "function_call",
        "executableCode",
        "executable_code",
        "codeExecutionResult",
        "code_execution_result",
    ):
        val = part.get(key)
        if val is not None:
            return _cheap_payload_size(val)
    return 0


def _approximate_gemini_payload_size(payload: object) -> int:
    if isinstance(payload, str):
        return len(payload)
    if isinstance(payload, Mapping):
        parts = payload.get("parts")
        if isinstance(parts, list):
            return sum(_approximate_gemini_payload_size(p) for p in parts)
        text = payload.get("text")
        if isinstance(text, str):
            return len(text)
        return _approximate_gemini_part_size(payload)
    if isinstance(payload, list):
        return sum(_approximate_gemini_payload_size(item) for item in payload)
    return 0


def _approximate_input_size(request_body: Mapping[str, object]) -> int:
    """Length of the request's input text, a cheap stand-in for tokenizing cost.

    Every field count_input_tokens_for_model hands the tokenizer is sized here,
    and rendering rather than walking keeps mapping keys in the total, which a
    tool schema's property names are."""
    size = sum(len(str(request_body.get(field, ""))) for field in _INPUT_SIZE_FIELDS)
    contents = request_body.get("contents")
    if contents is not None:
        size += _approximate_gemini_payload_size(contents)
    sys_inst = request_body.get("systemInstruction") or request_body.get("system_instruction")
    if sys_inst is not None:
        size += _approximate_gemini_payload_size(sys_inst)
    response_schema = _extract_gemini_response_schema(request_body)
    if response_schema is not None:
        size += _cheap_payload_size(response_schema)
    return size


async def count_input_tokens(
    request_body: dict,
    raw_body: bytes | None,
    models: Sequence[str],
    route: str | None = None,
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
            route=route,
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
    route: str | None = None,
) -> Mapping[str, int]:
    async def python() -> Mapping[str, int]:
        if _approximate_input_size(request_body) < TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS:
            return _count_input_tokens_for_models(request_body=request_body, models=models, route=route)
        return await asyncio.to_thread(
            _count_input_tokens_for_models,
            request_body=request_body,
            models=models,
            route=route,
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
    route: str | None = None,
) -> Mapping[str, int]:
    return MappingProxyType(
        {
            model: tokens
            for model in models
            if (tokens := count_input_tokens_for_model(request_body=request_body, model=model, route=route)) is not None
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


def _extract_gemini_part_text(part: Mapping[str, object]) -> str | None:
    if not isinstance(part, Mapping):
        return None
    if isinstance(part.get("text"), str):
        return part["text"]
    for key in ("functionResponse", "function_response"):
        if key in part:
            val = part[key]
            try:
                return json.dumps(val) if val is not None else ""
            except Exception:
                return None
    for key in ("functionCall", "function_call"):
        if key in part:
            val = part[key]
            try:
                return json.dumps(val) if val is not None else ""
            except Exception:
                return None
    for key in ("executableCode", "executable_code"):
        if key in part:
            val = part[key]
            try:
                return json.dumps(val) if val is not None else ""
            except Exception:
                return None
    for key in ("codeExecutionResult", "code_execution_result"):
        if key in part:
            val = part[key]
            try:
                return json.dumps(val) if val is not None else ""
            except Exception:
                return None
    return None


_GEMINI_BUILTIN_RETRIEVAL_TOOL_KEYS: Final = frozenset(
    {
        "urlContext",
        "url_context",
        "googleSearch",
        "google_search",
        "googleSearchRetrieval",
        "google_search_retrieval",
        "vertexRetrieval",
        "vertex_retrieval",
        "enterpriseWebSearch",
        "enterprise_web_search",
        "googleMaps",
        "google_maps",
        "retrieval",
    }
)


def _has_gemini_retrieval_tool(tools: object) -> bool:
    if isinstance(tools, Mapping):
        return any(key in tools for key in _GEMINI_BUILTIN_RETRIEVAL_TOOL_KEYS)
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, Mapping) and any(key in tool for key in _GEMINI_BUILTIN_RETRIEVAL_TOOL_KEYS):
                return True
    return False


_GEMINI_RESPONSE_SCHEMA_KEYS: Final = (
    "responseSchema",
    "response_schema",
    "responseJsonSchema",
    "response_json_schema",
)


def _extract_gemini_response_schema(request_body: Mapping[str, object]) -> object | None:
    for key in _GEMINI_RESPONSE_SCHEMA_KEYS:
        if key in request_body:
            return request_body[key]
    for config_key in ("generationConfig", "generation_config", "config"):
        cfg = request_body.get(config_key)
        if isinstance(cfg, Mapping):
            for key in _GEMINI_RESPONSE_SCHEMA_KEYS:
                if key in cfg:
                    return cfg[key]
    return None


def _extract_gemini_text(request_body: Mapping[str, object]) -> str | None:
    extra_body = request_body.get("extra_body")
    if isinstance(extra_body, Mapping):
        if any(
            key in extra_body
            for key in (
                "contents",
                "systemInstruction",
                "system_instruction",
                "tools",
                "cachedContent",
                "cached_content",
                "responseSchema",
                "response_schema",
                "responseJsonSchema",
                "response_json_schema",
                "generationConfig",
                "generation_config",
                "config",
            )
        ):
            return None
    if any(key in request_body for key in ("cachedContent", "cached_content")):
        return None

    texts: list[str] = []

    sys_inst = request_body.get("systemInstruction") or request_body.get("system_instruction")
    if isinstance(sys_inst, Mapping):
        parts = sys_inst.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, Mapping):
                    part_text = _extract_gemini_part_text(part)
                    if part_text is None:
                        return None
                    texts.append(part_text)
                elif isinstance(part, str):
                    texts.append(part)
                else:
                    return None
        elif isinstance(sys_inst.get("text"), str):
            texts.append(sys_inst["text"])
        else:
            return None
    elif isinstance(sys_inst, str):
        texts.append(sys_inst)
    elif sys_inst is not None:
        return None

    tools = request_body.get("tools")
    if tools is not None:
        if _has_gemini_retrieval_tool(tools):
            return None
        try:
            texts.append(json.dumps(tools))
        except Exception:
            return None

    response_schema = _extract_gemini_response_schema(request_body)
    if response_schema is not None:
        if isinstance(response_schema, str):
            texts.append(response_schema)
        else:
            try:
                texts.append(json.dumps(response_schema))
            except Exception:
                return None

    contents = request_body.get("contents")
    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, Mapping):
                parts = item.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, Mapping):
                            part_text = _extract_gemini_part_text(part)
                            if part_text is None:
                                return None
                            texts.append(part_text)
                        elif isinstance(part, str):
                            texts.append(part)
                        else:
                            return None
                elif isinstance(item.get("text"), str):
                    texts.append(item["text"])
                else:
                    return None
            elif isinstance(item, str):
                texts.append(item)
            else:
                return None
    elif isinstance(contents, Mapping):
        parts = contents.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, Mapping):
                    part_text = _extract_gemini_part_text(part)
                    if part_text is None:
                        return None
                    texts.append(part_text)
                elif isinstance(part, str):
                    texts.append(part)
                else:
                    return None
        elif isinstance(contents.get("text"), str):
            texts.append(contents["text"])
        else:
            return None
    elif isinstance(contents, str):
        texts.append(contents)
    elif contents is not None:
        return None

    combined = " ".join(texts)
    return combined if combined else None


_NATIVE_GEMINI_ACTIONS: Final = (
    ":generateContent",
    ":streamGenerateContent",
    "/generateContent",
    "/streamGenerateContent",
)


def is_native_gemini_route(route: str | None) -> bool:
    if not route:
        return False
    clean_route = route.split("?", 1)[0].rstrip("/")
    return clean_route.endswith(_NATIVE_GEMINI_ACTIONS)


def count_input_tokens_for_model(request_body: dict, model: str, route: str | None = None) -> int | None:
    try:
        if is_native_gemini_route(route):
            if "contents" in request_body:
                gemini_text: Final = _extract_gemini_text(request_body)
                if gemini_text is not None:
                    return _count_text_tokens(model=model, text=gemini_text)
                return None
            return None

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
            gemini_text: Final = _extract_gemini_text(request_body)
            if gemini_text is not None:
                return _count_text_tokens(model=model, text=gemini_text)
            return None
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

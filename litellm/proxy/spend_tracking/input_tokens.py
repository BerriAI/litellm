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
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.token_counter import pdf_page_pricing, reads_pdf_pages
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

_INPUT_SIZE_FIELDS: Final = ("messages", "prompt", "input", "query", "documents", "tools", "tool_choice")


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
    """The names a request for `model` from `team_id` is counted with: `model` itself, except for a deployment that
    reads PDF pages and prices them unlike `model`, which is counted with its own model minus the provider prefix,
    the name `/v1/messages/count_tokens` counts with."""
    served: Final = (
        llm_router.resolved_litellm_models(model, team_id, use_base_model=True) if llm_router is not None else ()
    )
    return tuple(dict.fromkeys(_counted_name(model, served_model) for served_model in served)) or (model,)


def _counted_name(model: str, served_model: str) -> str:
    if pdf_page_pricing(served_model) == pdf_page_pricing(model) or not reads_pdf_pages(served_model):
        return model
    return served_model.partition("/")[2] or served_model


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

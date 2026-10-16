"""Per-line records of a finished batch for the loggers listed in batch_line_item_callbacks."""

import asyncio
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate, chain, count
from typing import Final, Literal, TypeAlias

import litellm
from litellm._logging import verbose_logger
from litellm.batches.batch_utils import batch_cost_and_usage, iter_batch_input_lines, parse_batch_output_line
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import init_custom_logger_compatible_class, is_valid_sha256_hash
from litellm.llms.vertex_ai.files.transformation import vertex_batch_row_custom_id
from litellm.types.utils import (
    ModelInfo,
    StandardBatchLineItem,
    StandardBatchLineItemMetadata,
    StandardBatchLineItemsPayload,
    StandardBatchLineItemUsage,
)

BatchFileProvider: TypeAlias = Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "mistral"]

_VERTEX_AI_UNKNOWN_CUSTOM_ID: Final = "unknown"  # what a Vertex AI batch row that carries no custom_id is read as
_LINE: Final = re.compile(rb"[^\n]+")


@dataclass(frozen=True, slots=True)
class BatchLineItemSource:
    batch_id: str
    finished_at: int
    model: str
    custom_llm_provider: str | None
    cost_provider: BatchFileProvider
    model_info: ModelInfo | None
    metadata: StandardBatchLineItemMetadata


def line_item_metadata(attribution: Mapping[str, object]) -> StandardBatchLineItemMetadata:
    return StandardBatchLineItemMetadata(
        user_api_key_hash=_creator_key_hash(attribution),
        user_api_key_alias=_text(attribution, "user_api_key_alias"),
        user_api_key_team_id=_text(attribution, "user_api_key_team_id"),
        user_api_key_team_alias=_text(attribution, "user_api_key_team_alias"),
        user_api_key_user_id=_text(attribution, "user_api_key_user_id"),
        user_api_key_org_id=_text(attribution, "user_api_key_org_id"),
    )


def _text(attribution: Mapping[str, object], name: str) -> str | None:
    value: Final = attribution.get(name)
    return value if isinstance(value, str) else None


def _creator_key_hash(attribution: Mapping[str, object]) -> str | None:
    """The creating credential as a SHA-256 hash. A virtual key is saved hashed already, while a credential saved as
    is, such as an OAuth2 access token, is hashed here so the object never holds it."""
    credential: Final = _text(attribution, "user_api_key_hash")
    if credential is None or is_valid_sha256_hash(credential):
        return credential
    return hashlib.sha256(credential.encode()).hexdigest()


def _line_key(entry: Mapping[str, object]) -> str | None:
    """A line's custom_id. Bedrock calls it recordId, and native Vertex rows carry it in the row key or request labels."""
    key: Final = entry.get("custom_id", entry.get("recordId"))
    return key if isinstance(key, str) else vertex_batch_row_custom_id(entry)


def batch_line_item_parts(
    source: BatchLineItemSource, input_content: bytes, result_contents: Iterable[bytes], max_part_bytes: int
) -> Iterator[StandardBatchLineItemsPayload]:
    """Pair every output and error line with the input line that shares its custom_id, priced one line at a time, in
    parts that each close once their raw request and result lines reach max_part_bytes. Input lines stay unparsed until
    their result line is reached, so only one part's records are held at a time."""
    sized: Final = _sized_line_items(source, input_content, result_contents)
    for part in count():
        lines = tuple(_until_full(sized, max_part_bytes))
        if not lines:
            return
        yield StandardBatchLineItemsPayload(
            batch_id=source.batch_id, finished_at=source.finished_at, part=part, lines=lines
        )


def _sized_line_items(
    source: BatchLineItemSource, input_content: bytes, result_contents: Iterable[bytes]
) -> Iterator[tuple[StandardBatchLineItem, int]]:
    """Each output and error line's record, with the raw bytes of that line and of the input line it pairs with."""
    request_spans: Final = _request_spans_by_custom_id(input_content)
    for raw_result in chain.from_iterable(map(iter_batch_input_lines, result_contents)):
        entry = parse_batch_output_line(raw_result)
        if entry is None:
            continue
        custom_id = _line_key(entry)
        span = _request_span_for(source, request_spans, custom_id)
        request = None if span is None else parse_batch_output_line(input_content[span[0] : span[1]])
        yield (
            _line_item(source, custom_id, request, entry),
            len(raw_result) + (0 if span is None else span[1] - span[0]),
        )


def _until_full(
    sized: Iterator[tuple[StandardBatchLineItem, int]], max_part_bytes: int
) -> Iterator[StandardBatchLineItem]:
    """Take records until their raw bytes reach max_part_bytes, leaving the rest in sized for the next part."""
    for line, part_bytes in accumulate(sized, _with_bytes_so_far):
        yield line
        if part_bytes >= max_part_bytes:
            return


def _with_bytes_so_far(
    so_far: tuple[StandardBatchLineItem, int], line: tuple[StandardBatchLineItem, int]
) -> tuple[StandardBatchLineItem, int]:
    return line[0], so_far[1] + line[1]


def _request_spans_by_custom_id(input_content: bytes) -> Mapping[str | None, tuple[int, int]]:
    """Where each input line starts and ends, by its custom_id. A custom_id that several input lines share, as when a
    Vertex AI embeddings entry is split into one row per input, is paired with none of them."""
    keyed: Final = tuple(_keyed_spans(input_content))
    counts: Final = Counter(key for key, _ in keyed)
    return {key: span for key, span in keyed if counts[key] == 1}


def _keyed_spans(input_content: bytes) -> Iterator[tuple[str | None, tuple[int, int]]]:
    for line in _LINE.finditer(input_content):
        entry = parse_batch_output_line(line[0]) if line[0].strip() else None
        if entry is not None:
            yield _line_key(entry), line.span()


def _line_item(
    source: BatchLineItemSource, custom_id: str | None, request: Mapping[str, object] | None, entry: dict[str, object]
) -> StandardBatchLineItem:
    priced: Final = batch_cost_and_usage(
        (entry,), source.cost_provider, model_name=source.model, model_info=source.model_info
    )
    succeeded: Final = priced.successful_requests > 0
    return StandardBatchLineItem(
        batch_id=source.batch_id,
        custom_id=custom_id,
        status="success" if succeeded else "failure",
        request=request,
        response=entry,
        usage=StandardBatchLineItemUsage(
            prompt_tokens=priced.usage.prompt_tokens,
            completion_tokens=priced.usage.completion_tokens,
            total_tokens=priced.usage.total_tokens,
        )
        if succeeded
        else None,
        response_cost=priced.cost,
        model=source.model,
        custom_llm_provider=source.custom_llm_provider,
        metadata=source.metadata,
    )


def _request_span_for(
    source: BatchLineItemSource, request_spans: Mapping[str | None, tuple[int, int]], custom_id: str | None
) -> tuple[int, int] | None:
    """Where the input line that shares a result line's custom_id is. A Vertex AI line read as "unknown" pairs with
    none."""
    if custom_id is None or (source.cost_provider == "vertex_ai" and custom_id == _VERTEX_AI_UNKNOWN_CUSTOM_ID):
        return None
    return request_spans.get(custom_id)


def without_request_and_response(payload: StandardBatchLineItemsPayload) -> StandardBatchLineItemsPayload:
    return StandardBatchLineItemsPayload(
        batch_id=payload["batch_id"],
        finished_at=payload["finished_at"],
        part=payload["part"],
        lines=tuple(
            StandardBatchLineItem(
                batch_id=line["batch_id"],
                custom_id=line["custom_id"],
                status=line["status"],
                request=None,
                response=None,
                usage=line["usage"],
                response_cost=line["response_cost"],
                model=line["model"],
                custom_llm_provider=line["custom_llm_provider"],
                metadata=line["metadata"],
            )
            for line in payload["lines"]
        ),
    )


async def dispatch_batch_line_items(payload: StandardBatchLineItemsPayload, redact_bodies: bool) -> None:
    """Hand one part's lines to every batch_line_item_callbacks logger. A failing logger is logged and never stops the
    rest."""
    await asyncio.gather(
        *(_deliver(callback, payload, redact_bodies) for callback in litellm.batch_line_item_callbacks)
    )


async def _deliver(
    callback: litellm.CALLBACK_TYPES, payload: StandardBatchLineItemsPayload, redact_bodies: bool
) -> None:
    try:
        logger: Final = _resolve_logger(callback)
        if logger is None:
            verbose_logger.error(
                "batch_line_item_callbacks: %r cannot store batch line items, part %d of batch %s was not stored",
                callback,
                payload["part"],
                payload["batch_id"],
            )
            return
        await logger.async_log_batch_line_items(
            without_request_and_response(payload)
            if redact_bodies or _logger_turned_off_message_logging(logger)
            else payload
        )
    except Exception as error:  # noqa: BLE001  # one failing sink must not stop the other sinks or the cost poller
        verbose_logger.exception(
            "batch_line_item_callbacks: %r failed to store the %d lines of part %d of batch %s: %s",
            callback,
            len(payload["lines"]),
            payload["part"],
            payload["batch_id"],
            error,
        )


def _resolve_logger(callback: litellm.CALLBACK_TYPES) -> CustomLogger | None:
    match callback:
        case CustomLogger():
            return callback
        case "s3_v2" | "gcs_bucket":
            return init_custom_logger_compatible_class(callback, internal_usage_cache=None, llm_router=None)
        case _:
            return None


def _logger_turned_off_message_logging(logger: CustomLogger) -> bool:
    return (
        getattr(logger, "turn_off_message_logging", False) is True
        or getattr(logger, "message_logging", True) is not True
    )

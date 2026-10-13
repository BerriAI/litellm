"""Per-line records of a finished batch for the loggers listed in batch_line_item_callbacks."""

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal, TypeAlias

import litellm
from litellm._logging import verbose_logger
from litellm.batches.batch_utils import batch_cost_and_usage, get_file_content_as_dictionary
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import init_custom_logger_compatible_class
from litellm.llms.vertex_ai.files.transformation import vertex_batch_row_custom_id
from litellm.types.utils import (
    ModelInfo,
    StandardBatchLineItem,
    StandardBatchLineItemMetadata,
    StandardBatchLineItemsPayload,
    StandardBatchLineItemUsage,
)

BatchFileProvider: TypeAlias = Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "mistral"]


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
        user_api_key_hash=_text(attribution, "user_api_key_hash"),
        user_api_key_alias=_text(attribution, "user_api_key_alias"),
        user_api_key_team_id=_text(attribution, "user_api_key_team_id"),
        user_api_key_team_alias=_text(attribution, "user_api_key_team_alias"),
        user_api_key_user_id=_text(attribution, "user_api_key_user_id"),
        user_api_key_org_id=_text(attribution, "user_api_key_org_id"),
    )


def _text(attribution: Mapping[str, object], name: str) -> str | None:
    value: Final = attribution.get(name)
    return value if isinstance(value, str) else None


def _line_key(entry: Mapping[str, object]) -> str | None:
    """A line's custom_id. Bedrock calls it recordId, and native Vertex rows carry it in the row key or request labels."""
    key: Final = entry.get("custom_id", entry.get("recordId"))
    return key if isinstance(key, str) else vertex_batch_row_custom_id(entry)


def build_batch_line_items(
    source: BatchLineItemSource, input_content: bytes, result_contents: Iterable[bytes]
) -> StandardBatchLineItemsPayload:
    """Pair every output and error line with the input line that shares its custom_id, priced one line at a time."""
    requests: Final = {_line_key(entry): entry for entry in get_file_content_as_dictionary(input_content)}
    results: Final = chain.from_iterable(map(get_file_content_as_dictionary, result_contents))
    return StandardBatchLineItemsPayload(
        batch_id=source.batch_id,
        finished_at=source.finished_at,
        lines=tuple([_line_item(source, requests, entry) for entry in results]),
    )


def _line_item(
    source: BatchLineItemSource,
    requests: Mapping[str | None, Mapping[str, object]],
    entry: dict[str, object],
) -> StandardBatchLineItem:
    priced: Final = batch_cost_and_usage(
        (entry,), source.cost_provider, model_name=source.model, model_info=source.model_info
    )
    succeeded: Final = priced.successful_requests > 0
    custom_id: Final = _line_key(entry)
    return StandardBatchLineItem(
        batch_id=source.batch_id,
        custom_id=custom_id,
        status="success" if succeeded else "failure",
        request=requests.get(custom_id) if custom_id is not None else None,
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


def without_request_and_response(payload: StandardBatchLineItemsPayload) -> StandardBatchLineItemsPayload:
    return StandardBatchLineItemsPayload(
        batch_id=payload["batch_id"],
        finished_at=payload["finished_at"],
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
    """Hand the lines to every batch_line_item_callbacks logger. A failing logger is logged and never stops the rest."""
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
                "batch_line_item_callbacks: %r cannot store batch line items, the lines of batch %s were not stored",
                callback,
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
            "batch_line_item_callbacks: %r failed to store the %d lines of batch %s: %s",
            callback,
            len(payload["lines"]),
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

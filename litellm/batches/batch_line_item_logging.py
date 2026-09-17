import json
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm._logging import verbose_logger
from litellm.batches.batch_utils import (
    _batch_response_was_successful,  # pyright: ignore[reportPrivateUsage]  # batch-internal helper shared with the aggregate cost path by design
    _fetch_batch_managed_file_content,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # same reuse; helper is untyped upstream
    _get_response_from_batch_job_output_file,  # pyright: ignore[reportPrivateUsage]  # same reuse
    _iter_batch_output_entries,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # same reuse; helper is untyped upstream
    _safe_output_line_stats,  # pyright: ignore[reportPrivateUsage]  # same reuse
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    EmbeddingResponse,
    LiteLLMBatch,
    ModelInfo,
    ModelResponse,
    Usage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_BatchLineProvider: TypeAlias = Literal["openai", "azure", "vertex_ai", "hosted_vllm", "anthropic"]

_CALL_TYPE_BY_BATCH_URL: Final = MappingProxyType(
    {
        "/v1/chat/completions": "acompletion",
        "/v1/embeddings": "aembedding",
        "/v1/responses": "aresponses",
    }
)

_EMPTY_BODY: Final[Mapping[str, object]] = MappingProxyType({})


class _BatchLineFailure(Exception):
    """A provider-reported per-line batch failure; carries the batch's hidden
    params so the failure logging payload can attribute the line."""

    def __init__(self, error_payload: object) -> None:
        super().__init__(json.dumps(error_payload))
        self._hidden_params: dict[str, object] = {}  # mutable-ok: mirrors the plain-dict _hidden_params contract on litellm response objects


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):  # pyright: ignore[reportUnknownVariableType]  # keys of an unparameterized Mapping are unknown until checked here
        return value  # pyright: ignore[reportUnknownVariableType, reportReturnType]  # every key was verified str above
    return None


def _output_entries(file_content: bytes) -> Iterator[Mapping[str, object]]:
    for entry in _iter_batch_output_entries(file_content):  # pyright: ignore[reportUnknownVariableType]  # entries are validated into typed mappings below
        mapping = _as_object_mapping(entry)  # pyright: ignore[reportUnknownArgumentType]  # raw entry is unknown until validated here
        if mapping is not None:
            yield mapping


def _requests_by_custom_id(input_file_content: bytes) -> Mapping[str, Mapping[str, object]]:
    """Parse the batch input JSONL into {custom_id: request line}, skipping
    malformed lines and lines without a custom_id."""
    return MappingProxyType(
        {
            custom_id: entry
            for entry in _output_entries(input_file_content)
            if isinstance((custom_id := entry.get("custom_id")), str) and custom_id
        }
    )


def _request_body_for_entry(
    entry: Mapping[str, object], request_line: Mapping[str, object] | None
) -> Mapping[str, object]:
    if request_line is not None:
        body: Final = _as_object_mapping(request_line.get("body"))
        if body:
            return body
        params: Final = _as_object_mapping(request_line.get("params"))
        if params:
            return params
    model_input: Final = _as_object_mapping(entry.get("modelInput"))
    return model_input if model_input else _EMPTY_BODY


def _line_status_code(entry: Mapping[str, object], custom_llm_provider: str) -> int | None:
    response: Final = _as_object_mapping(entry.get("response"))
    status: Final = response.get("status_code") if response is not None else None
    if isinstance(status, int):
        return status
    if custom_llm_provider == "anthropic":
        result: Final = _as_object_mapping(entry.get("result"))
        if result is not None and result.get("type") == "succeeded":
            return 200
    return None


def _call_type_for_request(request_line: Mapping[str, object] | None) -> str:
    url: Final = request_line.get("url") if request_line is not None else None
    return _CALL_TYPE_BY_BATCH_URL.get(url if isinstance(url, str) else "", "acompletion")


def _line_messages(request_body: Mapping[str, object]) -> object:
    return request_body.get("messages") or request_body.get("input") or ()


def _line_model(
    response_body: Mapping[str, object],
    request_body: Mapping[str, object],
    parent: "Logging",
) -> str:
    for candidate in (response_body.get("model"), request_body.get("model"), parent.model):
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


_BatchLineResult: TypeAlias = "ModelResponse | EmbeddingResponse | ResponsesAPIResponse"


def _line_result(call_type: str, response_body: Mapping[str, object]) -> _BatchLineResult:
    if call_type == "aembedding":
        return EmbeddingResponse(**response_body)  # pyright: ignore[reportArgumentType]  # provider output bodies are dicts expanded as response ctor kwargs
    if call_type == "aresponses":
        return ResponsesAPIResponse(**response_body)  # pyright: ignore[reportArgumentType]  # same as above
    return ModelResponse(**response_body)  # pyright: ignore[reportArgumentType]  # same as above


def _new_child_logging(
    parent: "Logging",
    model: str,
    messages: object,
    call_type: str,
    start_time: datetime,
) -> "Logging":
    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model=model,
        messages=messages,
        stream=False,
        call_type=call_type,
        start_time=start_time,
        litellm_call_id=str(uuid.uuid4()),
        function_id=str(uuid.uuid4()),
        litellm_trace_id=parent.litellm_trace_id,
        dynamic_success_callbacks=parent.dynamic_success_callbacks,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging ctor takes these dynamic callback lists untyped
        dynamic_async_success_callbacks=parent.dynamic_async_success_callbacks,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # same as above
        dynamic_failure_callbacks=parent.dynamic_failure_callbacks,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # same as above
        dynamic_async_failure_callbacks=parent.dynamic_async_failure_callbacks,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # same as above
        kwargs={"litellm_session_id": parent.litellm_session_id},  # mutable-ok: Logging's kwargs param takes a plain dict
    )


def _line_hidden_params(
    batch: LiteLLMBatch,
    custom_id: object,
    status_code: int | None,
    response_cost: float | None = None,
) -> dict[str, object]:  # mutable-ok: response objects declare _hidden_params as a plain dict
    return {  # mutable-ok: same contract
        "batch_id": batch.id,
        "batch_custom_id": custom_id,
        "batch_line_status_code": status_code,
        "response_cost": response_cost,
    }


def _optional_params_for_body(request_body: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: update_environment_variables takes a plain dict
    return {  # mutable-ok: same contract
        key: value for key, value in request_body.items() if key not in ("model", "messages", "input")
    }


async def _emit_line_event(
    entry: Mapping[str, object],
    request_line: Mapping[str, object] | None,
    batch: LiteLLMBatch,
    custom_llm_provider: _BatchLineProvider,
    parent: "Logging",
    model_name: str | None,
    model_info: ModelInfo | None,
) -> bool:
    custom_id: Final = entry.get("custom_id") or entry.get("recordId")
    request_body: Final = _request_body_for_entry(entry, request_line)
    status_code: Final = _line_status_code(entry, custom_llm_provider)
    call_type: Final = _call_type_for_request(request_line)
    response_body: Final = _get_response_from_batch_job_output_file(entry, custom_llm_provider)
    parent_start_time: Final = parent.start_time  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # Logging.start_time is untyped upstream
    start_time: Final = parent_start_time if isinstance(parent_start_time, datetime) else datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline start_time
    parent_params: Final = _as_object_mapping(parent.litellm_params) or _EMPTY_BODY  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging.litellm_params is untyped upstream

    child: Final = _new_child_logging(
        parent=parent,
        model=_line_model(response_body, request_body, parent),
        messages=_line_messages(request_body),
        call_type=call_type,
        start_time=start_time,
    )
    child.update_environment_variables(  # pyright: ignore[reportUnknownMemberType]  # Logging.update_environment_variables is untyped upstream
        litellm_params={  # mutable-ok: update_environment_variables takes a plain dict
            **parent_params,
            "batch_parent_id": batch.id,
            "metadata": dict(_as_object_mapping(parent_params.get("metadata")) or {}),  # mutable-ok: copy of the parent's metadata dict
        },
        optional_params=_optional_params_for_body(request_body),
        model=child.model,
        custom_llm_provider=custom_llm_provider,
    )

    now: Final = datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline start_time
    if not _batch_response_was_successful(entry, custom_llm_provider):
        exception: Final = _BatchLineFailure(
            entry.get("error") or entry.get("response") or {}  # mutable-ok: fallback payload dict passed to Exception
        )
        exception._hidden_params = _line_hidden_params(batch, custom_id, status_code)  # pyright: ignore[reportPrivateUsage]  # _hidden_params is set on the exception instance itself
        await child.async_failure_handler(
            exception=exception,
            traceback_exception="",
            start_time=start_time,
            end_time=now,
        )
        return True

    stats: Final = _safe_output_line_stats(entry, custom_llm_provider, model_name, model_info)
    try:
        result: Final = _line_result(call_type, response_body)
    except Exception:  # noqa: BLE001  # one unparseable line must not drop the rest of the batch's line events
        verbose_logger.warning(
            "batch output line could not be reconstructed as a %s response, skipping it. custom_id=%s",
            call_type,
            custom_id,
        )
        return False

    result._hidden_params = _line_hidden_params(  # pyright: ignore[reportPrivateUsage]  # same hidden_params channel the aggregate batch event uses
        batch,
        custom_id,
        status_code,
        response_cost=stats.prompt_cost + stats.completion_cost if stats is not None else None,
    )
    if stats is not None and not response_body.get("usage") and isinstance(result, (ModelResponse, EmbeddingResponse)):
        setattr(  # noqa: B010  # ModelResponse.usage is set dynamically by its ctor, so setattr keeps parity for both response types
            result,
            "usage",
            Usage(
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                total_tokens=stats.total_tokens,
            ),
        )
    await child.async_success_handler(
        result=result,
        start_time=start_time,
        end_time=now,
        cache_hit=False,
    )
    return True


async def _fetch_managed_file_or_empty(
    file_id: str | None,
    custom_llm_provider: _BatchLineProvider,
    fetch_params: dict[str, object] | None,  # mutable-ok: batch_utils file fetch takes the shared litellm_params dict
) -> bytes:
    if file_id is None:
        return b""
    return await _fetch_batch_managed_file_content(
        file_id,
        custom_llm_provider=custom_llm_provider,
        litellm_params=fetch_params,  # pyright: ignore[reportArgumentType]  # batch_utils types this param as an unparameterized dict
    )


async def log_batch_line_items(
    batch: LiteLLMBatch,
    custom_llm_provider: _BatchLineProvider,
    parent: "Logging",
    model_name: str | None,
    litellm_params: dict[str, object] | None,  # mutable-ok: the logging object's shared litellm_params dict
    model_info: ModelInfo | None,
) -> int:
    """Emit one callback event per JSONL line of a completed batch (request
    paired with its response/error), behind the opt-in
    ``litellm.store_batch_line_items_in_callbacks`` flag. The aggregate
    aretrieve_batch event still bills the batch, so per-line events carry
    ``batch_parent_id`` and never update spend themselves. Any failure here
    is logged and swallowed: aggregate accounting must be unaffected."""
    emitted = 0  # rebind-ok: loop accumulator for emitted line count
    try:
        internal_credentials: Final = (
            litellm_params.get("_litellm_internal_model_credentials") if litellm_params else None
        )
        internal_mapping: Final = _as_object_mapping(internal_credentials)
        fetch_params: Final[dict[str, object] | None] = (  # mutable-ok: _fetch_batch_managed_file_content requires a plain dict
            dict(internal_mapping)  # mutable-ok: the file fetcher reads credential kwargs off a plain dict
            if internal_mapping is not None
            else litellm_params
        )

        input_file_content: Final = await _fetch_managed_file_or_empty(
            batch.input_file_id, custom_llm_provider, fetch_params
        )
        requests_by_id: Final = _requests_by_custom_id(input_file_content)

        output_content: Final = await _fetch_managed_file_or_empty(
            batch.output_file_id, custom_llm_provider, fetch_params
        )
        error_content: Final = await _fetch_managed_file_or_empty(
            batch.error_file_id, custom_llm_provider, fetch_params
        )
        for content in (output_content, error_content):
            for entry in _output_entries(content):
                try:
                    entry_key = entry.get("custom_id") or entry.get("recordId")  # rebind-ok: per-iteration binding inside a loop cannot carry Final
                    request_line = requests_by_id.get(entry_key if isinstance(entry_key, str) else "")  # rebind-ok: per-iteration binding inside a loop cannot carry Final
                    line_emitted = await _emit_line_event(  # rebind-ok: same per-iteration binding
                        entry=entry,
                        request_line=request_line,
                        batch=batch,
                        custom_llm_provider=custom_llm_provider,
                        parent=parent,
                        model_name=model_name,
                        model_info=model_info,
                    )
                    if line_emitted:
                        emitted += 1
                except Exception:  # noqa: BLE001  # one bad line must not drop the rest of the batch's line events
                    verbose_logger.exception(
                        "batch line item logging failed for entry, continuing with remaining lines. batch_id=%s",
                        batch.id,
                    )
    except Exception:  # noqa: BLE001  # line-item logging must never break the aggregate aretrieve_batch accounting
        verbose_logger.exception(
            "batch line item logging failed for batch_id=%s; aggregate logging unaffected",
            batch.id,
        )
    return emitted

import json
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast, get_args

from litellm._logging import verbose_logger
from litellm.batches.batch_utils import (
    _batch_response_was_successful,  # pyright: ignore[reportPrivateUsage]  # batch-internal helper shared with the aggregate cost path by design
    _fetch_batch_managed_file_content,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # same reuse; helper is untyped upstream
    _get_response_from_batch_job_output_file,  # pyright: ignore[reportPrivateUsage]  # same reuse
    _iter_batch_output_entries,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # same reuse; helper is untyped upstream
    _safe_output_line_stats,  # pyright: ignore[reportPrivateUsage]  # same reuse
    _uses_native_vertex_output,  # pyright: ignore[reportPrivateUsage]  # same reuse
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

_BatchLineProvider: TypeAlias = Literal[
    "openai", "azure", "vertex_ai", "hosted_vllm", "anthropic", "bedrock", "mistral"
]

_SUPPORTED_LINE_PROVIDERS: Final = frozenset(get_args(_BatchLineProvider))

_SECRET_PARAM_KEYS: Final = frozenset(
    {
        "api_key",
        "_litellm_internal_model_credentials",
        "azure_ad_token",
        "azure_ad_token_provider",
        "vertex_credentials",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_web_identity_token",
    }
)


def _supported_line_provider(value: str) -> _BatchLineProvider | None:
    if value in _SUPPORTED_LINE_PROVIDERS:
        return cast("_BatchLineProvider", value)  # cast-ok: membership in the literal's args was just checked
    return None


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
        self._hidden_params: dict[str, object] = {}  # mutable-ok: plain-dict contract like response _hidden_params


def _as_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):  # pyright: ignore[reportUnknownVariableType]  # keys of an unparameterized Mapping are unknown until checked here
        return value  # pyright: ignore[reportUnknownVariableType, reportReturnType]  # every key was verified str above
    return None


def _output_entries(file_content: bytes) -> Iterator[Mapping[str, object]]:
    for entry in _iter_batch_output_entries(file_content):  # pyright: ignore[reportUnknownVariableType]  # entries are validated into typed mappings below
        mapping = _as_object_mapping(entry)  # pyright: ignore[reportUnknownArgumentType]  # raw entry is unknown until validated here
        if mapping is not None:
            yield mapping


def _line_id(entry: Mapping[str, object]) -> str | None:
    line_id: Final = entry.get("custom_id") or entry.get("recordId")
    return line_id if isinstance(line_id, str) and line_id else None


def _requests_by_custom_id(input_file_content: bytes) -> Mapping[str, Mapping[str, object]]:
    """Parse the batch input JSONL into {line id: request line}, keyed by
    custom_id or recordId, skipping malformed lines and lines without either."""
    return MappingProxyType(
        {line_id: entry for entry in _output_entries(input_file_content) if (line_id := _line_id(entry)) is not None}
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
        request_model_input: Final = _as_object_mapping(request_line.get("modelInput"))
        if request_model_input:
            return request_model_input
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


def _line_error_payload(entry: Mapping[str, object], custom_llm_provider: _BatchLineProvider) -> object:
    if custom_llm_provider == "anthropic":
        return (
            (_as_object_mapping(entry.get("result")) or _EMPTY_BODY).get("error") or entry.get("result") or _EMPTY_BODY
        )
    return entry.get("error") or entry.get("response") or _EMPTY_BODY


def _call_type_for_request(request_line: Mapping[str, object] | None) -> str:
    url: Final = request_line.get("url") if request_line is not None else None
    return _CALL_TYPE_BY_BATCH_URL.get(url if isinstance(url, str) else "", "acompletion")


def _call_type_for_line(request_line: Mapping[str, object] | None, result: "_BatchLineResult | None") -> str:
    if isinstance(result, EmbeddingResponse):
        return "aembedding"
    if isinstance(result, ResponsesAPIResponse):
        return "aresponses"
    if isinstance(result, ModelResponse):
        return "acompletion"
    return _call_type_for_request(request_line)


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


def _line_result(
    call_type: str,
    custom_llm_provider: _BatchLineProvider,
    model: str,
    response_body: Mapping[str, object],
) -> _BatchLineResult:
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider_config: Final = ProviderConfigManager.get_provider_batches_config(model, LlmProviders(custom_llm_provider))
    if provider_config is not None:
        transformed: Final = provider_config.transform_batch_output_line(response_body, model)
        if transformed is not None:
            return transformed
        if custom_llm_provider == "bedrock":
            raise ValueError(f"unrecognized bedrock batch output line shape. keys={sorted(response_body)}")
    if call_type == "aembedding":
        return EmbeddingResponse(**response_body)  # pyright: ignore[reportArgumentType]  # provider output bodies are dicts expanded as response ctor kwargs
    if call_type == "aresponses":
        return ResponsesAPIResponse(**response_body)  # pyright: ignore[reportArgumentType]  # same as above
    if custom_llm_provider == "anthropic":
        from litellm.llms.anthropic.chat.transformation import anthropic_message_to_model_response

        return anthropic_message_to_model_response(response_body, None)
    return ModelResponse(**response_body)  # pyright: ignore[reportArgumentType]  # same as above


def _line_result_or_none(
    request_call_type: str,
    custom_llm_provider: _BatchLineProvider,
    model: str,
    response_body: Mapping[str, object],
    custom_id: object,
) -> "_BatchLineResult | None":
    try:
        return _line_result(request_call_type, custom_llm_provider, model, response_body)
    except Exception:  # noqa: BLE001  # one unparseable line must not drop the rest of the batch's line events
        verbose_logger.warning(
            "batch output line could not be reconstructed as a %s response, skipping it. custom_id=%s",
            request_call_type,
            custom_id,
        )
        return None


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
        kwargs={"litellm_session_id": parent.litellm_session_id},  # mutable-ok: kwargs takes a plain dict
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


def _optional_params_for_body(
    request_body: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: update_environment_variables takes a plain dict
    return {  # mutable-ok: same contract
        key: value for key, value in request_body.items() if key not in ("model", "messages", "input")
    }


def _metadata_copy(params: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: dict out for litellm_params
    metadata: Final = _as_object_mapping(params.get("metadata")) or _EMPTY_BODY
    return {**metadata}  # mutable-ok: plain-dict copy


async def _emit_line_event(
    entry: Mapping[str, object],
    requests_by_id: Mapping[str, Mapping[str, object]],
    batch: LiteLLMBatch,
    custom_llm_provider: _BatchLineProvider,
    parent: "Logging",
    model_name: str | None,
    model_info: ModelInfo | None,
) -> bool:
    custom_id: Final = _line_id(entry)
    request_line: Final = requests_by_id.get(custom_id or "")
    request_body: Final = _request_body_for_entry(entry, request_line)
    status_code: Final = _line_status_code(entry, custom_llm_provider)
    request_call_type: Final = _call_type_for_request(request_line)
    response_body: Final = _get_response_from_batch_job_output_file(entry, custom_llm_provider)
    parent_start_time: Final = parent.start_time  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # Logging.start_time is untyped upstream
    start_time: Final = parent_start_time if isinstance(parent_start_time, datetime) else datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline start_time
    parent_params: Final = _as_object_mapping(parent.litellm_params) or _EMPTY_BODY  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Logging.litellm_params is untyped upstream
    model: Final = _line_model(response_body, request_body, parent)

    successful: Final = _batch_response_was_successful(entry, custom_llm_provider)
    stats: Final = _safe_output_line_stats(entry, custom_llm_provider, model_name, model_info) if successful else None
    result: Final[_BatchLineResult | None] = (
        _line_result_or_none(request_call_type, custom_llm_provider, model, response_body, custom_id)
        if successful
        else None
    )
    if successful and result is None:
        return False

    call_type: Final = _call_type_for_line(request_line, result)
    child: Final = _new_child_logging(
        parent=parent,
        model=model,
        messages=_line_messages(request_body),
        call_type=call_type,
        start_time=start_time,
    )
    child.update_environment_variables(  # pyright: ignore[reportUnknownMemberType]  # Logging.update_environment_variables is untyped upstream
        litellm_params={  # mutable-ok: update_environment_variables takes a plain dict
            **parent_params,
            "batch_parent_id": batch.id,
            "metadata": _metadata_copy(parent_params),
        },
        optional_params=_optional_params_for_body(request_body),
        model=child.model,
        custom_llm_provider=custom_llm_provider,
    )
    for secret_key in _SECRET_PARAM_KEYS:
        child.litellm_params.pop(secret_key, None)  # pyright: ignore[reportUnknownMemberType]  # Logging.litellm_params is untyped upstream

    now: Final = datetime.now()  # noqa: DTZ005  # naive to match the logging pipeline start_time
    if result is None:
        exception: Final = _BatchLineFailure(_line_error_payload(entry, custom_llm_provider))
        exception._hidden_params = _line_hidden_params(batch, custom_id, status_code)  # pyright: ignore[reportPrivateUsage]  # _hidden_params is set on the exception instance itself
        await child.async_failure_handler(
            exception=exception,
            traceback_exception="",
            start_time=start_time,
            end_time=now,
        )
        return True

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
    custom_llm_provider: str,
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
    line_provider: Final = _supported_line_provider(custom_llm_provider)
    if line_provider is None:
        verbose_logger.warning(
            "batch line-item callbacks are not supported for provider %s, skipping. batch_id=%s",
            custom_llm_provider,
            batch.id,
        )
        return 0
    emitted = 0  # rebind-ok: loop accumulator for emitted line count
    try:
        internal_credentials: Final = parent._litellm_internal_model_credentials  # pyright: ignore[reportPrivateUsage]  # declared transport attribute on Logging
        internal_mapping: Final = _as_object_mapping(internal_credentials)
        fetch_params: Final[dict[str, object] | None] = (  # mutable-ok: file fetcher requires a plain dict
            dict(internal_mapping)  # mutable-ok: the file fetcher reads credential kwargs off a plain dict
            if internal_mapping is not None
            else litellm_params
        )

        input_file_content: Final = await _fetch_managed_file_or_empty(batch.input_file_id, line_provider, fetch_params)
        requests_by_id: Final = _requests_by_custom_id(input_file_content)

        output_content: Final = await _fetch_managed_file_or_empty(batch.output_file_id, line_provider, fetch_params)
        first_row: Final = next(_output_entries(output_content), None)
        if _uses_native_vertex_output(line_provider, model_name, first_row):
            verbose_logger.warning(
                "batch line-item callbacks do not support native vertex_ai batch output rows yet, skipping. batch_id=%s",
                batch.id,
            )
            return 0
        error_content: Final = await _fetch_managed_file_or_empty(batch.error_file_id, line_provider, fetch_params)
        for content in (output_content, error_content):
            for entry in _output_entries(content):
                try:
                    emitted += await _emit_line_event(
                        entry=entry,
                        requests_by_id=requests_by_id,
                        batch=batch,
                        custom_llm_provider=line_provider,
                        parent=parent,
                        model_name=model_name,
                        model_info=model_info,
                    )
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

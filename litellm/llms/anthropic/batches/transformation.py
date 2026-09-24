import json
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from uuid import uuid4

import httpx
from httpx import Headers, Response
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import EMPTY_MAPPING
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AnthropicBatchRequestCounts(TypedDict, total=False):
    """The ``request_counts`` object of an Anthropic Message Batch."""

    processing: ReadOnly[int]
    succeeded: ReadOnly[int]
    errored: ReadOnly[int]
    canceled: ReadOnly[int]
    expired: ReadOnly[int]


class AnthropicMessageBatch(TypedDict, total=False):
    """The fields of an Anthropic Message Batch that map onto an OpenAI Batch."""

    id: ReadOnly[str]
    processing_status: ReadOnly[str]
    created_at: ReadOnly[str | None]
    ended_at: ReadOnly[str | None]
    expires_at: ReadOnly[str | None]
    cancel_initiated_at: ReadOnly[str | None]
    archived_at: ReadOnly[str | None]
    request_counts: ReadOnly[AnthropicBatchRequestCounts]


class AnthropicBatchRequest(TypedDict):
    """One entry of the ``requests`` array in an Anthropic message batch create body."""

    custom_id: ReadOnly[str]
    params: ReadOnly[dict[str, object]]


class _OpenAIBatchOutputResponse(TypedDict, total=False):
    """The ``response`` object of one OpenAI batch output JSONL line."""

    status_code: ReadOnly[int]
    request_id: ReadOnly[str | None]
    body: ReadOnly[object]


class _OpenAIBatchOutputError(TypedDict, total=False):
    """The ``error`` object of one OpenAI batch output JSONL line."""

    code: ReadOnly[object]
    message: ReadOnly[object]


class OpenAIBatchOutputLine(TypedDict, total=False):
    """One line of an OpenAI batch output JSONL file."""

    id: ReadOnly[str]
    custom_id: ReadOnly[str]
    response: ReadOnly[_OpenAIBatchOutputResponse | None]
    error: ReadOnly[_OpenAIBatchOutputError | None]


class AnthropicMessageBatchList(TypedDict, total=False):
    """The response shape of ``GET /v1/messages/batches``."""

    data: ReadOnly[list[AnthropicMessageBatch]]
    first_id: ReadOnly[str | None]
    last_id: ReadOnly[str | None]
    has_more: ReadOnly[bool]


_ANTHROPIC_MESSAGE_BATCH_ADAPTER: Final = TypeAdapter(AnthropicMessageBatch)
_ANTHROPIC_MESSAGE_BATCH_LIST_ADAPTER: Final = TypeAdapter(AnthropicMessageBatchList)
_OBJECT_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


_ANTHROPIC_PROCESSING_STATUS_TO_OPENAI_STATUS: Final[
    Mapping[
        str,
        Literal[
            "validating",
            "failed",
            "in_progress",
            "finalizing",
            "completed",
            "expired",
            "cancelling",
            "cancelled",
        ],
    ]
] = MappingProxyType(
    {
        "in_progress": "in_progress",
        "canceling": "cancelling",
        "ended": "completed",
    }
)

_ANTHROPIC_BATCH_ERROR_TYPE_TO_STATUS_CODE: Final[Mapping[str, int]] = MappingProxyType(
    {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "permission_error": 403,
        "not_found_error": 404,
        "request_too_large": 413,
        "rate_limit_error": 429,
        "api_error": 500,
        "overloaded_error": 529,
    }
)


def _parse_anthropic_timestamp(ts_str: str | None) -> int | None:
    if not ts_str:
        return None
    try:
        from datetime import datetime

        dt: Final = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, OverflowError, OSError):
        return None


def transform_anthropic_message_batch(response_data: AnthropicMessageBatch) -> LiteLLMBatch:
    """Map an Anthropic MessageBatch object onto a LiteLLMBatch in OpenAI batch shape."""
    batch_id: Final = response_data.get("id", "")
    processing_status: Final = response_data.get("processing_status", "in_progress")

    created_at: Final = _parse_anthropic_timestamp(response_data.get("created_at"))
    ended_at: Final = _parse_anthropic_timestamp(response_data.get("ended_at"))
    expires_at: Final = _parse_anthropic_timestamp(response_data.get("expires_at"))
    cancel_initiated_at: Final = _parse_anthropic_timestamp(response_data.get("cancel_initiated_at"))
    archived_at: Final = _parse_anthropic_timestamp(response_data.get("archived_at"))

    openai_status: Final = (
        "cancelled"
        if processing_status == "ended" and cancel_initiated_at is not None
        else _ANTHROPIC_PROCESSING_STATUS_TO_OPENAI_STATUS.get(processing_status, "in_progress")
    )

    request_counts_data: Final = response_data.get("request_counts", EMPTY_MAPPING)
    from openai.types.batch import BatchRequestCounts

    request_counts: Final = BatchRequestCounts(
        total=sum(
            (
                request_counts_data.get("processing", 0),
                request_counts_data.get("succeeded", 0),
                request_counts_data.get("errored", 0),
                request_counts_data.get("canceled", 0),
                request_counts_data.get("expired", 0),
            )
        ),
        completed=request_counts_data.get("succeeded", 0),
        failed=request_counts_data.get("errored", 0),
    )

    return LiteLLMBatch(
        id=batch_id,
        object="batch",
        endpoint="/v1/messages",
        errors=None,
        input_file_id="None",
        completion_window="24h",
        status=openai_status,
        output_file_id=batch_id,
        error_file_id=None,
        created_at=created_at or int(time.time()),
        in_progress_at=created_at if openai_status == "in_progress" else None,
        expires_at=expires_at,
        finalizing_at=None,
        completed_at=ended_at if openai_status == "completed" else None,
        failed_at=None,
        expired_at=archived_at if archived_at else None,
        cancelling_at=(cancel_initiated_at if openai_status in ("cancelling", "cancelled") else None),
        cancelled_at=(ended_at if openai_status == "cancelled" else None),
        request_counts=request_counts,
        metadata={},  # mutable-ok: LiteLLMBatch wants a fresh dict per instance
    )


def transform_openai_batch_lines_to_anthropic_requests(
    lines: Sequence[Mapping[str, object]], model: str
) -> tuple[AnthropicBatchRequest, ...]:
    """Translate OpenAI batch input JSONL lines into Anthropic message batch requests.

    ``model`` is the bare Anthropic model name (any ``anthropic/`` prefix is stripped by
    the caller); each line's ``body.model`` is ignored in favor of it.
    """
    from ..chat.transformation import AnthropicConfig

    anthropic_config: Final = AnthropicConfig()

    def to_anthropic_request(line: Mapping[str, object]) -> AnthropicBatchRequest:
        custom_id: Final = line.get("custom_id")
        body: Final = line.get("body")
        if not custom_id:
            raise ValueError(f"OpenAI batch input line is missing 'custom_id': {line}")
        if not isinstance(body, Mapping):
            raise TypeError(f"OpenAI batch input line for custom_id={custom_id} is missing 'body'")
        messages: Final = body.get("messages")
        if not isinstance(messages, list):
            raise TypeError(f"OpenAI batch input line for custom_id={custom_id} is missing 'body.messages'")
        non_default_params: Final = {  # mutable-ok: map_openai_params takes a plain dict
            str(k): v for k, v in body.items() if k not in ("model", "messages")
        }
        optional_params: Final = _OBJECT_DICT_ADAPTER.validate_python(
            anthropic_config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},  # mutable-ok: map_openai_params writes into this per line
                model=model,
                drop_params=True,
            ),
        )
        params: Final = _OBJECT_DICT_ADAPTER.validate_python(
            anthropic_config.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params={},  # mutable-ok: transform_request may write into this per line
                headers={},  # mutable-ok: transform_request writes headers entries per line
            ),
        )
        params["model"] = model
        return AnthropicBatchRequest(custom_id=str(custom_id), params=params)

    return tuple(to_anthropic_request(line) for line in lines)


def transform_anthropic_batch_result_line(
    line: Mapping[str, object], raw_response: httpx.Response
) -> OpenAIBatchOutputLine:
    """Translate one Anthropic message batch results JSONL line into OpenAI batch output shape."""
    from ..chat.transformation import AnthropicConfig

    raw_custom_id: Final = line.get("custom_id")
    custom_id: Final = raw_custom_id if isinstance(raw_custom_id, str) else ""
    raw_result: Final = line.get("result")
    result: Final = raw_result if isinstance(raw_result, Mapping) else EMPTY_MAPPING
    result_type: Final = result.get("type")

    if result_type == "succeeded":
        raw_message: Final = result.get("message")
        message: Final = raw_message if isinstance(raw_message, Mapping) else EMPTY_MAPPING
        transformed_response: Final = AnthropicConfig().transform_parsed_response(
            completion_response=_OBJECT_DICT_ADAPTER.validate_python(message),
            raw_response=raw_response,
            model_response=ModelResponse(),
        )
        request_id: Final = message.get("id")
        return OpenAIBatchOutputLine(
            id=f"batch_req_{uuid4().hex}",
            custom_id=custom_id,
            response=_OpenAIBatchOutputResponse(
                status_code=200,
                request_id=request_id if isinstance(request_id, str) else None,
                body=_OBJECT_DICT_ADAPTER.validate_python(transformed_response.model_dump(exclude_none=True)),
            ),
            error=None,
        )
    if result_type == "errored":
        raw_error: Final = result.get("error")
        error_candidate: Final = raw_error if isinstance(raw_error, Mapping) else EMPTY_MAPPING
        nested_error: Final = error_candidate.get("error")
        error: Final = nested_error if isinstance(nested_error, Mapping) else error_candidate
        error_type: Final = error.get("type")
        status_code: Final = _ANTHROPIC_BATCH_ERROR_TYPE_TO_STATUS_CODE.get(
            error_type if isinstance(error_type, str) else "", 500
        )
        return OpenAIBatchOutputLine(
            id=f"batch_req_{uuid4().hex}",
            custom_id=custom_id,
            response=_OpenAIBatchOutputResponse(
                status_code=status_code, request_id=None, body=_OBJECT_DICT_ADAPTER.validate_python(error)
            ),
            error=_OpenAIBatchOutputError(code=error_type, message=error.get("message")),
        )
    if result_type in ("canceled", "expired"):
        failure_message: Final = (
            "The request was canceled." if result_type == "canceled" else "The request expired before it was processed."
        )
        return OpenAIBatchOutputLine(
            id=f"batch_req_{uuid4().hex}",
            custom_id=custom_id,
            response=None,
            error=_OpenAIBatchOutputError(code=result_type, message=failure_message),
        )
    raise ValueError(f"Unknown Anthropic batch result type: {result_type}")


class AnthropicBatchesConfig(BaseBatchesConfig):
    def __init__(self):
        from ..chat.transformation import AnthropicConfig
        from ..common_utils import AnthropicModelInfo

        self.anthropic_chat_config = AnthropicConfig()  # initialize once
        self.anthropic_model_info = AnthropicModelInfo()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        """Return the LLM provider type for this configuration."""
        return LlmProviders.ANTHROPIC

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """Validate and prepare environment-specific headers and parameters."""
        if api_base is None and isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")
        auth_header: Final = self.anthropic_model_info.get_auth_header(api_key, api_base)
        if auth_header is None:
            raise ValueError(
                "Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params"
            )
        _headers: Final = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        _headers.update(auth_header)
        # Add beta header for message batches
        if "anthropic-beta" not in headers:
            headers["anthropic-beta"] = "message-batches-2024-09-24"
        headers.update(_headers)
        return headers

    def get_complete_batch_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        data: CreateBatchRequest,
    ) -> str:
        """Get the complete URL for batch creation request."""
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        if not api_base.endswith("/v1/messages/batches"):
            api_base = f"{api_base.rstrip('/')}/v1/messages/batches"
        return api_base

    def transform_create_batch_request(
        self,
        model: str,
        create_batch_data: CreateBatchRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> bytes | str | dict[str, object]:
        """
        Transform the batch creation request to Anthropic format.

        Anthropic has no batch input file: the requests go inline on
        ``extra_body["requests"]`` as ``[{"custom_id": ..., "params": {...}}]``.
        """
        raw_extra_body: Final = create_batch_data.get("extra_body")
        extra_body: Final = raw_extra_body if isinstance(raw_extra_body, Mapping) else EMPTY_MAPPING
        requests: Final = extra_body.get("requests")
        if not isinstance(requests, list) or len(requests) == 0:
            raise self.get_error_class(
                error_message=(
                    "Anthropic message batches take their requests inline: pass "
                    "extra_body={'requests': [{'custom_id': ..., 'params': {...}}]} "
                    "(the proxy builds this from a LiteLLM managed batch input file)"
                ),
                status_code=400,
                headers=Headers(),
            )
        return {**extra_body, "requests": list(requests)}  # mutable-ok: base contract returns a plain JSON dict payload

    def transform_create_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LoggingClass,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """Transform Anthropic MessageBatch creation response to LiteLLM format."""
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        try:
            response_data: Final[AnthropicMessageBatch] = _ANTHROPIC_MESSAGE_BATCH_ADAPTER.validate_python(
                raw_response.json()
            )
        except ValueError as e:
            raise ValueError(f"Failed to parse Anthropic batch response: {e}")
        return transform_anthropic_message_batch(response_data)

    def transform_stored_batch_input(
        self,
        model: str,
        endpoint: str,
        lines: Sequence[Mapping[str, object]],
        extra_body: Mapping[str, str] | None,
    ) -> Mapping[str, object]:
        """Build the ``extra_body`` for ``create_batch`` from parsed OpenAI batch input lines."""
        if endpoint != "/v1/chat/completions":
            raise ValueError("Anthropic message batches only support the /v1/chat/completions endpoint")
        return MappingProxyType(
            {
                **(extra_body or EMPTY_MAPPING),
                "requests": list(transform_openai_batch_lines_to_anthropic_requests(lines=lines, model=model)),
            }
        )

    def get_list_batches_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
        litellm_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
        after: str | None,
        limit: int | None,
    ) -> str:
        """Get the URL for listing Anthropic message batches, with pagination params."""
        resolved_api_base: Final = self.anthropic_model_info.get_api_base(api_base) or "https://api.anthropic.com"
        params: Final = tuple(
            param
            for param in (
                ("limit", str(limit)) if limit is not None else None,
                ("after_id", after) if after else None,
            )
            if param is not None
        )
        return str(httpx.URL(f"{resolved_api_base.rstrip('/')}/v1/messages/batches", params=params))

    def transform_list_batches_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LoggingClass | None,
        litellm_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
    ) -> dict[str, object]:  # mutable-ok: list responses mirror the dict payload the proxy serializes
        """Transform the Anthropic batch list response into the OpenAI list shape."""
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        response_json: Final = _ANTHROPIC_MESSAGE_BATCH_LIST_ADAPTER.validate_python(raw_response.json())
        data: Final = tuple(transform_anthropic_message_batch(b) for b in response_json.get("data", ()))
        return {  # mutable-ok: list responses mirror the dict payload the proxy serializes
            "object": "list",
            "data": data,
            "first_id": response_json.get("first_id") or (data[0].id if data else None),
            "last_id": response_json.get("last_id") or (data[-1].id if data else None),
            "has_more": bool(response_json.get("has_more", False)),
        }

    def get_cancel_batch_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        batch_id: str,
        optional_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
        litellm_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
    ) -> str:
        """Get the URL for cancelling an Anthropic message batch."""
        resolved_api_base: Final = self.anthropic_model_info.get_api_base(api_base) or "https://api.anthropic.com"
        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        return f"{resolved_api_base.rstrip('/')}/v1/messages/batches/{encoded_batch_id}/cancel"

    def transform_cancel_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LoggingClass | None,
        litellm_params: dict,  # mutable-ok: batch-config signature contract is a plain dict
    ) -> LiteLLMBatch:
        """Transform Anthropic MessageBatch cancel response to LiteLLM format."""
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        try:
            response_data: Final[AnthropicMessageBatch] = _ANTHROPIC_MESSAGE_BATCH_ADAPTER.validate_python(
                raw_response.json()
            )
        except ValueError as e:
            raise ValueError(f"Failed to parse Anthropic batch response: {e}")
        return transform_anthropic_message_batch(response_data)

    def get_retrieve_batch_url(
        self,
        api_base: str | None,
        batch_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for batch retrieval request.

        Args:
            api_base: Base API URL (optional, will use default if not provided)
            batch_id: Batch ID to retrieve
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for Anthropic batch retrieval: {api_base}/v1/messages/batches/{batch_id}
        """
        api_base = api_base or self.anthropic_model_info.get_api_base(api_base)
        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        return f"{api_base.rstrip('/')}/v1/messages/batches/{encoded_batch_id}"

    def transform_retrieve_batch_request(
        self,
        batch_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> bytes | str | dict[str, object]:
        """
        Transform batch retrieval request for Anthropic.

        For Anthropic, the URL is constructed by get_retrieve_batch_url(),
        so this method returns an empty dict (no additional request params needed).
        """
        # No additional request params needed - URL is handled by get_retrieve_batch_url
        return {}

    def transform_retrieve_batch_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LoggingClass,
        litellm_params: dict,
    ) -> LiteLLMBatch:
        """Transform Anthropic MessageBatch retrieval response to LiteLLM format."""
        try:
            response_data: Final[AnthropicMessageBatch] = raw_response.json()
        except ValueError as e:
            raise ValueError(f"Failed to parse Anthropic batch response: {e}")
        return transform_anthropic_message_batch(response_data)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> "BaseLLMException":
        """Get the appropriate error class for Anthropic."""
        from ..common_utils import AnthropicError

        # Convert Dict to Headers if needed
        if isinstance(headers, dict):
            headers_obj: Headers | None = Headers(headers)
        else:
            headers_obj = headers if isinstance(headers, Headers) else None

        return AnthropicError(status_code=status_code, message=error_message, headers=headers_obj)

    def transform_response(
        self,
        model: str,
        raw_response: Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        from litellm.cost_calculator import BaseTokenUsageProcessor
        from litellm.types.utils import Usage

        response_text: Final = raw_response.text.strip()
        all_usage: Final[list[Usage]] = []

        try:
            # Split by newlines and try to parse each line as JSON
            lines: Final = response_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    response_json: Mapping[str, Mapping[str, dict[str, object]]] = json.loads(line)
                    # Update model_response with the parsed JSON
                    completion_response = response_json["result"]["message"]
                    transformed_response = self.anthropic_chat_config.transform_parsed_response(
                        completion_response=completion_response,
                        raw_response=raw_response,
                        model_response=model_response,
                    )

                    transformed_response_usage = getattr(transformed_response, "usage", None)
                    if transformed_response_usage:
                        all_usage.append(cast(Usage, transformed_response_usage))
                except json.JSONDecodeError:
                    continue

            ## SUM ALL USAGE
            combined_usage: Final = BaseTokenUsageProcessor.combine_usage_objects(all_usage)
            setattr(model_response, "usage", combined_usage)

            return model_response
        except Exception as e:
            raise e

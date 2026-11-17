from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from typing_extensions import Never

from litellm.exceptions import AuthenticationError, UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import bigmodel_business_error, bigmodel_json_body, get_bigmodel_api_key, get_bigmodel_url

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SUPPORTED_PARAMS: Final = (
    "input",
    "model",
    "instructions",
    "stream",
    "temperature",
    "top_p",
    "max_output_tokens",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "include",
    "reasoning",
    "text",
    "prompt_cache_key",
    "previous_response_id",
    "store",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
)


_STATUS_BY_BUSINESS_CODE: Final = MappingProxyType(
    {
        **dict.fromkeys((1000, 1001, 1003, 1005), 401),
        **dict.fromkeys((1210, 1211, 1212, 1213, 1214, 1215, 1221, 1222, 1261, 1301), 400),
        1220: 403,
        **dict.fromkeys(
            (1113, 1302, 1305, 1308, 1309, 1310, 1311, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321),
            429,
        ),
    }
)


class _RequestedTools(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    tools: JsonValue = None
    tool_choice: JsonValue = None
    parallel_tool_calls: JsonValue = None


def _requested_tools(logging_obj: Logging) -> _RequestedTools:
    try:
        return _RequestedTools.model_validate(logging_obj.model_call_details.get("optional_params") or {})
    except ValidationError:
        return _RequestedTools()


def _envelope_status_code(code: JsonValue) -> int:
    if not isinstance(code, int) or isinstance(code, bool):
        return 500
    if 400 <= code <= 599:
        return code
    return _STATUS_BY_BUSINESS_CODE.get(code, 500)


def _normalize_output_item(item: JsonValue) -> JsonValue:
    if not isinstance(item, dict) or item.get("type") not in ("message", "reasoning"):
        return item
    content: Final = item.get("content")
    normalized: Final[JsonValue] = {"summary": [], **item} if item["type"] == "reasoning" else item
    if not isinstance(content, dict):
        return normalized
    return {**normalized, "content": [content]}


def _normalize_response(
    body: Mapping[str, JsonValue],
    requested: _RequestedTools,
) -> dict[str, JsonValue]:  # mutable-ok: JSON serialization boundary
    usage: Final = body.get("usage")
    output: Final = body.get("output")
    tools: Final = body.get("tools")
    tool_choice: Final = body.get("tool_choice")
    parallel_tool_calls: Final = body.get("parallel_tool_calls")
    input_tokens: Final = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens: Final = usage.get("output_tokens") if isinstance(usage, dict) else None
    normalized_usage: Final = (
        {
            "total_tokens": input_tokens + output_tokens,
            **usage,
        }
        if isinstance(usage, dict) and isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else usage
    )
    normalized: Final[JsonValue] = {
        **body,
        "parallel_tool_calls": (
            requested.parallel_tool_calls is not False if parallel_tool_calls is None else parallel_tool_calls
        ),
        "tools": (requested.tools or []) if tools is None else tools,
        "tool_choice": (requested.tool_choice or "auto") if tool_choice is None else tool_choice,
        **({"usage": normalized_usage} if "usage" in body else {}),
        **({"output": [_normalize_output_item(item) for item in output]} if isinstance(output, list) else {}),
    }
    return normalized


def _normalize_stream_part(part: JsonValue) -> JsonValue:
    if not isinstance(part, dict):
        return part
    if part.get("type") == "output_text":
        return {
            "annotations": [],
            "logprobs": None,
            **part,
        }
    if part.get("type") == "reasoning_text":
        return {"reasoning": part.get("text", ""), **part}
    return part


class BigModelResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BIGMODEL

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

    def should_fake_stream(
        self, model: str | None, stream: bool | None, custom_llm_provider: str | None = None
    ) -> bool:
        return False

    def may_answer_stream_with_json(self) -> bool:
        return True

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited API requires a list
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: request pipeline adds parameters to the returned body
        return {key: value for key, value in response_api_optional_params.items() if key in _SUPPORTED_PARAMS}

    def validate_environment(
        self, headers: Mapping[str, str], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, str]:  # mutable-ok: HTTP handler merges forwarded headers
        resolved_key: Final = get_bigmodel_api_key(litellm_params.api_key if litellm_params else None)
        if resolved_key is None:
            raise AuthenticationError(
                message="Set BIGMODEL_API_KEY or pass api_key", llm_provider="bigmodel", model=model
            )
        return {
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return get_bigmodel_url(api_base, "responses")

    def transform_response_api_response(
        self, model: str, raw_response: httpx.Response, logging_obj: Logging
    ) -> ResponsesAPIResponse:
        body: Final = bigmodel_json_body(raw_response)
        if body.get("success") is False:
            message: Final = str(body.get("msg") or raw_response.text)
            raise bigmodel_business_error(body.get("code"), message, model, None) or BaseLLMException(
                message=message,
                status_code=_envelope_status_code(body.get("code")),
                headers=raw_response.headers,
            )
        normalized: Final = _normalize_response(body, _requested_tools(logging_obj))
        normalized_response: Final = httpx.Response(
            status_code=raw_response.status_code,
            headers={
                key: value
                for key, value in raw_response.headers.items()
                if key.lower() not in ("content-encoding", "content-length")
            },
            json=normalized,
            request=raw_response.request,
        )
        return super().transform_response_api_response(model, normalized_response, logging_obj)

    def transform_get_response_api_response(
        self, raw_response: httpx.Response, logging_obj: Logging
    ) -> ResponsesAPIResponse:
        return self.transform_response_api_response("", raw_response, logging_obj)

    def transform_streaming_response(
        self, model: str, parsed_chunk: Mapping[str, object], logging_obj: Logging
    ) -> ResponsesAPIStreamingResponse:
        chunk: Final = _JSON_OBJECT.validate_python(parsed_chunk)
        response: Final = chunk.get("response")
        item: Final = chunk.get("item")
        normalized: Final[JsonValue] = {
            **chunk,
            **(
                {"response": _normalize_response(response, _requested_tools(logging_obj))}
                if isinstance(response, dict)
                else {}
            ),
            **({"item": _normalize_output_item(item)} if "item" in chunk else {}),
        }
        event_model: Final = self.get_event_model_class(str(normalized.get("type")))
        match chunk.get("type"):
            case "response.output_text.done":
                return event_model.model_validate({"logprobs": [], **normalized})
            case "response.content_part.done":
                return event_model.model_validate({**normalized, "part": _normalize_stream_part(chunk.get("part"))})
            case "error":
                error: Final = chunk.get("error")
                error_fields: Final = error if isinstance(error, dict) else chunk
                normalized_error: Final[JsonValue] = {
                    **normalized,
                    "error": {"type": "api_error", **error_fields, "code": error_fields.get("code") or "unknown_error"},
                }
                return event_model.model_validate(normalized_error)
            case _:
                return event_model.model_validate(normalized)

    def transform_cancel_response_api_request(
        self, response_id: str, api_base: str, litellm_params: GenericLiteLLMParams, headers: Mapping[str, str]
    ) -> Never:
        raise UnsupportedParamsError(message="BigModel does not support cancelling responses", status_code=400)

    def transform_compact_response_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: Mapping[str, object],
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> Never:
        raise UnsupportedParamsError(message="BigModel does not support compacting responses", status_code=400)

from collections.abc import Mapping
from typing import Any, Final

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.sse_output_recovery import (
    parse_sse_json_chunk,
    record_output_item_chunk,
    record_output_text_chunk,
)
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)

# Codex >= 0.144 sends this header for models served over the "Responses
# Lite" transport (gpt-5.6-* at the time of writing). When the header is
# present the backend rejects the request unless `parallel_tool_calls` is
# exactly false and `reasoning.context` is "all_turns":
#   "X-OpenAI-Internal-Codex-Responses-Lite requires `parallel_tool_calls`
#    to be false." (param=parallel_tool_calls, code=unsupported_value)
CODEX_RESPONSES_LITE_HEADER: Final[str] = "x-openai-internal-codex-responses-lite"
_FALSY_HEADER_VALUES: Final[frozenset[str]] = frozenset({"", "0", "false", "no"})


def is_codex_responses_lite_request(headers: Mapping[str, object] | None) -> bool:
    """True when the outbound headers carry the Codex Responses-Lite marker."""
    if not headers:
        return False
    for key, value in headers.items():
        if key.lower() == CODEX_RESPONSES_LITE_HEADER:
            return str(value).strip().lower() not in _FALSY_HEADER_VALUES
    return False


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        try:
            access_token: Final = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id: Final = self.authenticator.get_account_id()
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(access_token, account_id, session_id)
        return {**default_headers, **headers}

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request: Final = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        base_instructions: Final = get_chatgpt_default_instructions()
        existing_instructions: Final = request.get("instructions")
        if existing_instructions:
            if base_instructions not in existing_instructions:
                request["instructions"] = f"{base_instructions}\n\n{existing_instructions}"
        else:
            request["instructions"] = base_instructions
        request["store"] = False
        request["stream"] = True
        include: Final = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys: Final = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "previous_response_id",
            "truncation",
        }

        filtered: Final[dict[str, object]] = {  # mutable-ok: outgoing JSON request body
            k: v for k, v in request.items() if k in allowed_keys
        }

        if is_codex_responses_lite_request(headers):
            # The Responses-Lite backend hard-rejects any other combination,
            # so normalize even when the caller omitted these fields (codex
            # 0.144.x can omit the reasoning object on a metadata race).
            filtered["parallel_tool_calls"] = False
            raw_reasoning: Final = filtered.get("reasoning")
            reasoning_items: Final = raw_reasoning.items() if isinstance(raw_reasoning, dict) else ()
            reasoning: dict[str, object] = {  # mutable-ok: rebuilt to force the required context key
                reasoning_key: reasoning_value
                for reasoning_key, reasoning_value in reasoning_items
                if isinstance(reasoning_key, str)
            }
            reasoning["context"] = "all_turns"
            filtered["reasoning"] = reasoning

        return filtered

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        body_text: Final = raw_response.text or ""
        if not self._should_parse_as_sse(raw_response=raw_response, body_text=body_text):
            return super().transform_response_api_response(
                model=model,
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response, error_message = self._extract_completed_response_from_sse(body_text=body_text)
        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        self._attach_response_headers(completed_response=completed_response, raw_response=raw_response)
        return completed_response

    def _should_parse_as_sse(self, raw_response: Any, body_text: str) -> bool:
        content_type: Final = (raw_response.headers or {}).get("content-type", "")
        if "text/event-stream" in content_type.lower():
            return True
        trimmed_body: Final = body_text.lstrip()
        return bool(
            trimmed_body.startswith("event:")
            or trimmed_body.startswith("data:")
            or "\nevent:" in body_text
            or "\ndata:" in body_text
        )

    def _extract_completed_response_from_sse(self, body_text: str) -> tuple[ResponsesAPIResponse | None, str | None]:
        completed_response = None
        error_message = None
        streamed_output_items: Final[dict[int, dict]] = {}
        text_only_output_items: Final[dict[int, dict]] = {}
        for chunk in body_text.splitlines():
            parsed_chunk = parse_sse_json_chunk(chunk)
            if parsed_chunk is None:
                continue

            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
                record_output_item_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=streamed_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
                record_output_text_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=streamed_output_items,
                    text_only_items=text_only_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                # Real OUTPUT_ITEM_DONE events take precedence at any given
                # output_index, but text-only items at indices without a
                # matching OUTPUT_ITEM_DONE must still be preserved (e.g.
                # providers that emit only OUTPUT_TEXT_DONE for some indices).
                merged_items: dict[int, dict] = {**text_only_output_items}
                merged_items.update(streamed_output_items)
                completed_response = self._build_completed_response_from_chunk(
                    parsed_chunk=parsed_chunk,
                    streamed_output_items=merged_items,
                )
                break

            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                extracted_error = self._extract_error_message(parsed_chunk)
                if extracted_error is not None:
                    error_message = extracted_error

        return completed_response, error_message

    def _build_completed_response_from_chunk(
        self, parsed_chunk: dict[str, Any], streamed_output_items: dict[int, dict]
    ) -> ResponsesAPIResponse | None:
        response_payload = parsed_chunk.get("response")
        if not isinstance(response_payload, dict):
            return None
        response_payload = dict(response_payload)
        if not response_payload.get("output") and streamed_output_items:
            response_payload["output"] = [item for _, item in sorted(streamed_output_items.items())]
        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(response_payload["created_at"])
        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    def _extract_error_message(self, parsed_chunk: dict[str, Any]) -> str | None:
        error_obj: Final = parsed_chunk.get("error") or (parsed_chunk.get("response") or {}).get("error")
        if error_obj is None:
            return None
        if isinstance(error_obj, dict):
            return error_obj.get("message") or str(error_obj)
        return str(error_obj)

    def _attach_response_headers(
        self,
        completed_response: ResponsesAPIResponse,
        raw_response: Any,
    ) -> None:
        raw_headers: Final = dict(raw_response.headers)
        processed_headers: Final = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False

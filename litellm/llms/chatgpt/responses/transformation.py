import json
from typing import Any, Dict, Optional

from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import CustomStreamWrapper

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
)


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
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        try:
            access_token = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token, account_id, session_id
        )
        return {**default_headers, **headers}

    def transform_responses_api_request(
        self,
        model: str,
        input: Any,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        base_instructions = get_chatgpt_default_instructions()
        existing_instructions = request.get("instructions")
        if existing_instructions:
            if base_instructions not in existing_instructions:
                request["instructions"] = (
                    f"{base_instructions}\n\n{existing_instructions}"
                )
        else:
            request["instructions"] = base_instructions
        request["store"] = False
        request["stream"] = True
        include = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        allowed_keys = {
            "model",
            "input",
            "instructions",
            "stream",
            "store",
            "include",
            "tools",
            "tool_choice",
            "reasoning",
            "previous_response_id",
            "truncation",
        }

        return {k: v for k, v in request.items() if k in allowed_keys}

    def transform_response_api_response(
        self,
        model: str,
        raw_response: Any,
        logging_obj: Any,
    ):
        body_text = raw_response.text or ""
        if not self._should_parse_as_sse(
            raw_response=raw_response, body_text=body_text
        ):
            return super().transform_response_api_response(
                model=model,
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response, error_message = self._extract_completed_response_from_sse(
            body_text=body_text
        )
        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        self._attach_response_headers(
            completed_response=completed_response, raw_response=raw_response
        )
        return completed_response

    def _should_parse_as_sse(self, raw_response: Any, body_text: str) -> bool:
        content_type = (raw_response.headers or {}).get("content-type", "")
        if "text/event-stream" in content_type.lower():
            return True
        trimmed_body = body_text.lstrip()
        return bool(
            trimmed_body.startswith("event:")
            or trimmed_body.startswith("data:")
            or "\nevent:" in body_text
            or "\ndata:" in body_text
        )

    def _extract_completed_response_from_sse(
        self, body_text: str
    ) -> tuple[Optional[ResponsesAPIResponse], Optional[str]]:
        completed_response = None
        error_message = None
        streamed_output_items: Dict[int, dict] = {}
        for chunk in body_text.splitlines():
            parsed_chunk = self._parse_sse_json_chunk(chunk)
            if parsed_chunk is None:
                continue
            if parsed_chunk == STREAM_SSE_DONE_STRING:
                break

            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
                self._record_output_item_chunk(
                    parsed_chunk=parsed_chunk,
                    streamed_output_items=streamed_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                completed_response = self._build_completed_response_from_chunk(
                    parsed_chunk=parsed_chunk,
                    streamed_output_items=streamed_output_items,
                )
                break

            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_message = self._extract_error_message(parsed_chunk)

        return completed_response, error_message

    def _parse_sse_json_chunk(self, chunk: str) -> Optional[Any]:
        stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
        if not stripped_chunk:
            return None
        stripped_chunk = stripped_chunk.strip()
        if not stripped_chunk:
            return None
        if stripped_chunk == STREAM_SSE_DONE_STRING:
            return STREAM_SSE_DONE_STRING
        try:
            parsed_chunk = json.loads(stripped_chunk)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed_chunk, dict):
            return None
        return parsed_chunk

    def _record_output_item_chunk(
        self, parsed_chunk: Dict[str, Any], streamed_output_items: Dict[int, dict]
    ) -> None:
        item = parsed_chunk.get("item")
        output_index = parsed_chunk.get("output_index")
        if not isinstance(item, dict):
            return
        try:
            if output_index is None:
                raise ValueError("missing output_index")
            index = int(output_index)
        except (TypeError, ValueError):
            index = len(streamed_output_items)
        streamed_output_items[index] = item

    def _build_completed_response_from_chunk(
        self, parsed_chunk: Dict[str, Any], streamed_output_items: Dict[int, dict]
    ) -> Optional[ResponsesAPIResponse]:
        response_payload = parsed_chunk.get("response")
        if not isinstance(response_payload, dict):
            return None
        response_payload = dict(response_payload)
        if not response_payload.get("output") and streamed_output_items:
            response_payload["output"] = [
                item for _, item in sorted(streamed_output_items.items())
            ]
        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(
                response_payload["created_at"]
            )
        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    def _extract_error_message(self, parsed_chunk: Dict[str, Any]) -> Optional[str]:
        error_obj = parsed_chunk.get("error") or (
            parsed_chunk.get("response") or {}
        ).get("error")
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
        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False

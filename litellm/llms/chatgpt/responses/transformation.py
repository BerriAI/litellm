import json
from typing import Any, Optional

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
        if self._should_use_openai_json_response_parser(raw_response, body_text):
            return super().transform_response_api_response(
                model=model,
                raw_response=raw_response,
                logging_obj=logging_obj,
            )
        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        parsed_sse = self._parse_chatgpt_sse_response(body_text)
        completed_response = parsed_sse["completed_response"]
        completed_response_payload = parsed_sse["completed_response_payload"]
        error_message = parsed_sse["error_message"]
        output_text_parts = parsed_sse["output_text_parts"]

        completed_response = self._synthesize_empty_completed_output_response(
            completed_response=completed_response,
            completed_response_payload=completed_response_payload,
            output_text_parts=output_text_parts,
        )

        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers
        return completed_response

    @staticmethod
    def _should_use_openai_json_response_parser(raw_response: Any, body_text: str) -> bool:
        content_type = (raw_response.headers or {}).get("content-type", "")
        if "text/event-stream" in content_type.lower():
            return False

        trimmed_body = body_text.lstrip()
        return not (
            trimmed_body.startswith("event:")
            or trimmed_body.startswith("data:")
            or "\nevent:" in body_text
            or "\ndata:" in body_text
        )

    @staticmethod
    def _build_response_api_response(response_payload: dict) -> ResponsesAPIResponse:
        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(
                response_payload["created_at"]
            )
        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    @classmethod
    def _parse_chatgpt_sse_response(cls, body_text: str) -> dict:
        completed_response = None
        completed_response_payload = None
        error_message = None
        output_text_parts = []

        for chunk in body_text.splitlines():
            stripped_chunk = CustomStreamWrapper._strip_sse_data_from_chunk(chunk)
            if not stripped_chunk:
                continue
            stripped_chunk = stripped_chunk.strip()
            if not stripped_chunk:
                continue
            if stripped_chunk == STREAM_SSE_DONE_STRING:
                break

            try:
                parsed_chunk = json.loads(stripped_chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_chunk, dict):
                continue

            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
                content_part = parsed_chunk.get("delta", None)
                if isinstance(content_part, str) and content_part:
                    output_text_parts.append(content_part)
                continue

            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    completed_response_payload = dict(response_payload)
                    completed_response = cls._build_response_api_response(
                        completed_response_payload
                    )
                break

            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                error_obj = parsed_chunk.get("error") or (
                    parsed_chunk.get("response") or {}
                ).get("error")
                if error_obj is not None:
                    if isinstance(error_obj, dict):
                        error_message = error_obj.get("message") or str(error_obj)
                    else:
                        error_message = str(error_obj)

        return {
            "completed_response": completed_response,
            "completed_response_payload": completed_response_payload,
            "error_message": error_message,
            "output_text_parts": output_text_parts,
        }

    @classmethod
    def _synthesize_empty_completed_output_response(
        cls,
        completed_response: Optional[ResponsesAPIResponse],
        completed_response_payload: Optional[dict],
        output_text_parts: list[str],
    ) -> Optional[ResponsesAPIResponse]:
        if (
            completed_response_payload is None
            or completed_response_payload.get("output")
            or len(output_text_parts) == 0
        ):
            return completed_response

        completed_response_payload["output"] = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "".join(output_text_parts),
                    }
                ],
            }
        ]
        return cls._build_response_api_response(completed_response_payload)

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

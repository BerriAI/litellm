import json
from typing import Any, Optional

from litellm.exceptions import AuthenticationError
from litellm.constants import STREAM_SSE_DONE_STRING
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
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
        content_type = (raw_response.headers or {}).get("content-type", "")
        body_text = raw_response.text or ""
        if "text/event-stream" not in content_type.lower():
            trimmed_body = body_text.lstrip()
            if not (
                trimmed_body.startswith("event:")
                or trimmed_body.startswith("data:")
                or "\nevent:" in body_text
                or "\ndata:" in body_text
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

        completed_response = None
        error_message = None
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
            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                response_payload = parsed_chunk.get("response")
                if isinstance(response_payload, dict):
                    response_payload = dict(response_payload)
                    if "created_at" in response_payload:
                        response_payload["created_at"] = _safe_convert_created_field(
                            response_payload["created_at"]
                        )
                    try:
                        completed_response = ResponsesAPIResponse(**response_payload)
                    except Exception:
                        completed_response = ResponsesAPIResponse.model_construct(
                            **response_payload
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

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

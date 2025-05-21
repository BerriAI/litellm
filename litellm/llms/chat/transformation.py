"""
Chat provider mapping chat completion requests to /responses API and back to chat format
"""
import httpx
from typing import Any, Dict, List, Optional, Union

from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response import Response as ResponsesAPIResponse
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.types.utils import ModelResponse, Choices, Usage
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS


class ChatConfig(BaseConfig):
    """
    Provider config for chat that uses the /responses API under the hood.
    Transforms chat completion requests into Responses API requests and vice versa.
    """
    def get_supported_openai_params(self, model: str) -> List[str]:  # noqa: U100
        # Support standard OpenAI chat parameters
        return OPENAI_CHAT_COMPLETION_PARAMS  # type: ignore

    def validate_environment(
        self,
        headers: Dict[str, Any],
        model: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # No additional env validation for responses proxy
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        **kwargs,
    ) -> str:
        if not api_base:
            raise ValueError("api_base is required for chat via responses API")
        return api_base.rstrip("/") + "/responses"

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        optional_params: Dict[str, Any],
        litellm_params: Dict[str, Any],  # noqa: U100
        headers: Dict[str, Any],  # noqa: U100
    ) -> Dict[str, Any]:
        # Build Responses API request
        # Convert chat messages to Responses API input
        input_items: List[Any] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            # wrap text content
            input_items.append({
                "type": "message",
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        # start with required fields
        data: Dict[str, Any] = {"model": model, "input": input_items}
        # map optional params: rename max_tokens or max_completion_tokens -> max_output_tokens
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                data["max_output_tokens"] = value
            else:
                data[key] = value
        return dict(ResponseCreateParamsBase(**data))

    def transform_response(
        self,
        model: str,  # noqa: U100
        raw_response: httpx.Response,
        model_response: ModelResponse,  # noqa: U100
        logging_obj: Any,  # noqa: U100
        request_data: Dict[str, Any],  # noqa: U100
        messages: List[Any],  # noqa: U100
        optional_params: Dict[str, Any],  # noqa: U100
        litellm_params: Dict[str, Any],  # noqa: U100
        encoding: Any,  # noqa: U100
        api_key: Optional[str] = None,  # noqa: U100
        json_mode: Optional[bool] = None,  # noqa: U100
    ) -> ModelResponse:
        # Parse Responses API response and convert to chat ModelResponse
        try:
            resp_json = raw_response.json()
        except Exception:
            raise httpx.HTTPError(f"Invalid JSON from responses API: {raw_response.text}")
        resp_obj = ResponsesAPIResponse(**resp_json)
        # transform output items to chat messages
        chat_messages = LiteLLMCompletionResponsesConfig._transform_responses_api_outputs_to_chat_completion_messages(resp_obj)
        # build choices: each message as separate choice
        choices: List[Dict[str, Any]] = []
        for idx, msg in enumerate(chat_messages):
            choices.append({"index": idx, "message": msg, "finish_reason": getattr(resp_obj, "status", "completed")})
        # build usage
        usage = Usage(
            prompt_tokens=getattr(resp_obj.usage, "input_tokens", 0),
            completion_tokens=getattr(resp_obj.usage, "output_tokens", 0),
            total_tokens=getattr(resp_obj.usage, "total_tokens", 0),
        )
        return ModelResponse(
            id=resp_obj.id,
            choices=choices,
            created=resp_obj.created_at,
            model=resp_obj.model,
            usage=usage,
        )

    def transform_streaming_response(
        self,
        model: str,  # noqa: U100
        parsed_chunk: dict,
        logging_obj: Any,  # noqa: U100
    ) -> Any:
        # For streaming, pass through the parsed chunk
        return parsed_chunk
    
    def map_openai_params(
        self,
        non_default_params: Dict[str, Any],
        optional_params: Dict[str, Any],
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        # Pass through all non-default parameters into optional params
        for key, value in non_default_params.items():
            optional_params[key] = value
        return optional_params

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, "httpx.Headers"],  # noqa: F821
    ):
        # Return a BaseLLMException for chat errors
        from litellm.llms.base_llm.chat.transformation import BaseLLMException
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
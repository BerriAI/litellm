
"""
Translates from OpenAI's `/v1/chat/completions` to Heroku's `/v1/chat/completions`
"""
import pdb
from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ...openai_like.chat.transformation import OpenAILikeChatConfig
import httpx
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from typing import (
    Optional,
    Union,
    List,
    Any
)
from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class HerokuError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url=get_secret_str("HEROKU_API_BASE"))
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            status_code=status_code,
            message=self.message,
            headers=self.headers,
            request=self.request,
            response=self.response,
        )


class HerokuChatConfig(OpenAILikeChatConfig):
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HEROKU_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("HEROKU_API_KEY") or "fake-api-key"
        )
        return api_base, dynamic_api_key


    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        return {
            "Authorization": f"Bearer {api_key or get_secret_str("HEROKU_API_KEY")}",
            "content-type": "application/json",
            **headers,
        }

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HerokuError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _handle_streaming_response(
        self,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        messages: List,
    ) -> ModelResponse:
        """
        Handle streamed responses from the API.
        """
        streamed_content = []

        # Assuming `raw_response` is an httpx.StreamingResponse object
        for chunk in raw_response.iter_text():
            streamed_content.append(chunk)
            logging_obj.post_call(
                input=messages,
                api_key="",
                original_response={"chunk": chunk},
            )

        # Join streamed content and set it as the final response
        full_response = "".join(streamed_content)
        model_response.content = full_response

        return model_response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the response from the API.

        Handles both standard and streamed responses.
        """
        # Check if streaming is requested
        stream = optional_params.get("stream", False)
        print_verbose = optional_params.get("print_verbose", False)

        if stream:
            # Handle streaming logic separately if needed
            return self._handle_streaming_response(
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                messages=messages,
            )

        # Call the base method for standard (non-streaming) responses
        return super()._transform_response(
            model=model,
            response=raw_response,
            model_response=model_response,
            stream=stream,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=request_data,
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
            json_mode=json_mode,
            custom_llm_provider="heroku",
            base_model=model,
        )

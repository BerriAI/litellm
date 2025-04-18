"""
Handles the chat completion request for ASI
"""

from typing import Any, Callable, List, Optional, Union, cast

from httpx._config import Timeout

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CustomStreamingDecoder
from litellm.utils import ModelResponse

from ...asi.chat.transformation import ASIChatConfig
from ...openai_like.chat.handler import OpenAILikeChatHandler


class ASIChatCompletion(OpenAILikeChatHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = ASIChatConfig()

    def completion(
        self,
        *,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        custom_endpoint: Optional[bool] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
        fake_stream: bool = False,
    ):
        # Transform messages for ASI
        messages = self.config._transform_messages(
            messages=cast(List[AllMessageValues], messages), model=model
        )

        # Handle JSON response format
        response_format = optional_params.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            # Set flag for JSON extraction in the transformation layer
            optional_params["json_response_requested"] = True
            
            # Add a system message to instruct the model to return JSON
            has_system_message = False
            for message in messages:
                if isinstance(message, dict) and message.get('role') == 'system':
                    has_system_message = True
                    # Enhance existing system message to emphasize JSON format
                    if 'content' in message and message['content']:
                        if 'JSON' not in message['content'] and 'json' not in message['content']:
                            message['content'] += "\n\nIMPORTANT: Format your response as a valid JSON object."
                    break
            
            # If no system message, add one specifically for JSON formatting
            if not has_system_message:
                messages.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant that always responds with valid JSON objects."
                })
            
            # Set json_mode flag for consistent handling
            optional_params["json_mode"] = True

        # ASI handles streaming correctly, no need to fake stream
        fake_stream = False

        # Call the parent class's completion method
        response = super().completion(
            model=model,
            messages=messages,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            custom_prompt_dict=custom_prompt_dict,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            api_key=api_key,
            logging_obj=logging_obj,
            optional_params=optional_params,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            headers=headers,
            timeout=timeout,
            client=client,
            custom_endpoint=custom_endpoint,
            streaming_decoder=streaming_decoder,
            fake_stream=fake_stream,
        )

        return response

    def transform_response(
        self, raw_response: Any, model: str, optional_params: dict, logging_obj: Any = None
    ) -> Any:
        """
        Apply ASI-specific response transformations
        """
        # For ASI, we need to adapt to the OpenAIGPTConfig transform_response signature
        # Create empty/default values for required parameters
        model_response = ModelResponse()
        request_data: dict = {}
        messages: list = []
        litellm_params: dict = {}
        encoding: Optional[Any] = None
        
        # Use our config to transform the response
        transformed_response = self.config.transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding
        )
        
        # Return the transformed response directly
        return transformed_response

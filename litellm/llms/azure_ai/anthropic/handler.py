"""
Azure Anthropic handler - reuses AnthropicChatCompletion logic with Azure authentication
"""
import copy
import json
from typing import TYPE_CHECKING, Callable, Union

import httpx

from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

from .transformation import AzureAnthropicConfig

if TYPE_CHECKING:
    pass


class AzureAnthropicChatCompletion(AnthropicChatCompletion):
    """
    Azure Anthropic chat completion handler.
    Reuses all Anthropic logic but with Azure authentication.
    """

    def __init__(self) -> None:
        super().__init__()

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        """
        Completion method that uses Azure authentication instead of Anthropic's x-api-key.
        All other logic is the same as AnthropicChatCompletion.
        """

        optional_params = copy.deepcopy(optional_params)
        stream = optional_params.pop("stream", None)
        json_mode: bool = optional_params.pop("json_mode", False)
        is_vertex_request: bool = optional_params.pop("is_vertex_request", False)
        _is_function_call = False
        messages = copy.deepcopy(messages)

        # Use AzureAnthropicConfig for both azure_anthropic and azure_ai Claude models
        config = AzureAnthropicConfig()
        
        headers = config.validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params={**optional_params, "is_vertex_request": is_vertex_request},
            litellm_params=litellm_params,
        )

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )
        print_verbose(f"_is_function_call: {_is_function_call}")
        if acompletion is True:
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                print_verbose("makes async azure anthropic streaming POST request")
                data["stream"] = stream
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    json_mode=json_mode,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    provider_config=config,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=client,
                    json_mode=json_mode,
                    timeout=timeout,
                )
        else:
            ## COMPLETION CALL
            if (
                stream is True
            ):  # if function call - fake the streaming (need complete blocks for output parsing in openai format)
                data["stream"] = stream
                # Import the make_sync_call from parent
                from litellm.llms.anthropic.chat.handler import make_sync_call

                completion_stream, response_headers = make_sync_call(
                    client=client,
                    api_base=api_base,
                    headers=headers,  # type: ignore
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    timeout=timeout,
                    json_mode=json_mode,
                )
                from litellm.llms.anthropic.common_utils import (
                    process_anthropic_headers,
                )

                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="azure_ai",
                    logging_obj=logging_obj,
                    _response_headers=process_anthropic_headers(response_headers),
                )

            else:
                if client is None or not isinstance(client, HTTPHandler):
                    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

                    client = _get_httpx_client(params={"timeout": timeout})
                else:
                    client = client

                try:
                    response = client.post(
                        api_base,
                        headers=headers,
                        data=json.dumps(data),
                        timeout=timeout,
                    )
                except Exception as e:
                    from litellm.llms.anthropic.common_utils import AnthropicError

                    status_code = getattr(e, "status_code", 500)
                    error_headers = getattr(e, "headers", None)
                    error_text = getattr(e, "text", str(e))
                    error_response = getattr(e, "response", None)
                    if error_headers is None and error_response:
                        error_headers = getattr(error_response, "headers", None)
                    if error_response and hasattr(error_response, "text"):
                        error_text = getattr(error_response, "text", error_text)
                    raise AnthropicError(
                        message=error_text,
                        status_code=status_code,
                        headers=error_headers,
                    )

        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )


"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""
from typing import TYPE_CHECKING, Any, Coroutine, TypedDict, Union

from litellm import CustomLLM

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper, LiteLLMLoggingObj, ModelResponse


class ResponsesToCompletionBridgeHandlerInputKwargs(TypedDict):
    model: str
    messages: list
    optional_params: dict
    litellm_params: dict
    headers: dict
    model_response: "ModelResponse"
    logging_obj: "LiteLLMLoggingObj"


class ResponsesToCompletionBridgeHandler(CustomLLM):
    def __init__(self):
        from .transformation import LiteLLMResponsesTransformationHandler

        super().__init__()
        self.transformation_handler = LiteLLMResponsesTransformationHandler()

    def validate_input_kwargs(
        self, kwargs: dict
    ) -> ResponsesToCompletionBridgeHandlerInputKwargs:
        from litellm import LiteLLMLoggingObj
        from litellm.types.utils import ModelResponse

        model = kwargs.get("model")
        if model is None or not isinstance(model, str):
            raise ValueError("model is required")

        messages = kwargs.get("messages")
        if messages is None or not isinstance(messages, list):
            raise ValueError("messages is required")

        optional_params = kwargs.get("optional_params")
        if optional_params is None or not isinstance(optional_params, dict):
            raise ValueError("optional_params is required")

        litellm_params = kwargs.get("litellm_params")
        if litellm_params is None or not isinstance(litellm_params, dict):
            raise ValueError("litellm_params is required")

        headers = kwargs.get("headers")
        if headers is None or not isinstance(headers, dict):
            raise ValueError("headers is required")

        model_response = kwargs.get("model_response")
        if model_response is None or not isinstance(model_response, ModelResponse):
            raise ValueError("model_response is required")

        logging_obj = kwargs.get("logging_obj")
        if logging_obj is None or not isinstance(logging_obj, LiteLLMLoggingObj):
            raise ValueError("logging_obj is required")

        return ResponsesToCompletionBridgeHandlerInputKwargs(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
            model_response=model_response,
            logging_obj=logging_obj,
        )

    def completion(
        self, *args, **kwargs
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        from litellm import responses
        from litellm.types.llms.openai import ResponsesAPIResponse

        validated_kwargs = self.validate_input_kwargs(kwargs)
        model = validated_kwargs["model"]
        messages = validated_kwargs["messages"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        headers = validated_kwargs["headers"]
        model_response = validated_kwargs["model_response"]
        logging_obj = validated_kwargs["logging_obj"]

        request_data = self.transformation_handler.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        result = responses(
            **request_data,
        )

        if isinstance(result, ResponsesAPIResponse):
            return self.transformation_handler.transform_response(
                model=model,
                raw_response=result,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        else:
            raise ValueError(f"Unexpected response type: {type(result)}")

    async def acompletion(
        self, *args, **kwargs
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        from litellm import aresponses
        from litellm.types.llms.openai import ResponsesAPIResponse

        validated_kwargs = self.validate_input_kwargs(kwargs)
        model = validated_kwargs["model"]
        messages = validated_kwargs["messages"]
        optional_params = validated_kwargs["optional_params"]
        litellm_params = validated_kwargs["litellm_params"]
        headers = validated_kwargs["headers"]
        model_response = validated_kwargs["model_response"]
        logging_obj = validated_kwargs["logging_obj"]

        request_data = self.transformation_handler.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        result = await aresponses(
            **request_data,
            aresponses=True,
        )

        if isinstance(result, ResponsesAPIResponse):
            return self.transformation_handler.transform_response(
                model=model,
                raw_response=result,
                model_response=model_response,
                logging_obj=logging_obj,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=kwargs.get("encoding"),
                api_key=kwargs.get("api_key"),
                json_mode=kwargs.get("json_mode"),
            )
        else:
            raise ValueError(f"Unexpected response type: {type(result)}")


responses_api_bridge = ResponsesToCompletionBridgeHandler()

import json
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Final, Literal, cast, overload

import httpx

import litellm
from litellm.constants import MIN_NON_ZERO_TEMPERATURE
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )


class DeepInfraConfig(OpenAIGPTConfig):
    """
    Reference: https://deepinfra.com/docs/advanced/openai_api

    The class `DeepInfra` provides configuration for the DeepInfra's Chat Completions API interface. Below are the parameters:
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "deepinfra"

    frequency_penalty: int | None = None
    function_call: str | dict | None = None
    functions: list | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    stop: str | list | None = None
    temperature: int | None = None
    top_p: int | None = None
    response_format: dict | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None

    def __init__(
        self,
        frequency_penalty: int | None = None,
        function_call: str | dict | None = None,
        functions: list | None = None,
        logit_bias: dict | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | list | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        supported_openai_params: Final = [
            "stream",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "tools",
            "tool_choice",
        ]

        if litellm.supports_reasoning(
            model=model,
            custom_llm_provider=self.custom_llm_provider,
        ):
            supported_openai_params.append("reasoning_effort")
        return supported_openai_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if (
                param == "temperature" and value == 0 and model == "mistralai/Mistral-7B-Instruct-v0.1"
            ):  # this model does no support temperature == 0
                value = MIN_NON_ZERO_TEMPERATURE  # close to 0
            if param == "tool_choice":
                if value != "auto" and value != "none":  # https://deepinfra.com/docs/advanced/function_calling
                    ## UNSUPPORTED TOOL CHOICE VALUE
                    if litellm.drop_params is True or drop_params is True:
                        value = None
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message=f"Deepinfra doesn't support tool_choice={value}. To drop unsupported openai params from the call, set `litellm.drop_params = True`",
                            status_code=400,
                        )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _transform_tool_message_content(self, messages: list[AllMessageValues]) -> list[AllMessageValues]:
        """
        Transform tool message content from array to string format for DeepInfra compatibility.

        DeepInfra requires tool message content to be a string, not an array.
        This method converts tool message content from array format to string format.

        Example transformation:
        - Input:  {"role": "tool", "content": [{"type": "text", "text": "20"}]}
        - Output: {"role": "tool", "content": "20"}

        Or if content is complex:
        - Input:  {"role": "tool", "content": [{"type": "text", "text": "result"}]}
        - Output: {"role": "tool", "content": "[{\"type\": \"text\", \"text\": \"result\"}]"}
        """
        for message in messages:
            if message.get("role") == "tool":
                content = message.get("content")

                # If content is a list/array, convert it to string
                if isinstance(content, list):
                    # Check if it's a simple single text item
                    if (
                        len(content) == 1
                        and isinstance(content[0], dict)
                        and content[0].get("type") == "text"
                        and "text" in content[0]
                    ):
                        # Extract just the text value for simple cases
                        message["content"] = content[0]["text"]
                    else:
                        # For complex content, serialize the entire array as JSON string
                        message["content"] = json.dumps(content)

        return messages

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
        """
        Transform messages for DeepInfra compatibility.
        Handles both sync and async transformations.
        """
        if is_async:
            # For async case, create an async function that awaits parent and applies our transformation
            async def _async_transform():
                # Call parent with is_async=True (literal) for async case
                parent_result: Final = super(DeepInfraConfig, self)._transform_messages(
                    messages=messages, model=model, is_async=cast(Literal[True], True)
                )
                transformed_messages: Final = await parent_result
                return self._transform_tool_message_content(transformed_messages)

            return _async_transform()
        else:
            # Call parent with is_async=False (literal) for sync case
            parent_result: Final = super()._transform_messages(
                messages=messages, model=model, is_async=cast(Literal[False], False)
            )
            # For sync case, parent_result is already the transformed messages
            return self._transform_tool_message_content(parent_result)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = api_base or get_secret_str("DEEPINFRA_API_BASE") or "https://api.deepinfra.com/v1/openai"
        dynamic_api_key: Final = api_key or get_secret_str("DEEPINFRA_API_KEY")
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform the response from DeepInfra.

        DeepInfra reports the charge it actually applied as
        ``usage.estimated_cost``, which already reflects the delivered
        ``service_tier``. Pricing from the static map instead under-reports
        priority-tier traffic, so the reported value is treated as
        authoritative, matching how OpenRouter's ``usage.cost`` is handled.
        """
        model_response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        try:
            response_json: Final = raw_response.json()
            usage: Final = response_json.get("usage") or {}
            reported_cost: Final = usage.get("estimated_cost")
            service_tier: Final = response_json.get("service_tier")

            if reported_cost is not None:
                if not hasattr(model_response, "_hidden_params"):
                    model_response._hidden_params = {}
                if "additional_headers" not in model_response._hidden_params:
                    model_response._hidden_params["additional_headers"] = {}
                model_response._hidden_params["additional_headers"][
                    "llm_provider-x-litellm-response-cost"
                ] = float(reported_cost)

            if service_tier is not None:
                setattr(model_response, "service_tier", service_tier)
        except Exception:
            pass

        return model_response

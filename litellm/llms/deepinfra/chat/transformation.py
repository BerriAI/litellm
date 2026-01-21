import json
from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import litellm
from litellm.constants import MIN_NON_ZERO_TEMPERATURE
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class DeepInfraConfig(OpenAIGPTConfig):
    """
    Reference: https://deepinfra.com/docs/advanced/openai_api

    The class `DeepInfra` provides configuration for the DeepInfra's Chat Completions API interface. Below are the parameters:
    """
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "deepinfra"

    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        supported_openai_params = [
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
            "tool_choice"
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
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if (
                param == "temperature"
                and value == 0
                and model == "mistralai/Mistral-7B-Instruct-v0.1"
            ):  # this model does no support temperature == 0
                value = MIN_NON_ZERO_TEMPERATURE  # close to 0
            if param == "tool_choice":
                if (
                    value != "auto" and value != "none"
                ):  # https://deepinfra.com/docs/advanced/function_calling
                    ## UNSUPPORTED TOOL CHOICE VALUE
                    if litellm.drop_params is True or drop_params is True:
                        value = None
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Deepinfra doesn't support tool_choice={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                value
                            ),
                            status_code=400,
                        )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _transform_tool_message_content(self, messages: List[AllMessageValues]) -> List[AllMessageValues]:
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
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[False] = False
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Transform messages for DeepInfra compatibility.
        Handles both sync and async transformations.
        """
        if is_async:
            # For async case, create an async function that awaits parent and applies our transformation
            async def _async_transform():
                # Call parent with is_async=True (literal) for async case
                parent_result = super(DeepInfraConfig, self)._transform_messages(
                    messages=messages, model=model, is_async=cast(Literal[True], True)
                )
                transformed_messages = await parent_result
                return self._transform_tool_message_content(transformed_messages)
            return _async_transform()
        else:
            # Call parent with is_async=False (literal) for sync case
            parent_result = super()._transform_messages(
                messages=messages, model=model, is_async=cast(Literal[False], False)
            )
            # For sync case, parent_result is already the transformed messages
            return self._transform_tool_message_content(parent_result)

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
        api_base = (
            api_base
            or get_secret_str("DEEPINFRA_API_BASE")
            or "https://api.deepinfra.com/v1/openai"
        )
        dynamic_api_key = api_key or get_secret_str("DEEPINFRA_API_KEY")
        return api_base, dynamic_api_key

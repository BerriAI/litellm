from typing import Optional, Tuple, Union

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class FeatherlessAIConfig(OpenAIGPTConfig):
    """
    Reference: https://featherless.ai/docs/completions

    The class `FeatherlessAI` provides configuration for the FeatherlessAI's Chat Completions API interface. Below are the parameters:
    """

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
    tool_choice: Optional[str] = None
    tools: Optional[list] = None

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
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        return [
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
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "tool_choice" or param == "tools":
                if param == "tool_choice" and (value == "auto" or value == "none"):
                    # These values are supported, so add them to optional_params
                    optional_params[param] = value
                else:  # https://featherless.ai/docs/completions
                    ## UNSUPPORTED TOOL CHOICE VALUE
                    if litellm.drop_params is True or drop_params is True:
                        value = None
                    else:
                        error_message = f"Featherless AI doesn't support {param}={value}. To drop unsupported openai params from the call, set `litellm.drop_params = True`"
                        raise litellm.utils.UnsupportedParamsError(
                            message=error_message,
                            status_code=400,
                        )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # FeatherlessAI is openai compatible, set to custom_openai and use FeatherlessAI's endpoint
        api_base = (
            api_base
            or get_secret_str("FEATHERLESS_API_BASE")
            or "https://api.featherless.ai/v1"
        )
        dynamic_api_key = api_key or get_secret_str("FEATHERLESS_API_KEY")
        return api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if not api_key:
            raise ValueError("Missing Featherless AI API Key")

        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers

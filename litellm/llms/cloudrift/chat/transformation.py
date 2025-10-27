import litellm
from typing import Optional, Tuple, Union
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

from litellm.secret_managers.main import get_secret_str

class CloudRiftChatConfig(OpenAIGPTConfig):
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


    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "cloudrift"

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
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
            if param == "tool_choice":
                raise litellm.utils.UnsupportedParamsError(
                    message="Cloudrift doesn't support tool_choice={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
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

    def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Cloudrift api is openai compatible
        api_base = (
                api_base
                or get_secret_str("CLOUDRIFT_API_BASE")
                or "https://inference.cloudrift.ai/v1"  # Default Lambda API base URL
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CLOUDRIFT_API_KEY")
        return api_base, dynamic_api_key

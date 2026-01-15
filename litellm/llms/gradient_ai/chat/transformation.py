from typing import List, Optional, Tuple, Union, Dict, Literal

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
)

from ...openai_like.chat.transformation import OpenAILikeChatConfig

# Default GradientAI endpoint
GRADIENT_AI_SERVERLESS_ENDPOINT = "https://inference.do-ai.run"


class GradientAIConfig(OpenAILikeChatConfig):

    k: Optional[int] = None
    kb_filters: Optional[List[Dict]] = None
    filter_kb_content_by_query_metadata: Optional[bool] = None
    instruction_override: Optional[str] = None
    include_functions_info: Optional[bool] = None
    include_retrieval_info: Optional[bool] = None
    include_guardrails_info: Optional[bool] = None
    provide_citations: Optional[bool] = None
    retrieval_method: Optional[Literal["rewrite", "step_back", "sub_queries", "none"]] = None

    def __init__(
        self,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        retrieval_method: Optional[str] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        k: Optional[int] = None,
        kb_filters: Optional[List[Dict]] = None,
        filter_kb_content_by_query_metadata: Optional[bool] = None,
        instruction_override: Optional[str] = None,
        include_functions_info: Optional[bool] = None,
        include_retrieval_info: Optional[bool] = None,
        include_guardrails_info: Optional[bool] = None,
        provide_citations: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        supported_params = [
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            # GradientAI specific parameters
            "k",
            "kb_filters",
            "filter_kb_content_by_query_metadata",
            "instruction_override",
            "include_functions_info",
            "include_retrieval_info",
            "include_guardrails_info",
            "provide_citations",
            "retrieval_method",
        ]
        return supported_params

    def validate_environment(self,
                             headers: dict,
                             model: str,
                             messages: List[AllMessageValues],
                             optional_params: dict,
                             litellm_params: dict,
                             api_key: Optional[str] = None,
                             api_base: Optional[str] = None):
        api_key = api_key or get_secret_str("GRADIENT_AI_API_KEY")
        if api_key is None:
            raise ValueError("GradientAI API key not found")
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        gradient_ai_endpoint = get_secret_str("GRADIENT_AI_AGENT_ENDPOINT")
        complete_url = f"{GRADIENT_AI_SERVERLESS_ENDPOINT}/v1/chat/completions"

        if api_base and api_base != GRADIENT_AI_SERVERLESS_ENDPOINT:
            complete_url = f"{api_base}/api/v1/chat/completions"
        elif gradient_ai_endpoint and gradient_ai_endpoint != GRADIENT_AI_SERVERLESS_ENDPOINT:
            complete_url = f"{gradient_ai_endpoint}/api/v1/chat/completions"

        return complete_url

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        gradient_ai_endpoint = get_secret_str("GRADIENT_AI_AGENT_ENDPOINT")

        if not api_base and not gradient_ai_endpoint:
            api_base = GRADIENT_AI_SERVERLESS_ENDPOINT
        else:
            api_base = api_base or gradient_ai_endpoint

        dynamic_api_key = api_key or get_secret_str("GRADIENT_AI_API_KEY")
        return api_base, dynamic_api_key

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
        replace_max_completion_tokens_with_max_tokens: bool = False,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
            elif not drop_params:
                from litellm.utils import UnsupportedParamsError
                raise UnsupportedParamsError(
                    status_code=400,
                    message=f"GradientAI does not support parameter '{param}'. To drop unsupported params, set `drop_params=True`."
                )

        return optional_params

from typing import Final, Literal

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
)

from ...openai_like.chat.transformation import OpenAILikeChatConfig

# Default GradientAI endpoint
GRADIENT_AI_SERVERLESS_ENDPOINT: Final = "https://inference.do-ai.run"


class GradientAIConfig(OpenAILikeChatConfig):
    k: int | None = None
    kb_filters: list[dict] | None = None
    filter_kb_content_by_query_metadata: bool | None = None
    instruction_override: str | None = None
    include_functions_info: bool | None = None
    include_retrieval_info: bool | None = None
    include_guardrails_info: bool | None = None
    provide_citations: bool | None = None
    retrieval_method: Literal["rewrite", "step_back", "sub_queries", "none"] | None = None

    def __init__(
        self,
        frequency_penalty: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        presence_penalty: float | None = None,
        retrieval_method: str | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        k: int | None = None,
        kb_filters: list[dict] | None = None,
        filter_kb_content_by_query_metadata: bool | None = None,
        instruction_override: str | None = None,
        include_functions_info: bool | None = None,
        include_retrieval_info: bool | None = None,
        include_guardrails_info: bool | None = None,
        provide_citations: bool | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        supported_params: Final = [
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

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
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
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        gradient_ai_endpoint: Final = get_secret_str("GRADIENT_AI_AGENT_ENDPOINT")
        complete_url = f"{GRADIENT_AI_SERVERLESS_ENDPOINT}/v1/chat/completions"

        if api_base and api_base != GRADIENT_AI_SERVERLESS_ENDPOINT:
            complete_url = f"{api_base}/api/v1/chat/completions"
        elif gradient_ai_endpoint and gradient_ai_endpoint != GRADIENT_AI_SERVERLESS_ENDPOINT:
            complete_url = f"{gradient_ai_endpoint}/api/v1/chat/completions"

        return complete_url

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        gradient_ai_endpoint: Final = get_secret_str("GRADIENT_AI_AGENT_ENDPOINT")

        if not api_base and not gradient_ai_endpoint:
            api_base = GRADIENT_AI_SERVERLESS_ENDPOINT
        else:
            api_base = api_base or gradient_ai_endpoint

        dynamic_api_key: Final = api_key or get_secret_str("GRADIENT_AI_API_KEY")
        return api_base, dynamic_api_key

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
        replace_max_completion_tokens_with_max_tokens: bool = False,
    ) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
            elif not drop_params:
                from litellm.utils import UnsupportedParamsError

                raise UnsupportedParamsError(
                    status_code=400,
                    message=f"GradientAI does not support parameter '{param}'. To drop unsupported params, set `drop_params=True`.",
                )

        return optional_params

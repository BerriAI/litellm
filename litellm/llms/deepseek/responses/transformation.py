from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class DeepSeekResponsesConfig(OpenAIResponsesAPIConfig):
    @staticmethod
    def supports_model(model: str | None) -> bool:
        return model is not None and model.removeprefix("deepseek/") in {
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        }

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DEEPSEEK

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: the caller concatenates this with a list
        return [
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_logprobs",
            "tools",
            "tool_choice",
            "reasoning",
            "text",
            "user",
            "stream",
            "store",
            "parallel_tool_calls",
            "max_tool_calls",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: provider base contract requires a dict
        if (
            "store" in response_api_optional_params
            and response_api_optional_params["store"] is True
            and not drop_params
        ):
            raise litellm.UnsupportedParamsError(
                message="DeepSeek Responses is stateless and does not support store=True. Use store=False.",
                status_code=400,
            )
        supported: Final = self.get_supported_openai_params(model)
        return {
            key: value for key, value in response_api_optional_params.items() if key in supported and key != "store"
        }

    def validate_environment(
        self, headers: Mapping[str, str], model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, str]:  # mutable-ok: the HTTP handler updates the returned headers
        api_key: Final = (
            (litellm_params.api_key if litellm_params is not None else None)
            or get_secret_str("DEEPSEEK_API_KEY")
            or litellm.api_key
        )
        return {
            "Content-Type": "application/json",
            **headers,
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        }

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        base: Final = (api_base or get_secret_str("DEEPSEEK_API_BASE") or "https://api.deepseek.com").rstrip("/")
        if base in {"https://api.deepseek.com/beta", "https://api.deepseek.com/v1"}:
            return "https://api.deepseek.com/responses"
        return base if base.endswith("/responses") else f"{base}/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: the HTTP handler merges extra_body into this dict
        return {"model": model, "input": input, **response_api_optional_request_params}

    def supports_native_websocket(self) -> bool:
        return False

    def supports_native_file_search(self) -> bool:
        return False

    def supports_encrypted_agent_messages(self) -> bool:
        return False

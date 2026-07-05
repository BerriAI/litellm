from typing import List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import (
    get_secret_str,
    normalize_nonempty_secret_str,
)
from litellm.types.llms.openai import AllMessageValues


class CloudflareError(BaseLLMException):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cloudflare.com")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            request=self.request,
            response=self.response,
        )


class CloudflareChatConfig(OpenAIGPTConfig):
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return super().get_complete_url(
            api_base=self._resolve_api_base(api_base),
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    @staticmethod
    def _resolve_api_base(api_base: Optional[str]) -> str:
        if not api_base:
            account_id = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
            if account_id is None:
                raise ValueError(
                    "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID in the environment or pass api_base explicitly"
                )
            return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        trimmed = api_base.rstrip("/")
        if trimmed.endswith("/ai/run"):
            verbose_logger.warning(
                "Cloudflare api_base ending in '/ai/run' is the legacy Workers AI path and no longer serves OpenAI-compatible requests; rewriting to the '/ai/v1' endpoint"
            )
            return f"{trimmed[: -len('/ai/run')]}/ai/v1"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            raise ValueError(
                "Missing Cloudflare API Key - A call is being made to cloudflare but no key is set either in the environment variables or via params"
            )
        return super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CloudflareError(
            status_code=status_code,
            message=error_message,
        )

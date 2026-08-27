from collections.abc import Coroutine
from typing import (  # noqa: TID251  # Any only appears in the base overload's Coroutine[Any, Any, ...]
    Any,
    Final,
    Literal,
    overload,
)

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
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
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
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
    def _resolve_api_base(api_base: str | None) -> str:
        if not api_base:
            account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
            if account_id is None:
                raise ValueError(
                    "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID in the environment or pass api_base explicitly"
                )
            return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        trimmed: Final = api_base.rstrip("/")
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
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

    # fmt: off
    # The signature mirrors OpenAIGPTConfig._transform_messages, which is overloaded
    # on is_async; the annotations below are the base's own.
    @overload
    def _transform_messages(  # mutable-ok: base signature
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]  # mutable-ok: base signature
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...  # mutable-ok: base signature

    @overload
    def _transform_messages(  # mutable-ok: base signature
        self, messages: list[AllMessageValues], model: str, is_async: Literal[False] = False  # mutable-ok: base sig
    ) -> list[AllMessageValues]: ...  # mutable-ok: base signature

    def _transform_messages(  # mutable-ok: base signature
        self, messages: list[AllMessageValues], model: str, is_async: bool = False  # mutable-ok: base signature
    ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:  # mutable-ok: base signature
        # fmt: on
        """
        Cloudflare Workers AI requires message content to be a string, so OpenAI
        content-part arrays have to be flattened before the request is sent
        """
        flattened: Final = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(messages=flattened, model=model, is_async=True)
        return super()._transform_messages(messages=flattened, model=model, is_async=False)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return CloudflareError(
            status_code=status_code,
            message=error_message,
        )

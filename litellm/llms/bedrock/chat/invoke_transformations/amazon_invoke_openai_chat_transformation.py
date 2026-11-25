from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING

import httpx

from litellm.llms.bedrock.chat.chat_completions.transformation import AmazonBedrockRuntimeChatCompletionsConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import AmazonInvokeConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer


def _without_model(body: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: BaseConfig request body
    return {key: value for key, value in body.items() if key != "model"}


class AmazonInvokeOpenAIChatConfig(AmazonInvokeConfig):
    @cached_property
    def _chat_body(self) -> AmazonBedrockRuntimeChatCompletionsConfig:
        return AmazonBedrockRuntimeChatCompletionsConfig()

    @property
    def uses_async_transform_request(self) -> bool:
        return True

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseConfig signature
        return self._chat_body.get_supported_openai_params(model)

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, object],  # mutable-ok: BaseConfig signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
        return self._chat_body.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, object],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, object],  # mutable-ok: BaseConfig signature
        headers: dict[str, object],  # mutable-ok: BaseConfig signature
    ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
        return _without_model(
            self._chat_body.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        )

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, object],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, object],  # mutable-ok: BaseConfig signature
        headers: dict[str, object],  # mutable-ok: BaseConfig signature
    ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
        return _without_model(
            await self._chat_body.async_transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: BaseConfig signature
        messages: list[AllMessageValues],  # mutable-ok: BaseConfig signature
        optional_params: dict[str, object],  # mutable-ok: BaseConfig signature
        litellm_params: dict[str, object],  # mutable-ok: BaseConfig signature
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return self._chat_body.transform_response(
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

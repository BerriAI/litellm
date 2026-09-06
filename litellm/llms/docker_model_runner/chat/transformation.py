"""
Translates from OpenAI's `/v1/chat/completions` to Docker Model Runner's `/engines/v1/chat/completions`

Docker Model Runner API Reference: https://docs.docker.com/ai/model-runner/api-reference/
"""

from collections.abc import Coroutine
from typing import Any, Final, Literal, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DockerModelRunnerChatConfig(OpenAIGPTConfig):
    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by OpenAIGPTConfig
        model: str,
        is_async: Literal[True],
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...  # mutable-ok: signature dictated by OpenAIGPTConfig

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by OpenAIGPTConfig
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...  # mutable-ok: signature dictated by OpenAIGPTConfig

    @staticmethod
    def _has_multimodal_content(message: AllMessageValues) -> bool:
        message_content: Final = message.get("content")
        if not message_content or not isinstance(message_content, list):
            return False
        for c in message_content:
            block_type = c.get("type")
            if block_type is not None and block_type != "text":
                return True
            if block_type is None and isinstance(c, dict) and "text" not in c:
                return True
        return False

    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by OpenAIGPTConfig
        model: str,
        is_async: bool = False,
    ) -> (
        list[AllMessageValues]  # mutable-ok: signature dictated by OpenAIGPTConfig
        | Coroutine[Any, Any, list[AllMessageValues]]  # mutable-ok: signature dictated by OpenAIGPTConfig
    ):
        text_only_messages: Final = [  # mutable-ok: API request payload
            m for m in messages if not self._has_multimodal_content(m)
        ]

        converted: Final = iter(handle_messages_with_content_list_to_str_conversion(text_only_messages))
        transformed: Final = [  # mutable-ok: API request payload
            m if self._has_multimodal_content(m) else next(converted) for m in messages
        ]

        if is_async:
            return super()._transform_messages(messages=transformed, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=transformed, model=model, is_async=False)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        """
        Get API base and key for Docker Model Runner.

        Default API base: http://localhost:12434/engines/v1
        """
        api_base = (  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
            api_base or get_secret_str("DOCKER_MODEL_RUNNER_API_BASE") or "http://localhost:12434/engines/v1"
        )
        # Docker Model Runner may not require authentication for local instances
        dynamic_api_key: Final = api_key or get_secret_str("DOCKER_MODEL_RUNNER_API_KEY") or "dummy-key"
        return api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature dictated by OpenAIGPTConfig
        optional_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        litellm_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: signature dictated by OpenAIGPTConfig
        default_headers: Final = {  # mutable-ok: API request payload
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'dummy-key'}",
        }

        return {**default_headers, **headers}  # mutable-ok: API request payload

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        litellm_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        stream: bool | None = None,
    ) -> str:
        base_url: Final = (api_base or "http://localhost:12434/engines/v1").rstrip("/")
        return f"{base_url}/chat/completions"

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: signature dictated by OpenAIGPTConfig
        """
        Get the supported OpenAI params for Docker Model Runner.

        Docker Model Runner is OpenAI-compatible and supports standard parameters.
        """
        return super().get_supported_openai_params(model=model)

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        optional_params: dict,  # mutable-ok: signature dictated by OpenAIGPTConfig
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature dictated by OpenAIGPTConfig
        """
        Map OpenAI parameters to Docker Model Runner parameters.

        Docker Model Runner is OpenAI-compatible, so most parameters map directly.
        """
        supported_openai_params: Final = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value  # rebind-ok: out-param store
            elif param in supported_openai_params:
                optional_params[param] = value  # rebind-ok: out-param store

        return optional_params

"""
Translates calls from OpenAI's `/v1/completions` endpoint to
Docker Model Runner's `/engines/v1/completions` endpoint.

Docker Model Runner API Reference: https://docs.docker.com/ai/model-runner/api-reference/
"""

from typing import Final

from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
from litellm.llms.openai.completion.utils import (
    _transform_prompt,  # pyright: ignore[reportPrivateUsage]  # shared helper already imported by the fireworks_ai and together_ai completion transforms
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage


class DockerModelRunnerCompletionConfig(OpenAITextCompletionConfig):
    def _get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple:
        api_base = (  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
            api_base or get_secret_str("DOCKER_MODEL_RUNNER_API_BASE") or "http://localhost:12434/engines/v1"
        )
        dynamic_api_key: Final = api_key or get_secret_str("DOCKER_MODEL_RUNNER_API_KEY") or "dummy-key"
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: signature dictated by OpenAITextCompletionConfig
        litellm_params: dict,  # mutable-ok: signature dictated by OpenAITextCompletionConfig
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            api_base = (  # rebind-ok: normalize the argument locally, mirrors hosted_vllm
                "http://localhost:12434/engines/v1"
            )

        return f"{api_base.rstrip('/')}/completions"

    def transform_text_completion_request(
        self,
        model: str,
        messages: list[AllMessageValues]  # mutable-ok: signature dictated by OpenAITextCompletionConfig
        | list[OpenAITextCompletionUserMessage],
        optional_params: dict,  # mutable-ok: signature dictated by OpenAITextCompletionConfig
        headers: dict,  # mutable-ok: signature dictated by OpenAITextCompletionConfig
    ) -> dict:  # mutable-ok: signature dictated by OpenAITextCompletionConfig
        prompt: Final = _transform_prompt(messages)
        return {  # mutable-ok: API request payload
            "model": model,
            "prompt": prompt,
            **optional_params,
        }

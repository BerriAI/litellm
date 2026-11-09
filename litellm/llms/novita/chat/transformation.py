"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Novita AI is openai-compatible.

Docs: https://novita.ai/docs/guides/llm-api
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ....types.llms.openai import AllMessageValues
from ...openai.chat.gpt_transformation import OpenAIGPTConfig

_NOVITA_ATTRIBUTION_HEADERS: Final[Mapping[str, str]] = MappingProxyType({"X-Novita-Source": "litellm"})


class NovitaConfig(OpenAIGPTConfig):
    def get_attribution_headers(self) -> Mapping[str, str]:
        return _NOVITA_ATTRIBUTION_HEADERS

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
                "Missing Novita AI API Key - A call is being made to novita but no key is set either in the environment variables or via params"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        if not any(name.lower() == "x-novita-source" for name in headers):
            headers["X-Novita-Source"] = "litellm"
        return headers

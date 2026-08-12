"""
Translate from OpenAI's `/v1/chat/completions` to Parallel AI's `/chat/completions`.

Parallel AI Chat API Reference: https://docs.parallel.ai/chat-api/chat-quickstart
"""

from typing import Any, Final

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


def _citation_urls_from_basis(basis: object) -> list[str]:
    """Collect unique citation URLs from Parallel's per-field research basis."""
    if not isinstance(basis, list):
        return []
    urls: Final = [
        url
        for field_basis in basis
        if isinstance(field_basis, dict)
        for citations in [field_basis.get("citations")]
        if isinstance(citations, list)
        for citation in citations
        if isinstance(citation, dict)
        for url in [citation.get("url")]
        if isinstance(url, str) and url
    ]
    return list(dict.fromkeys(urls))


class ParallelAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "parallel_ai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("PARALLEL_AI_API_BASE") or "https://api.parallel.ai"
        resolved_api_key: Final = api_key or get_secret_str("PARALLEL_AI_API_KEY") or get_secret_str("PARALLEL_API_KEY")
        return resolved_api_base, resolved_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Parallel's Chat API supports a subset of OpenAI params.

        Ref: https://docs.parallel.ai/chat-api/chat-quickstart

        Sampling params (temperature, top_p, penalties, ...) and tool calling are not
        supported; the research models ground every answer with built-in web research.
        """
        return [
            "stream",
            "response_format",
            "max_retries",
            "extra_headers",
        ]

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        transformed_response: Final = super().transform_response(
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

        try:
            raw_response_json: Final = raw_response.json()
        except ValueError as e:
            verbose_logger.debug("Error parsing Parallel AI response for basis extraction: %s", e)
            return transformed_response

        basis: Final = raw_response_json.get("basis") if isinstance(raw_response_json, dict) else None
        if basis:
            transformed_response.basis = basis
            citation_urls: Final = _citation_urls_from_basis(basis)
            if citation_urls:
                transformed_response.citations = citation_urls

        return transformed_response

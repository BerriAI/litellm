"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as OrcaRouter is openai-compatible.

Docs: https://docs.orcarouter.ai
"""

from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OrcaRouterException


class OrcaRouterConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "orcarouter"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("ORCAROUTER_API_BASE") or "https://api.orcarouter.ai/v1"
        dynamic_api_key = api_key or get_secret_str("ORCAROUTER_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        # OrcaRouter accepts OpenAI's `extra_body` to carry routing preferences
        # (e.g. {"models": [...], "route": "fallback"}). See docs.orcarouter.ai.
        if "extra_body" not in base_params:
            base_params.append("extra_body")
        return base_params

    @staticmethod
    def _normalize_orcarouter_model(model: str) -> str:
        """Map a LiteLLM model name to the id OrcaRouter's API expects.

        OrcaRouter routes by *namespaced* model id. After LiteLLM strips the
        ``orcarouter/`` provider prefix, a bare router/model name (no ``/``)
        such as ``auto`` would be sent on its own and the backend responds
        ``503 No available channel for model auto``. Such bare names must be
        re-qualified as ``orcarouter/<name>``. Names that already carry an
        upstream namespace (e.g. ``openai/gpt-5``) are what the backend wants
        and are sent unchanged.

        Note: this uses a simple "contains ``/``" heuristic. Every model
        OrcaRouter serves is namespaced (``<vendor>/<model>``) and router
        names are bare, so the only names without a ``/`` are routers that do
        need the prefix. A hypothetical bare model whose own id contained a
        ``/`` would be left unchanged; no such id exists in OrcaRouter's
        catalog today.
        """
        if "/" not in model:
            return f"orcarouter/{model}"
        return model

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return super().transform_request(
            self._normalize_orcarouter_model(model),
            messages,
            optional_params,
            litellm_params,
            headers,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OrcaRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

"""
Support for Scaleway's `/v1/rerank` endpoint.

The request and response match Jina AI's, so this reuses that config.

API reference: https://www.scaleway.com/en/developers/api/generative-apis/#path-rerank-create-a-reranking
"""

from collections.abc import Mapping
from typing import Final

from litellm.llms.jina_ai.rerank.transformation import JinaAIRerankConfig
from litellm.secret_managers.main import get_secret_str

SCALEWAY_API_BASE: Final = "https://api.scaleway.ai/v1"


class ScalewayRerankConfig(JinaAIRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list[str]:  # mutable-ok: BaseRerankConfig contract
        return ["query", "top_n", "documents"]

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        base: Final = SCALEWAY_API_BASE if api_base is None else api_base.rstrip("/")
        return f"{base}/rerank"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRerankConfig contract
        key: Final = api_key or get_secret_str("SCW_SECRET_KEY")
        if not key:
            raise ValueError(
                "Scaleway API key not found. Pass `api_key=...` or set the SCW_SECRET_KEY environment variable."
            )
        provider_headers: Final = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }
        caller_headers: Final = {name: value for name, value in headers.items() if name.lower() not in provider_headers}
        return {**caller_headers, **provider_headers}

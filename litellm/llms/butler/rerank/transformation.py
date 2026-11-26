"""
Support for Butler's `/v1/rerank` endpoint.

The request and response match Jina AI's, so this reuses that config.
"""

from collections.abc import Mapping

from litellm.llms.jina_ai.rerank.transformation import JinaAIRerankConfig


class ButlerRerankConfig(JinaAIRerankConfig):
    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseRerankConfig contract
        if api_key is None:
            raise ValueError(
                "Butler API key not found. Pass `api_key=...` or set the BUTLER_API_KEY environment variable."
            )
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }

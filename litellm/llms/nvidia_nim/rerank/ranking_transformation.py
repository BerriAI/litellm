"""
Transformation for NVIDIA NIM Ranking models that use /v1/ranking endpoint.

Use this by passing "nvidia_nim/ranking/<model>" to force the /v1/ranking endpoint.

Reference: https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy
"""

from typing import Any

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig
from litellm.types.rerank import RerankResponse


class NvidiaNimRankingConfig(NvidiaNimRerankConfig):
    """
    Configuration for NVIDIA NIM models that use the /v1/ranking endpoint.

    The native /v1/ranking request schema accepts only 'model', 'query',
    'passages', and 'truncate' -- requests containing 'top_k' are rejected
    with a 400 validation error. Cohere-compatible 'top_n' is therefore
    applied client-side by truncating the converted response instead of
    being forwarded to the endpoint.

    Example:
        curl -X "POST" 'https://ai.api.nvidia.com/v1/ranking' \
            -H 'Accept: application/json' \
            -H 'Content-Type: application/json' \
            -d '{
                "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                "query": {"text": "which way did the traveler go?"},
                "passages": [{"text": "..."}, {"text": "..."}],
                "truncate": "END"
            }'
    """

    def __init__(self) -> None:
        super().__init__()
        # top_n captured in transform_rerank_request and applied in
        # transform_rerank_response. The provider config is instantiated
        # per-request (see ProviderConfigManager.get_provider_rerank_config),
        # so this does not leak across requests.
        self._client_side_top_n: int | None = None

    def _get_clean_model_name(self, model: str) -> str:
        """Strip 'nvidia_nim/' and 'ranking/' prefixes from model name."""
        # First strip nvidia_nim/ prefix if present
        if model.startswith("nvidia_nim/"):
            model = model[len("nvidia_nim/") :]
        # Then strip ranking/ prefix if present
        if model.startswith("ranking/"):
            model = model[len("ranking/") :]
        return model

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        """
        Construct the Nvidia NIM ranking URL.

        Format: {api_base}/v1/ranking
        """
        if not api_base:
            api_base = self.DEFAULT_NIM_RERANK_API_BASE

        api_base = api_base.rstrip("/")

        if api_base.endswith("/ranking"):
            return api_base

        if api_base.endswith("/v1"):
            api_base = api_base[:-3]

        return f"{api_base}/v1/ranking"

    def map_cohere_rerank_params(
        self,
        non_default_params: dict | None,
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, Any]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:
        """
        Keep Cohere's top_n as-is instead of mapping it to top_k.

        The native /v1/ranking endpoint rejects top_k, so top_n is applied
        client-side after the response is converted.
        """
        optional_params = super().map_cohere_rerank_params(
            non_default_params=non_default_params,
            model=model,
            drop_params=drop_params,
            query=query,
            documents=documents,
            custom_llm_provider=custom_llm_provider,
            top_n=None,  # do not map top_n -> top_k for /v1/ranking
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            max_tokens_per_doc=max_tokens_per_doc,
            instruction=instruction,
        )
        # /v1/ranking rejects top_k even when passed as a provider-specific param
        optional_params.pop("top_k", None)
        if top_n is not None:
            optional_params["top_n"] = top_n
        return optional_params

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        """
        Transform request, using clean model name without 'ranking/' prefix.

        top_n / top_k are stripped from the outgoing request: the native
        /v1/ranking endpoint accepts only model, query, passages, and
        truncate. top_n is stashed and applied client-side in
        transform_rerank_response.
        """
        top_n = optional_rerank_params.get("top_n")
        if top_n is not None:
            if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
                raise ValueError(f"top_n must be a positive integer, got: {top_n!r}")
            self._client_side_top_n = top_n

        clean_model = self._get_clean_model_name(model)
        filtered_params = {k: v for k, v in optional_rerank_params.items() if k not in ("top_n", "top_k")}
        return super().transform_rerank_request(
            model=clean_model,
            optional_rerank_params=filtered_params,
            headers=headers,
            litellm_params=litellm_params,
        )

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Convert the native ranking response, then apply top_n client-side.

        /v1/ranking returns rankings sorted by relevance, but sort before
        truncating in case a server returns them unsorted.
        """
        response = super().transform_rerank_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        top_n = optional_params.get("top_n") or self._client_side_top_n
        if top_n is not None and response.results is not None and len(response.results) > top_n:
            response.results = sorted(
                response.results,
                key=lambda result: result["relevance_score"],
                reverse=True,
            )[:top_n]
        return response

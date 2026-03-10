"""
Transformation for NVIDIA NIM Ranking models that use /v1/ranking endpoint.

Use this by passing "nvidia_nim/ranking/<model>" to force the /v1/ranking endpoint.

Reference: https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy
"""

from typing import Dict, Optional

from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig


class NvidiaNimRankingConfig(NvidiaNimRerankConfig):
    """
    Configuration for NVIDIA NIM models that use the /v1/ranking endpoint.
    
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

    def _get_clean_model_name(self, model: str) -> str:
        """Strip 'nvidia_nim/' and 'ranking/' prefixes from model name."""
        # First strip nvidia_nim/ prefix if present
        if model.startswith("nvidia_nim/"):
            model = model[len("nvidia_nim/"):]
        # Then strip ranking/ prefix if present
        if model.startswith("ranking/"):
            model = model[len("ranking/"):]
        return model

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
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

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        """
        Transform request, using clean model name without 'ranking/' prefix.
        """
        clean_model = self._get_clean_model_name(model)
        return super().transform_rerank_request(
            model=clean_model,
            optional_rerank_params=optional_rerank_params,
            headers=headers,
        )


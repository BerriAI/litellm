"""
Common utilities for NVIDIA NIM rerank provider.
"""


def get_nvidia_nim_rerank_config(model: str):
    """
    Get the appropriate NVIDIA NIM rerank config based on the model.
    
    Args:
        model: The model string (e.g., "nvidia/llama-3.2-nv-rerankqa-1b-v2" or "ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2")
    
    Returns:
        NvidiaNimRankingConfig if model starts with "ranking/", else NvidiaNimRerankConfig
    
    Example:
        - "ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2" -> NvidiaNimRankingConfig
        - "nvidia/llama-3.2-nv-rerankqa-1b-v2" -> NvidiaNimRerankConfig
    """
    from litellm.llms.nvidia_nim.rerank.ranking_transformation import (
        NvidiaNimRankingConfig,
    )
    from litellm.llms.nvidia_nim.rerank.transformation import NvidiaNimRerankConfig

    if model.startswith("ranking/"):
        return NvidiaNimRankingConfig()
    return NvidiaNimRerankConfig()


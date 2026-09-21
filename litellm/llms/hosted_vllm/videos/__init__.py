from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

from .transformation import HostedVLLMVideoConfig

__all__ = ("HostedVLLMVideoConfig",)


def get_hosted_vllm_video_config(model: str | None) -> BaseVideoConfig:
    return HostedVLLMVideoConfig()

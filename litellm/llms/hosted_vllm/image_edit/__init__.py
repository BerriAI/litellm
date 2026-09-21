from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import HostedVLLMImageEditConfig

__all__ = ("HostedVLLMImageEditConfig",)


def get_hosted_vllm_image_edit_config(model: str) -> BaseImageEditConfig:
    return HostedVLLMImageEditConfig()

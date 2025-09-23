from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import AzureFoundryFluxImageEditConfig

__all__ = ["AzureFoundryFluxImageEditConfig"]


def get_azure_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    model = model.lower()
    model = model.replace("-", "")
    model = model.replace("_", "")
    if model == "" or "flux" in model:  # empty model is flux
        return AzureFoundryFluxImageEditConfig()
    else:
        raise ValueError(f"Model {model} is not supported for Azure AI image editing.")

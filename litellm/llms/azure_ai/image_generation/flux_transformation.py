from typing import Optional

from litellm.llms.openai.image_generation import GPTImageGenerationConfig


class AzureFoundryFluxImageGenerationConfig(GPTImageGenerationConfig):
    """
    Azure Foundry flux image generation config

    From manual testing it follows the gpt-image-1 image generation config

    (Azure Foundry does not have any docs on supported params at the time of writing)

    From our test suite - following GPTImageGenerationConfig is working for this model
    """

    @staticmethod
    def get_flux2_image_generation_url(
        api_base: Optional[str],
        model: str,
        api_version: Optional[str],
    ) -> str:
        """
        Constructs the complete URL for Azure AI FLUX 2 image generation.

        FLUX 2 models on Azure AI use a different URL pattern than standard Azure OpenAI:
        - Standard: /openai/deployments/{model}/images/generations
        - FLUX 2: /providers/blackforestlabs/v1/flux-2-pro

        Args:
            api_base: Base URL (e.g., https://litellm-ci-cd-prod.services.ai.azure.com)
            model: Model name (e.g., flux.2-pro)
            api_version: API version (e.g., preview)

        Returns:
            Complete URL for the FLUX 2 image generation endpoint
        """
        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AI FLUX 2 image generation"
            )

        api_base = api_base.rstrip("/")
        api_version = api_version or "preview"

        # If the api_base already contains /providers/, it's already a complete path
        if "/providers/" in api_base:
            if "?" in api_base:
                return api_base
            return f"{api_base}?api-version={api_version}"

        # Construct the FLUX 2 provider path
        # Model name flux.2-pro maps to endpoint flux-2-pro
        return f"{api_base}/providers/blackforestlabs/v1/flux-2-pro?api-version={api_version}"

    @staticmethod
    def is_flux2_model(model: str) -> bool:
        """
        Check if the model is an Azure AI FLUX 2 model.

        Args:
            model: Model name (e.g., flux.2-pro, azure_ai/flux.2-pro)

        Returns:
            True if the model is a FLUX 2 model
        """
        model_lower = model.lower().replace(".", "-").replace("_", "-")
        return "flux-2" in model_lower or "flux2" in model_lower

from .flux_schnell_transformation import FalAIFluxSchnellConfig


class FalAIFluxDevConfig(FalAIFluxSchnellConfig):
    """
    Configuration for Fal AI Flux Dev model.

    Model endpoint: fal-ai/flux/dev
    Documentation: https://fal.ai/models/fal-ai/flux/dev
    """

    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/flux/dev"

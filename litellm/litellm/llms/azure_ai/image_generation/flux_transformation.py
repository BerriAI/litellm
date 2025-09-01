from litellm.llms.openai.image_generation import GPTImageGenerationConfig


class AzureFoundryFluxImageGenerationConfig(GPTImageGenerationConfig):
    """
    Azure Foundry flux image generation config

    From manual testing it follows the gpt-image-1 image generation config

    (Azure Foundry does not have any docs on supported params at the time of writing)

    From our test suite - following GPTImageGenerationConfig is working for this model
    """
    pass

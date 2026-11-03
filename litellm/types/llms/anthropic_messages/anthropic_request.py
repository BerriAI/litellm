from litellm.types.llms.base import LiteLLMBaseModel


class AnthropicMetadata(LiteLLMBaseModel):
    """
    Object with allowed fields for Anthropic API metadata

    https://docs.anthropic.com/en/api/messages#body-metadata-user-id
    """

    user_id: str | None = None

from typing import TYPE_CHECKING, Any

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAnthropicClaude3MessagesConfig(AnthropicMessagesConfig):
    """
    Call Claude model family in the /v1/messages API spec
    """

    pass

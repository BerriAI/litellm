"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""
from typing import TYPE_CHECKING, Union

from litellm import CustomLLM

from .transformation import LiteLLMResponsesTransformationHandler

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper, ModelResponse


class ResponsesToCompletionBridgeHandler(CustomLLM):
    def completion(
        self, *args, **kwargs
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        from litellm import mock_completion

        return mock_completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )


responses_api_bridge = ResponsesToCompletionBridgeHandler()

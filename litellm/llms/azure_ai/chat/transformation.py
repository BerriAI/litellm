from typing import List

from litellm.llms.OpenAI.openai import OpenAIConfig
from litellm.llms.prompt_templates.common_utils import convert_content_list_to_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderField


class AzureAIStudioConfig(OpenAIConfig):
    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Azure AI Studio API Key.",
                field_value="zEJ...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Azure AI Studio API Base.",
                field_value="https://Mistral-serverless.",
            ),
        ]

    def _transform_messages(self, messages: List[AllMessageValues]) -> List:
        for message in messages:
            texts = convert_content_list_to_str(message=message)
            if texts:
                message["content"] = texts
        return messages

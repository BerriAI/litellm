"""
Translates from OpenAI's `/v1/chat/completions` to Databricks' `/chat/completions`
"""

from typing import List, Optional, Union

from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    strip_name_from_messages,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderField

from ...openai_like.chat.transformation import OpenAILikeChatConfig
from ..exceptions import DatabricksError


class DatabricksConfig(OpenAILikeChatConfig):
    """
    Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    n: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop: Optional[Union[List[str], str]] = None,
        n: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Databricks API Key.",
                field_value="dapi...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Databricks API Base.",
                field_value="https://adb-..",
            ),
        ]

    def get_supported_openai_params(self, model: Optional[str] = None) -> list:
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "response_format",
        ]

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Databricks doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        """
        Databricks does not support:
        - content in list format.
        - 'name' in user message.
        """
        new_messages = []
        for idx, message in enumerate(messages):
            if isinstance(message, BaseModel):
                _message = message.model_dump(exclude_none=True)
            else:
                _message = message
            new_messages.append(_message)
        new_messages = handle_messages_with_content_list_to_str_conversion(new_messages)
        new_messages = strip_name_from_messages(new_messages)
        return super()._transform_messages(messages=new_messages, model=model)

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base = api_base or litellm.api_base or get_secret_str("DATABRICKS_API_BASE")

        if api_base is None:
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client = WorkspaceClient()

                api_base = (
                    api_base or f"{databricks_client.config.host}/serving-endpoints"
                )
            except ImportError:
                raise DatabricksError(
                    status_code=400,
                    message=(
                        "If the Databricks base URL and API key are not set, the databricks-sdk "
                        "Python library must be installed. Please install the databricks-sdk, set "
                        "{LLM_PROVIDER}_API_BASE and {LLM_PROVIDER}_API_KEY environment variables, "
                        "or provide the base URL and API key as arguments."
                    ),
                )

        return super().get_complete_url(
            api_base=api_base,
            model=model,
            optional_params=optional_params,
            stream=stream,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = headers or {"Content-Type": "application/json"}

        api_key = (
            api_key
            or litellm.api_key  # for databricks we check in get_llm_provider and pass in the api key from there
            or litellm.databricks_key
            or get_secret_str("DATABRICKS_API_KEY")
        )

        if api_key is None:
            # try using dbrx sdk to get cred from env
            try:
                from databricks.sdk import WorkspaceClient

                databricks_client = WorkspaceClient()

                databricks_auth_headers: dict[str, str] = (
                    databricks_client.config.authenticate()
                )
                headers = {**databricks_auth_headers, **headers}

                return headers
            except ImportError:
                raise DatabricksError(
                    status_code=400,
                    message=(
                        "If the Databricks base URL and API key are not set, the databricks-sdk "
                        "Python library must be installed. Please install the databricks-sdk, set "
                        "{LLM_PROVIDER}_API_BASE and {LLM_PROVIDER}_API_KEY environment variables, "
                        "or provide the base URL and API key as arguments."
                    ),
                )

        headers.update({"Authorization": "Bearer {}".format(api_key)})

        return headers

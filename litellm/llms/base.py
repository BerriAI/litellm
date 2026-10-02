## This is a template base class to be used for adding new LLM providers via API calls
from typing import TYPE_CHECKING, Any, Union

import httpx

import litellm
from litellm.llms.custom_httpx.upstream_response import install_capture_hook, with_capture_hooks

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.types.utils import ModelResponse, TextCompletionResponse


class BaseLLM:
    _client_session: httpx.Client | None = None
    _owns_client_session: bool = True
    _owns_aclient_session: bool = True

    def process_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: "ModelResponse",
        stream: bool,
        logging_obj: "LiteLLMLoggingObj",
        optional_params: dict,
        api_key: str,
        data: dict | str,
        messages: list,
        print_verbose,
        encoding,
    ) -> Union["ModelResponse", "CustomStreamWrapper"]:
        """
        Helper function to process the response across sync + async completion calls
        """
        return model_response

    def process_text_completion_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: "TextCompletionResponse",
        stream: bool,
        logging_obj: "LiteLLMLoggingObj",
        optional_params: dict,
        api_key: str,
        data: dict | str,
        messages: list,
        print_verbose,
        encoding,
    ) -> Union["TextCompletionResponse", "CustomStreamWrapper"]:
        """
        Helper function to process the response across sync + async completion calls
        """
        return model_response

    def create_client_session(self) -> httpx.Client:
        if litellm.client_session:
            self._owns_client_session = False
            install_capture_hook(litellm.client_session)
            return litellm.client_session

        self._owns_client_session = True
        return httpx.Client(event_hooks=with_capture_hooks(None, is_async=False))

    def create_aclient_session(self) -> httpx.AsyncClient:
        if litellm.aclient_session:
            self._owns_aclient_session = False
            install_capture_hook(litellm.aclient_session)
            return litellm.aclient_session

        self._owns_aclient_session = True
        return httpx.AsyncClient(event_hooks=with_capture_hooks(None, is_async=True))

    def __exit__(self):
        if self._owns_client_session and hasattr(self, "_client_session") and self._client_session is not None:
            self._client_session.close()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_aclient_session and hasattr(self, "_aclient_session"):
            await self._aclient_session.aclose()

    def validate_environment(self, *args, **kwargs) -> Any | None:  # set up the environment required to run the model
        return None

    def completion(self, *args, **kwargs) -> Any:  # logic for parsing in - calling - parsing out model completion calls
        return None

    def embedding(self, *args, **kwargs) -> Any:  # logic for parsing in - calling - parsing out model embedding calls
        return None

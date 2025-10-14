"""
Translates from OpenAI's `/v1/embeddings` to IBM's `/text/embeddings` route.
"""

from typing import Optional, TYPE_CHECKING

import httpx

from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

# Type checking block for optional imports
if TYPE_CHECKING:
    from gen_ai_hub.proxy.gen_ai_hub_proxy import (
        GenAIHubProxyClient,
        temporary_headers_addition,
    )

# Try to import the optional module
try:
    from gen_ai_hub.proxy.gen_ai_hub_proxy import (
        GenAIHubProxyClient,
        temporary_headers_addition,
    )

    _gen_ai_hub_import_error = None
except ImportError as err:
    GenAIHubProxyClient = None  # type: ignore
    _gen_ai_hub_import_error = err


from ..chat.handler import OptionalDependencyError, GenAIHubOrchestrationError


class GenAIHubEmbeddingConfig(BaseEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self._client: Optional["GenAIHubProxyClient"] = None
        self._orchestration_client = None

    def _ensure_gen_ai_hub_installed(self) -> None:
        """Ensure the gen-ai-hub package is available."""
        if _gen_ai_hub_import_error is not None:
            raise OptionalDependencyError(
                "The gen-ai-hub package is required for this functionality. "
                "Please install it with: pip install sap-ai-sdk-gen[all]"
            ) from _gen_ai_hub_import_error

    def get_error_class(self, error_message, status_code, headers):
        return GenAIHubOrchestrationError(status_code, error_message)

    def get_supported_openai_params(self, model: str) -> list:
        if "text-embedding-3" in model:
            return ["encoding_format", "dimensions"]
        else:
            return [
                "encoding_format",
            ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    @property
    def proxy_client(self) -> "GenAIHubProxyClient":
        """Initialize and get the orchestration client."""
        self._ensure_gen_ai_hub_installed()
        if (
            GenAIHubProxyClient is None
        ):  # This should never happen due to _ensure_dependency
            raise RuntimeError(
                "GenAIHubProxyClient is None despite passing dependency check"
            )
        if not self._client:
            self._client = GenAIHubProxyClient()
        return self._client

    def _add_api_version_to_url(self, url: str, api_version: str) -> str:
        from gen_ai_hub.proxy.native.openai.clients import DEFAULT_API_VERSION

        api_version = (
            DEFAULT_API_VERSION
            if not api_version or api_version is None or api_version == "None"
            else api_version
        )
        return url.rstrip("/") + f"?api-version={api_version}"

    def validate_environment(self, headers: dict, *args, **kwargs) -> dict:
        self._ensure_gen_ai_hub_installed()
        with temporary_headers_addition(headers):
            return {**self.proxy_client.request_header}

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        deployment = self.proxy_client.select_deployment(model_name=model)
        url = deployment.url.rstrip("/") + "/embeddings"
        ## add api version
        url = self._add_api_version_to_url(
            url=url, api_version=optional_params.pop("api_version", "None")
        )
        return url

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return {"input": input, "model": model}

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return EmbeddingResponse.model_validate(raw_response.json())

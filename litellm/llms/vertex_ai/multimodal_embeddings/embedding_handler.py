import json
from typing import Final, Literal

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexAIError,
    VertexLLM,
)
from litellm.types.utils import EmbeddingResponse

from .transformation import VertexAIMultimodalEmbeddingConfig

vertex_multimodal_embedding_handler: Final = VertexAIMultimodalEmbeddingConfig()


class VertexMultimodalEmbedding(VertexLLM):
    def __init__(self) -> None:
        super().__init__()
        self.SUPPORTED_MULTIMODAL_EMBEDDING_MODELS = [
            "multimodalembedding",
            "multimodalembedding@001",
        ]

    def multimodal_embedding(
        self,
        model: str,
        input: list | str,
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        api_base: str | None = None,
        headers: dict = {},
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding: bool | None = False,
        timeout=300,
        client=None,
    ) -> EmbeddingResponse:
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            auth_header=_auth_header,
            gemini_api_key=api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=None,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="embedding",
        )

        if client is None:
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout: Final = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)
        else:
            sync_handler = client

        request_data: Final = vertex_multimodal_embedding_handler.transform_embedding_request(
            model, input, optional_params, headers
        )

        headers = vertex_multimodal_embedding_handler.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params=optional_params,
            api_key=auth_header,
            api_base=api_base,
            litellm_params=litellm_params,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": url,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self.async_multimodal_embedding(
                model=model,
                api_base=url,
                data=request_data,
                timeout=timeout,
                headers=headers,
                client=client,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                api_key=api_key,
            )

        response: Final = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        return vertex_multimodal_embedding_handler.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    async def async_multimodal_embedding(
        self,
        model: str,
        api_base: str,
        optional_params: dict,
        litellm_params: dict,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: float | httpx.Timeout | None,
        logging_obj: LiteLLMLoggingObj,
        headers={},
        client: AsyncHTTPHandler | None = None,
        api_key: str | None = None,
    ) -> EmbeddingResponse:
        if client is None:
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            client = client

        try:
            response: Final = await client.post(api_base, headers=headers, json=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code: Final = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        return vertex_multimodal_embedding_handler.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

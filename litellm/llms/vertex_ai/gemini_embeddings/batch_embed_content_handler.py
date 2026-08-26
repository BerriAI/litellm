"""
Google AI Studio /batchEmbedContents Embeddings Endpoint
"""

import json
from typing import Any, Final, Literal

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.vertex_ai import (
    GeminiEmbeddingInput,
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)
from litellm.types.utils import EmbeddingResponse

from ..gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from .batch_embed_content_transformation import (
    _is_file_reference,
    process_embed_content_response,
    process_response,
    transform_openai_input_gemini_content,
    transform_openai_input_gemini_embed_content,
)


class GoogleBatchEmbeddings(VertexLLM):
    @staticmethod
    def _flatten_and_detect_file_refs(
        input: GeminiEmbeddingInput,
    ) -> tuple[list[str], bool]:
        """Flatten nested input lists and detect file references."""
        input_list: Final = [input] if isinstance(input, str) else input
        flat_elements: Final = [
            e for item in input_list for e in (item if isinstance(item, list) else [item]) if isinstance(e, str)
        ]
        has_file_refs: Final = any(_is_file_reference(e) for e in flat_elements)
        return flat_elements, has_file_refs

    def _resolve_file_references(
        self,
        input: GeminiEmbeddingInput,
        api_key: str,
        sync_handler: HTTPHandler,
    ) -> dict[str, dict[str, str]]:
        """
        Resolve Gemini file references (files/...) to get mime_type and uri.

        Args:
            input: GeminiEmbeddingInput that may contain file references
            api_key: Gemini API key
            sync_handler: HTTP client

        Returns:
            Dict mapping file name to {mime_type, uri}
        """
        input_list: Final = [input] if isinstance(input, str) else input
        resolved_files: Final[dict[str, dict[str, str]]] = {}

        for element in input_list:
            if isinstance(element, str) and _is_file_reference(element):
                url = f"https://generativelanguage.googleapis.com/v1beta/{element}"
                headers = {"x-goog-api-key": api_key}
                response = sync_handler.get(url=url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Error fetching file {element}: {response.status_code} {response.text}")

                file_data = response.json()
                resolved_files[element] = {
                    "mime_type": file_data.get("mimeType", ""),
                    "uri": file_data.get("uri", element),
                }

        return resolved_files

    async def _async_resolve_file_references(
        self,
        input: GeminiEmbeddingInput,
        api_key: str,
        async_handler: AsyncHTTPHandler,
    ) -> dict[str, dict[str, str]]:
        """
        Async version of _resolve_file_references.

        Args:
            input: GeminiEmbeddingInput that may contain file references
            api_key: Gemini API key
            async_handler: Async HTTP client

        Returns:
            Dict mapping file name to {mime_type, uri}
        """
        input_list: Final = [input] if isinstance(input, str) else input
        resolved_files: Final[dict[str, dict[str, str]]] = {}

        for element in input_list:
            if isinstance(element, str) and _is_file_reference(element):
                url = f"https://generativelanguage.googleapis.com/v1beta/{element}"
                headers = {"x-goog-api-key": api_key}
                response = await async_handler.get(url=url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Error fetching file {element}: {response.status_code} {response.text}")

                file_data = response.json()
                resolved_files[element] = {
                    "mime_type": file_data.get("mimeType", ""),
                    "uri": file_data.get("uri", element),
                }

        return resolved_files

    def batch_embeddings(
        self,
        model: str,
        input: GeminiEmbeddingInput,
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        logging_obj: Any,
        api_key: str | None = None,
        api_base: str | None = None,
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding: bool | None = False,
        timeout=300,
        client=None,
        extra_headers: dict | None = None,
    ) -> EmbeddingResponse:
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
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

        optional_params = optional_params or {}

        use_embed_content: Final = custom_llm_provider == "vertex_ai"
        mode: Literal["embedding", "batch_embedding"]
        if use_embed_content:
            mode = "embedding"
        else:
            mode = "batch_embedding"

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
            mode=mode,
        )

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
        }
        if auth_header is not None:
            if isinstance(auth_header, dict):
                headers.update(auth_header)
            else:
                headers["Authorization"] = f"Bearer {auth_header}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if aembedding is True:
            return self.async_batch_embeddings(
                model=model,
                api_base=api_base,
                url=url,
                data=None,
                model_response=model_response,
                timeout=timeout,
                headers=headers,
                input=input,
                use_embed_content=use_embed_content,
                api_key=api_key,
                optional_params=optional_params,
                logging_obj=logging_obj,
            )

        ### TRANSFORMATION (sync path) ###
        request_data: Any
        if use_embed_content:
            resolved_files = {}
            if api_key:
                resolved_files = self._resolve_file_references(input=input, api_key=api_key, sync_handler=sync_handler)
            request_data = transform_openai_input_gemini_embed_content(
                input=input,
                model=model,
                optional_params=optional_params,
                resolved_files=resolved_files,
            )
        else:
            flat_elements, has_file_refs = self._flatten_and_detect_file_refs(input)
            if has_file_refs and not api_key:
                raise ValueError(
                    "An API key is required to resolve Gemini file references (files/...). "
                    "Pass api_key= or set GEMINI_API_KEY."
                )
            resolved_files = {}
            if api_key and has_file_refs:
                resolved_files = self._resolve_file_references(
                    input=flat_elements, api_key=api_key, sync_handler=sync_handler
                )
            request_data = transform_openai_input_gemini_content(
                input=input,
                model=model,
                optional_params=optional_params,
                resolved_files=resolved_files,
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

        response: Final = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response: Final = response.json()

        if use_embed_content:
            return process_embed_content_response(
                input=input,
                model_response=model_response,
                model=model,
                response_json=_json_response,
                resolved_files=resolved_files,
            )
        else:
            _predictions: Final = VertexAIBatchEmbeddingsResponseObject(**_json_response)
            return process_response(
                model=model,
                model_response=model_response,
                _predictions=_predictions,
                input=input,
            )

    async def async_batch_embeddings(
        self,
        model: str,
        api_base: str | None,
        url: str,
        data: VertexAIBatchEmbeddingsRequestBody | dict | None,
        model_response: EmbeddingResponse,
        input: GeminiEmbeddingInput,
        timeout: float | httpx.Timeout | None,
        headers={},
        client: AsyncHTTPHandler | None = None,
        use_embed_content: bool = False,
        api_key: str | None = None,
        optional_params: dict | None = None,
        logging_obj: Any | None = None,
    ) -> EmbeddingResponse:
        if client is None:
            _params: Final = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout: Final = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            async_handler: AsyncHTTPHandler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            async_handler = client

        ### TRANSFORMATION (async path) ###
        if use_embed_content:
            resolved_files = {}
            if api_key:
                resolved_files = await self._async_resolve_file_references(
                    input=input, api_key=api_key, async_handler=async_handler
                )
            data = transform_openai_input_gemini_embed_content(
                input=input,
                model=model,
                optional_params=optional_params or {},
                resolved_files=resolved_files,
            )
        else:
            flat_elements, has_file_refs = self._flatten_and_detect_file_refs(input)
            if has_file_refs and not api_key:
                raise ValueError(
                    "An API key is required to resolve Gemini file references (files/...). "
                    "Pass api_key= or set GEMINI_API_KEY."
                )
            resolved_files = {}
            if api_key and has_file_refs:
                resolved_files = await self._async_resolve_file_references(
                    input=flat_elements, api_key=api_key, async_handler=async_handler
                )
            data = transform_openai_input_gemini_content(
                input=input,
                model=model,
                optional_params=optional_params or {},
                resolved_files=resolved_files,
            )

        ## LOGGING
        if logging_obj is not None:
            logging_obj.pre_call(
                input=input,
                api_key="",
                additional_args={
                    "complete_input_dict": data,
                    "api_base": url,
                    "headers": headers,
                },
            )

        response: Final = await async_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response: Final = response.json()

        if use_embed_content:
            return process_embed_content_response(
                input=input,
                model_response=model_response,
                model=model,
                response_json=_json_response,
                resolved_files=resolved_files,
            )
        else:
            _predictions: Final = VertexAIBatchEmbeddingsResponseObject(**_json_response)
            return process_response(
                model=model,
                model_response=model_response,
                _predictions=_predictions,
                input=input,
            )

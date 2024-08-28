"""
Google AI Studio Embeddings Endpoint
"""

import json
from typing import Literal, Optional, Union

import httpx

import litellm
from litellm import EmbeddingResponse
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.vertex_ai import (
    VertexAITextEmbeddingsRequestBody,
    VertexAITextEmbeddingsResponseObject,
)
from litellm.types.utils import Embedding
from litellm.utils import get_formatted_prompt

from .embeddings_transformation import transform_openai_input_gemini_content
from .vertex_and_google_ai_studio_gemini import VertexLLM


class GoogleEmbeddings(VertexLLM):
    def text_embeddings(
        self,
        model: str,
        input: Union[list, str],
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        logging_obj=None,
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding=False,
        timeout=300,
        client=None,
    ) -> EmbeddingResponse:

        auth_header, url = self._get_token_and_url(
            model=model,
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
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        optional_params = optional_params or {}

        ### TRANSFORMATION ###
        content = transform_openai_input_gemini_content(input=input)

        request_data: VertexAITextEmbeddingsRequestBody = {
            "content": content,
            **optional_params,
        }

        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

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
            pass

        response = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        _predictions = VertexAITextEmbeddingsResponseObject(**_json_response)  # type: ignore

        model_response.data = [
            Embedding(
                embedding=_predictions["embedding"]["values"],
                index=0,
                object="embedding",
            )
        ]

        model_response.model = model

        input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
        prompt_tokens = litellm.token_counter(model=model, text=input_text)
        model_response.usage = litellm.Usage(
            prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
        )

        return model_response

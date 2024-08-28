"""
Google AI Studio Embeddings Endpoint
"""

import json
from typing import Literal, Optional, Union

import httpx

import litellm
from litellm import EmbeddingResponse
from litellm.llms.custom_httpx.http_handler import HTTPHandler

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
        return model_response
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

        # request_data = VertexMultimodalEmbeddingRequest()

        # if "instances" in optional_params:
        #     request_data["instances"] = optional_params["instances"]
        # elif isinstance(input, list):
        #     request_data["instances"] = input
        # else:
        #     # construct instances
        #     vertex_request_instance = Instance(**optional_params)

        #     if isinstance(input, str):
        #         vertex_request_instance["text"] = input

        #     request_data["instances"] = [vertex_request_instance]

        # headers = {
        #     "Content-Type": "application/json; charset=utf-8",
        #     "Authorization": f"Bearer {auth_header}",
        # }

        # ## LOGGING
        # logging_obj.pre_call(
        #     input=input,
        #     api_key="",
        #     additional_args={
        #         "complete_input_dict": request_data,
        #         "api_base": url,
        #         "headers": headers,
        #     },
        # )

        # if aembedding is True:
        #     pass

        # response = sync_handler.post(
        #     url=url,
        #     headers=headers,
        #     data=json.dumps(request_data),
        # )

        # if response.status_code != 200:
        #     raise Exception(f"Error: {response.status_code} {response.text}")

        # _json_response = response.json()
        # if "predictions" not in _json_response:
        #     raise litellm.InternalServerError(
        #         message=f"embedding response does not contain 'predictions', got {_json_response}",
        #         llm_provider="vertex_ai",
        #         model=model,
        #     )
        # _predictions = _json_response["predictions"]

        # model_response.data = _predictions
        # model_response.model = model

        # return model_response

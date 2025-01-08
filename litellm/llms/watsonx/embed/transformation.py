"""
Translates from OpenAI's `/v1/embeddings` to IBM's `/text/embeddings` route.
"""

from typing import Optional

import httpx

from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.llms.watsonx import WatsonXAIEndpoint
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import IBMWatsonXMixin, WatsonXAIError, _get_api_params


class IBMWatsonXEmbeddingConfig(IBMWatsonXMixin, BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        watsonx_api_params = _get_api_params(params=optional_params)
        project_id = watsonx_api_params["project_id"]
        if not project_id:
            raise ValueError("project_id is required")
        return {
            "inputs": input,
            "model_id": model,
            "project_id": project_id,
            "parameters": optional_params,
        }

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        url = self._get_base_url(api_base=api_base)
        endpoint = WatsonXAIEndpoint.EMBEDDINGS.value
        if model.startswith("deployment/"):
            # deployment models are passed in as 'deployment/<deployment_id>'
            if optional_params.get("space_id") is None:
                raise WatsonXAIError(
                    status_code=401,
                    message="Error: space_id is required for models called using the 'deployment/' endpoint. Pass in the space_id as a parameter or set it in the WX_SPACE_ID environment variable.",
                )
            deployment_id = "/".join(model.split("/")[1:])
            endpoint = endpoint.format(deployment_id=deployment_id)
        url = url.rstrip("/") + endpoint

        ## add api version
        url = self._add_api_version_to_url(
            url=url, api_version=optional_params.pop("api_version", None)
        )
        return url

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
        logging_obj.post_call(
            original_response=raw_response.text,
        )
        json_resp = raw_response.json()
        if model_response is None:
            model_response = EmbeddingResponse(model=json_resp.get("model_id", None))
        results = json_resp.get("results", [])
        embedding_response = []
        for idx, result in enumerate(results):
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": result["embedding"],
                }
            )
        model_response.object = "list"
        model_response.data = embedding_response
        input_tokens = json_resp.get("input_token_count", 0)
        setattr(
            model_response,
            "usage",
            Usage(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
            ),
        )
        return model_response

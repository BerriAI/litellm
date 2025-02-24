import json
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.openai import Batch, CreateBatchRequest
from litellm.types.llms.vertex_ai import VertexAIBatchPredictionJob

from .transformation import VertexAIBatchTransformation


class VertexAIBatchPrediction(VertexLLM):
    def __init__(self, gcs_bucket_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcs_bucket_name = gcs_bucket_name

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[Batch, Coroutine[Any, Any, Batch]]:

        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        vertex_batch_request: VertexAIBatchPredictionJob = (
            VertexAIBatchTransformation.transform_openai_batch_request_to_vertex_ai_batch_request(
                request=create_batch_data
            )
        )

        if _is_async is True:
            return self._async_create_batch(
                vertex_batch_request=vertex_batch_request,
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_create_batch(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        api_base: str,
        headers: Dict[str, str],
    ) -> Batch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def create_vertex_url(
        self,
        vertex_location: str,
        vertex_project: str,
    ) -> str:
        """Return the base url for the vertex garden models"""
        #  POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/batchPredictionJobs
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/batchPredictionJobs"

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        vertex_credentials: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[Batch, Coroutine[Any, Any, Batch]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        # Append batch_id to the URL
        default_api_base = f"{default_api_base}/{batch_id}"

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_retrieve_batch(
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.get(
            url=api_base,
            headers=headers,
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_retrieve_batch(
        self,
        api_base: str,
        headers: Dict[str, str],
    ) -> Batch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.get(
            url=api_base,
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

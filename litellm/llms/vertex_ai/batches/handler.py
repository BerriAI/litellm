import json
from collections.abc import Coroutine
from typing import Any, Final

import httpx

import litellm
from litellm.litellm_core_utils.url_utils import (
    async_safe_get,
    encode_url_path_segment,
    safe_get,
)
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError, get_vertex_base_url
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    VertexAIBatchPredictionJob,
)
from litellm.types.utils import LiteLLMBatch

from .transformation import VertexAIBatchTransformation


class VertexAIBatchPrediction(VertexLLM):
    def __init__(self, gcs_bucket_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcs_bucket_name = gcs_bucket_name

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_base: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
    ) -> LiteLLMBatch | Coroutine[Any, Any, LiteLLMBatch]:
        sync_handler: Final = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base: Final = self.create_vertex_batch_url(
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
            model=None,
            vertex_project=vertex_project or project_id,
            vertex_location=vertex_location or "us-central1",
            vertex_api_version="v1",
        )

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        vertex_batch_request: Final[VertexAIBatchPredictionJob] = (
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

        response: Final = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )

        _json_response: Final = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_create_batch(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        api_base: str,
        headers: dict[str, str],
    ) -> LiteLLMBatch:
        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        try:
            response: Final = await client.post(
                url=api_base,
                headers=headers,
                data=json.dumps(vertex_batch_request),
            )
        except httpx.HTTPStatusError as e:
            error_body: Final = e.response.text
            litellm.verbose_logger.error(
                "Vertex AI batch create failed: status=%s, body=%s",
                e.response.status_code,
                error_body[:1000],
            )
            raise

        _json_response: Final = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def create_vertex_batch_url(
        self,
        vertex_location: str,
        vertex_project: str,
    ) -> str:
        """Return the base url for the vertex garden models"""
        #  POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/batchPredictionJobs
        base_url: Final = get_vertex_base_url(vertex_location)
        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/batchPredictionJobs"

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
        logging_obj: Any | None = None,
    ) -> LiteLLMBatch | Coroutine[Any, Any, LiteLLMBatch]:
        sync_handler: Final = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        # Append batch_id to the URL
        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        default_api_base = f"{default_api_base}/{encoded_batch_id}"

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
            model=None,
            vertex_project=vertex_project or project_id,
            vertex_location=vertex_location or "us-central1",
            vertex_api_version="v1",
        )

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_retrieve_batch(
                api_base=api_base,
                headers=headers,
                logging_obj=logging_obj,
            )

        # Log the request using logging_obj if available
        if logging_obj is not None:
            from litellm.litellm_core_utils.litellm_logging import Logging

            if isinstance(logging_obj, Logging):
                logging_obj.pre_call(
                    input="",
                    api_key="",
                    additional_args={
                        "complete_input_dict": {},
                        "api_base": api_base,
                        "headers": headers,
                        "request_str": (
                            f"\nGET Request Sent from LiteLLM:\n"
                            f"curl -X GET \\\n"
                            f"{api_base} \\\n"
                            f"-H 'Authorization: Bearer ***REDACTED***' \\\n"
                            f"-H 'Content-Type: application/json; charset=utf-8'\n"
                        ),
                    },
                )

        # ``api_base`` here can come from caller-supplied request kwargs
        # (clientside override). Wrap the fetch in ``safe_get`` so DNS
        # rebind / private / cloud-metadata targets are rejected; the
        # proxy auth gate already blocks malicious clientside ``api_base``
        # at the boundary — this is defense-in-depth for SDK callers.
        response: Final = safe_get(
            sync_handler,
            api_base,
            headers=headers,
        )

        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        _json_response: Final = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_retrieve_batch(
        self,
        api_base: str,
        headers: dict[str, str],
        logging_obj: Any | None = None,
    ) -> LiteLLMBatch:
        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )

        # Log the request using logging_obj if available
        if logging_obj is not None:
            from litellm.litellm_core_utils.litellm_logging import Logging

            if isinstance(logging_obj, Logging):
                logging_obj.pre_call(
                    input="",
                    api_key="",
                    additional_args={
                        "complete_input_dict": {},
                        "api_base": api_base,
                        "headers": headers,
                        "request_str": (
                            f"\nGET Request Sent from LiteLLM:\n"
                            f"curl -X GET \\\n"
                            f"{api_base} \\\n"
                            f"-H 'Authorization: Bearer ***REDACTED***' \\\n"
                            f"-H 'Content-Type: application/json; charset=utf-8'\n"
                        ),
                    },
                )

        # Mirror the sync path: ``api_base`` may come from caller-supplied
        # request kwargs, so wrap the fetch in ``async_safe_get`` to reject
        # DNS-rebind / private / cloud-metadata targets. Defense-in-depth
        # behind the proxy auth gate's clientside ``api_base`` check.
        response: Final = await async_safe_get(
            client,
            api_base,
            headers=headers,
        )
        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        _json_response: Final = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def list_batches(
        self,
        _is_async: bool,
        after: str | None,
        limit: int | None,
        api_base: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
    ):
        sync_handler: Final = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base: Final = self.create_vertex_batch_url(
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

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        params: Final[dict[str, Any]] = {}
        if limit is not None:
            params["pageSize"] = str(limit)
        if after is not None:
            params["pageToken"] = after

        if _is_async is True:
            return self._async_list_batches(
                api_base=api_base,
                headers=headers,
                params=params,
            )

        response: Final = sync_handler.get(
            url=api_base,
            headers=headers,
            params=params,
        )

        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        _json_response: Final = response.json()
        vertex_batch_response: Final = (
            VertexAIBatchTransformation.transform_vertex_ai_batch_list_response_to_openai_list_response(
                response=_json_response
            )
        )
        return vertex_batch_response

    async def _async_list_batches(
        self,
        api_base: str,
        headers: dict[str, str],
        params: dict[str, Any],
    ):
        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response: Final = await client.get(
            url=api_base,
            headers=headers,
            params=params,
        )
        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        _json_response: Final = response.json()
        vertex_batch_response: Final = (
            VertexAIBatchTransformation.transform_vertex_ai_batch_list_response_to_openai_list_response(
                response=_json_response
            )
        )
        return vertex_batch_response

    def cancel_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: str | None,
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None,
        vertex_project: str | None,
        vertex_location: str | None,
        timeout: float | httpx.Timeout,
        max_retries: int | None,
    ) -> LiteLLMBatch | Coroutine[Any, Any, LiteLLMBatch]:
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base: Final = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        encoded_batch_id: Final = encode_url_path_segment(batch_id, field_name="batch_id")
        retrieve_api_base_default: Final = f"{default_api_base}/{encoded_batch_id}"
        cancel_api_base_default: Final = f"{retrieve_api_base_default}:cancel"

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="cancel",
            stream=None,
            auth_header=None,
            url=cancel_api_base_default,
            model=None,
            vertex_project=vertex_project or project_id,
            vertex_location=vertex_location or "us-central1",
            vertex_api_version="v1",
        )

        if api_base.endswith(":cancel"):
            retrieve_api_base = api_base.removesuffix(":cancel")
        else:
            retrieve_api_base = api_base.rsplit(":cancel", 1)[0].rstrip("/")

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_cancel_batch(
                api_base=api_base,
                retrieve_api_base=retrieve_api_base,
                headers=headers,
                timeout=timeout,
            )

        sync_handler: Final = _get_httpx_client()
        try:
            sync_handler.post(
                url=api_base,
                headers=headers,
                data=json.dumps({}),
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            litellm.verbose_logger.error(
                "Vertex AI batch cancel failed: status=%s, body=%s",
                e.response.status_code,
                e.response.text[:1000],
            )
            raise

        # HTTPHandler.get() does not accept a timeout parameter
        retrieve_response: Final = sync_handler.get(
            url=retrieve_api_base,
            headers=headers,
        )
        if retrieve_response.status_code != 200:
            litellm.verbose_logger.error(
                "Vertex AI batch retrieve-after-cancel failed: status=%s, body=%s",
                retrieve_response.status_code,
                retrieve_response.text[:1000],
            )
            raise VertexAIError(
                status_code=retrieve_response.status_code,
                message=f"Error: {retrieve_response.status_code} {retrieve_response.text}",
            )

        _json_response: Final = retrieve_response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_cancel_batch(
        self,
        api_base: str,
        retrieve_api_base: str,
        headers: dict[str, str],
        timeout: float | httpx.Timeout = 600.0,
    ) -> LiteLLMBatch:
        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        try:
            await client.post(
                url=api_base,
                headers=headers,
                data=json.dumps({}),
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            litellm.verbose_logger.error(
                "Vertex AI batch cancel failed: status=%s, body=%s",
                e.response.status_code,
                e.response.text[:1000],
            )
            raise

        # AsyncHTTPHandler.get() does not accept a timeout parameter
        retrieve_response: Final = await client.get(
            url=retrieve_api_base,
            headers=headers,
        )
        if retrieve_response.status_code != 200:
            litellm.verbose_logger.error(
                "Vertex AI batch retrieve-after-cancel failed: status=%s, body=%s",
                retrieve_response.status_code,
                retrieve_response.text[:1000],
            )
            raise VertexAIError(
                status_code=retrieve_response.status_code,
                message=f"Error: {retrieve_response.status_code} {retrieve_response.text}",
            )

        _json_response: Final = retrieve_response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

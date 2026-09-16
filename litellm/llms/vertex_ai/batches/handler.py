import json
from collections.abc import Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlparse

import httpx
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.url_utils import (
    async_safe_get,
    encode_url_path_segment,
    safe_get,
)
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.common_utils import (
    VERTEX_CUSTOM_ENDPOINT_KEY_FIELD,
    VertexAIError,
    get_custom_endpoint_id_from_api_base,
    get_vertex_base_url,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.llms.vertex_ai.vertex_llm_base import _graft_default_vertex_path
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    BatchDedicatedResources,
    UnmanagedContainerModel,
    VertexAIBatchPredictionJob,
    VertexBatchPredictionResponse,
)
from litellm.types.utils import LiteLLMBatch

from .transformation import VertexAIBatchTransformation

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _VertexBatchJsonSource(Protocol):
    """An HTTP response whose JSON body is a single Vertex AI batch prediction job."""

    def json(self) -> VertexBatchPredictionResponse: ...


class _VertexBatchListJsonSource(Protocol):
    """An HTTP response whose JSON body is a page of Vertex AI batch prediction jobs."""

    def json(self) -> dict[str, object]: ...


class _VertexBatchPayloadView(TypedDict):
    """Holds one decoded batch prediction job so the payload reads back typed."""

    payload: ReadOnly[VertexBatchPredictionResponse]


class _FetchedResponseView(TypedDict):
    """Holds one ``safe_get`` result so the response reads back as ``httpx.Response``."""

    response: ReadOnly[httpx.Response]


class _VertexOnlineDedicatedResources(TypedDict, total=False):
    """The dedicatedResources block on an online endpoint deployment; replica bounds are named
    min/max there, unlike the batch job's starting/max."""

    machineSpec: ReadOnly[Mapping[str, object]]
    minReplicaCount: ReadOnly[int]
    maxReplicaCount: ReadOnly[int]


class _VertexEndpointDeployedModel(TypedDict, total=False):
    model: ReadOnly[str]
    dedicatedResources: ReadOnly[_VertexOnlineDedicatedResources]


class _VertexEndpointResponse(TypedDict, total=False):
    deployedModels: ReadOnly[Sequence[_VertexEndpointDeployedModel]]


class _VertexEndpointPayloadView(TypedDict):
    """Holds one decoded GET endpoints/<id> response so the payload reads back typed."""

    payload: ReadOnly[_VertexEndpointResponse]


class _VertexModelResourceResponse(TypedDict, total=False):
    containerSpec: ReadOnly[Mapping[str, object]]


class _VertexModelResourcePayloadView(TypedDict):
    """Holds one decoded GET models/<id> response so the payload reads back typed."""

    payload: ReadOnly[_VertexModelResourceResponse]


def _gateway_api_base_or_none(api_base: str | None) -> str | None:
    """
    A deployment `api_base` whose path names a concrete Vertex resource (contains `/projects/`,
    e.g. the `.../endpoints/<id>:rawPredict` url configured for online inference) is not a Vertex
    API gateway; grafting `batchPredictionJobs` or resource-GET paths onto it can only produce
    urls Google answers with an HTML 404 (LIT-7386). Batch operations ignore it and use the real
    Vertex host; only a host-level or `/v1`-style gateway mount passes through.
    """
    if api_base and "/projects/" in urlparse(api_base).path:
        return None
    return api_base


def _vertex_batch_payload(response: _VertexBatchJsonSource) -> VertexBatchPredictionResponse:
    return response.json()


def _vertex_batch_list_payload(response: _VertexBatchListJsonSource) -> dict[str, object]:
    return response.json()


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
        custom_endpoint: bool | None = None,
    ) -> LiteLLMBatch | Coroutine[object, object, LiteLLMBatch]:
        sync_handler: Final = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        headers: Final = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        transformed_batch_request: Final[VertexAIBatchPredictionJob] = (
            VertexAIBatchTransformation.transform_openai_batch_request_to_vertex_ai_batch_request(
                request=create_batch_data,
                vertex_project=vertex_project or project_id,
                vertex_location=vertex_location or "us-central1",
            )
        )
        # The file-path endpoint id is caller-controlled and the job runs with the deployment's
        # credentials, so it must match the endpoint the deployment's own api_base names.
        job_model: Final = transformed_batch_request.get("model", "")
        deployment_endpoint_id: Final = get_custom_endpoint_id_from_api_base(api_base)
        if custom_endpoint and not job_model.endswith(f"/custom-endpoints/{deployment_endpoint_id}"):
            raise VertexAIError(
                status_code=400,
                message=(
                    "Vertex AI batch prediction on a `custom_endpoint` deployment requires an input "
                    "file uploaded through LiteLLM against that same deployment: the file id's "
                    "custom-endpoints/<endpoint id> path must name the endpoint the deployment's "
                    "`api_base` serves from."
                ),
            )
        if not custom_endpoint and "/custom-endpoints/" in job_model:
            raise VertexAIError(
                status_code=400,
                message=(
                    "This input file was uploaded for a `custom_endpoint` deployment; create the "
                    "batch against that deployment instead."
                ),
            )
        gateway_api_base: Final = _gateway_api_base_or_none(api_base)
        resolved_batch_request: Final = self._resolve_fine_tuned_endpoint_model(
            vertex_batch_request=transformed_batch_request,
            headers=headers,
            sync_handler=sync_handler,
            api_base=gateway_api_base,
            vertex_location=vertex_location or "us-central1",
        )
        vertex_batch_request: Final = (
            self._resolve_custom_endpoint_container(
                vertex_batch_request=resolved_batch_request,
                headers=headers,
                sync_handler=sync_handler,
                api_base=gateway_api_base,
                vertex_location=vertex_location or "us-central1",
            )
            if custom_endpoint
            else resolved_batch_request
        )
        is_unmanaged_container_job: Final = "unmanagedContainerModel" in vertex_batch_request

        default_api_base: Final = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
            vertex_api_version="v1beta1" if is_unmanaged_container_job else "v1",
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=gateway_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
            model=None,
            vertex_project=vertex_project or project_id,
            vertex_location=vertex_location or "us-central1",
            vertex_api_version="v1beta1" if is_unmanaged_container_job else "v1",
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

        payload_view: Final[_VertexBatchPayloadView] = {"payload": response.json()}
        _json_response: Final = payload_view["payload"]
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    @staticmethod
    def _build_endpoint_resolution_url(api_base: str | None, model: str, vertex_location: str) -> str:
        """
        Builds the GET url for resolving an endpoint resource (`projects/../endpoints/<id>`).

        A custom `api_base` replaces the Google host: its `/v1`/`/v1beta1` path swallows the
        version segment (matching `_check_custom_proxy`'s grafting), any other path is kept as a
        mount prefix in front of the full default path. The `:operation` suffix convention from
        `_check_custom_proxy` does not apply to a plain resource GET.
        """
        default_endpoint_url: Final = f"{get_vertex_base_url(vertex_location)}/v1/{model}"
        if not api_base:
            return default_endpoint_url
        api_base_path: Final = urlparse(api_base).path.rstrip("/")
        if api_base_path in ("/v1", "/v1beta1"):
            return _graft_default_vertex_path(api_base=api_base, default_url=default_endpoint_url)
        return api_base.rstrip("/") + urlparse(default_endpoint_url).path

    def _resolve_fine_tuned_endpoint_model(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        headers: dict[str, str],  # mutable-ok: HTTPHandler.get only accepts dict headers
        sync_handler: HTTPHandler,
        api_base: str | None,
        vertex_location: str,
    ) -> VertexAIBatchPredictionJob:
        """
        A fine-tuned Gemini deployment is configured by its endpoint id, but the v1 batch API only
        accepts Model resources, so swap the endpoint resource for its deployed tuned model
        (`projects/../locations/../models/<id>`) read from GET endpoints/<id>.
        """
        model: Final = vertex_batch_request.get("model", "")
        if "/endpoints/" not in model:
            return vertex_batch_request

        endpoint_url: Final = self._build_endpoint_resolution_url(
            api_base=api_base,
            model=model,
            vertex_location=vertex_location,
        )
        # ``api_base`` can come from caller-supplied request kwargs, so wrap the
        # fetch in ``safe_get``: it rejects DNS-rebind / private / cloud-metadata
        # targets before the bearer token leaves the process (mirrors retrieve_batch).
        fetched: Final[_FetchedResponseView] = {
            "response": safe_get(
                sync_handler,
                endpoint_url,
                headers=headers,
            )
        }
        response: Final = fetched["response"]
        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code,
                message=f"Failed to resolve fine-tuned Vertex endpoint '{model}': {response.text}",
            )

        payload_view: Final[_VertexEndpointPayloadView] = {"payload": response.json()}
        deployed_models: Final = payload_view["payload"].get("deployedModels") or ()
        deployed_model: Final = deployed_models[0].get("model", "") if deployed_models else ""
        if not deployed_model:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Vertex endpoint '{model}' has no deployed model, so there is no tuned model "
                    "resource to run batch predictions against"
                ),
            )
        resolved_request: Final[VertexAIBatchPredictionJob] = {**vertex_batch_request, "model": deployed_model}
        return resolved_request

    def _resolve_custom_endpoint_container(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        headers: dict[str, str],  # mutable-ok: HTTPHandler.get only accepts dict headers
        sync_handler: HTTPHandler,
        api_base: str | None,
        vertex_location: str,
    ) -> VertexAIBatchPredictionJob:
        """
        A `custom_endpoint` deployment serves an OpenAI-compatible container on a Vertex endpoint.
        The batch API accepts neither that endpoint (the v1beta1 BYOE `endpoint` field is refused
        with "specify model or unmanaged_container_model") nor its Model-Garden-sourced model
        resource ("Unknown ModelSource source_type: MODEL_GARDEN"), so the job instead runs
        batch-owned replicas of the same container: `unmanagedContainerModel` with the
        containerSpec read verbatim from the endpoint's deployed model (a hand-built spec loses
        model-source args and crash-loops) plus `dedicatedResources` copied from the endpoint's
        own deployment.
        """
        model: Final = vertex_batch_request.get("model", "")
        if "/custom-endpoints/" not in model:
            return vertex_batch_request
        endpoint_resource: Final = model.replace("/custom-endpoints/", "/endpoints/")

        endpoint_url: Final = self._build_endpoint_resolution_url(
            api_base=api_base,
            model=endpoint_resource,
            vertex_location=vertex_location,
        )
        endpoint_fetched: Final[_FetchedResponseView] = {
            "response": safe_get(sync_handler, endpoint_url, headers=headers)
        }
        endpoint_response: Final = endpoint_fetched["response"]
        if endpoint_response.status_code != 200:
            raise VertexAIError(
                status_code=endpoint_response.status_code,
                message=f"Failed to resolve custom Vertex endpoint '{endpoint_resource}': {endpoint_response.text}",
            )
        endpoint_view: Final[_VertexEndpointPayloadView] = {"payload": endpoint_response.json()}
        deployed_models: Final = endpoint_view["payload"].get("deployedModels") or ()
        if len(deployed_models) > 1:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Vertex endpoint '{endpoint_resource}' serves {len(deployed_models)} deployed "
                    "models behind a traffic split, so there is no single container to replicate "
                    "for batch prediction; use an endpoint with exactly one deployed model"
                ),
            )
        deployed: Final = deployed_models[0] if deployed_models else _VertexEndpointDeployedModel()
        deployed_model_resource: Final = deployed.get("model", "")
        if not deployed_model_resource:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Vertex endpoint '{endpoint_resource}' has no deployed model, so there is no "
                    "serving container to run batch predictions with"
                ),
            )

        model_url: Final = self._build_endpoint_resolution_url(
            api_base=api_base,
            model=deployed_model_resource,
            vertex_location=vertex_location,
        )
        model_fetched: Final[_FetchedResponseView] = {"response": safe_get(sync_handler, model_url, headers=headers)}
        model_response: Final = model_fetched["response"]
        if model_response.status_code != 200:
            raise VertexAIError(
                status_code=model_response.status_code,
                message=f"Failed to read model resource '{deployed_model_resource}': {model_response.text}",
            )
        model_view: Final[_VertexModelResourcePayloadView] = {"payload": model_response.json()}
        container_spec: Final = model_view["payload"].get("containerSpec")
        if not container_spec:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Model resource '{deployed_model_resource}' carries no containerSpec, so its "
                    "serving container cannot be replicated for batch prediction"
                ),
            )

        online_resources: Final = deployed.get("dedicatedResources") or _VertexOnlineDedicatedResources()
        machine_spec: Final = online_resources.get("machineSpec")
        if not machine_spec:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Vertex endpoint '{endpoint_resource}' exposes no dedicatedResources machine "
                    "spec to size the batch replicas from"
                ),
            )
        # A scale-to-zero online endpoint reports minReplicaCount 0, but a batch job must start
        # at least one replica.
        batch_resources: Final[BatchDedicatedResources] = {
            "machineSpec": machine_spec,
            "startingReplicaCount": max(online_resources.get("minReplicaCount", 1), 1),
            "maxReplicaCount": max(online_resources.get("maxReplicaCount", 1), 1),
        }
        unmanaged: Final[UnmanagedContainerModel] = {"containerSpec": container_spec}
        resolved: Final[VertexAIBatchPredictionJob] = {
            "displayName": vertex_batch_request["displayName"],
            "inputConfig": vertex_batch_request["inputConfig"],
            "outputConfig": vertex_batch_request["outputConfig"],
            "unmanagedContainerModel": unmanaged,
            "dedicatedResources": batch_resources,
            # excludedFields (not keyField, which does not actually strip and 400s vLLM) removes
            # the custom_id tag before the container sees it and echoes it in the output row.
            "instanceConfig": {
                "instanceType": "object",
                "excludedFields": (VERTEX_CUSTOM_ENDPOINT_KEY_FIELD,),
            },
        }
        return resolved

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

        payload_view: Final[_VertexBatchPayloadView] = {"payload": response.json()}
        _json_response: Final = payload_view["payload"]
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def create_vertex_batch_url(
        self,
        vertex_location: str,
        vertex_project: str,
        vertex_api_version: str = "v1",
    ) -> str:
        """Return the base url for the vertex garden models"""
        #  POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/batchPredictionJobs
        base_url: Final = get_vertex_base_url(vertex_location)
        return (
            f"{base_url}/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/batchPredictionJobs"
        )

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
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> LiteLLMBatch | Coroutine[object, object, LiteLLMBatch]:
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
            api_base=_gateway_api_base_or_none(api_base),
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
        fetched: Final[_FetchedResponseView] = {
            "response": safe_get(
                sync_handler,
                api_base,
                headers=headers,
            )
        }
        response: Final = fetched["response"]

        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_vertex_batch_payload(response)
        )
        return vertex_batch_response

    async def _async_retrieve_batch(
        self,
        api_base: str,
        headers: dict[str, str],
        logging_obj: "LiteLLMLoggingObj | None" = None,
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
        fetched: Final[_FetchedResponseView] = {
            "response": await async_safe_get(
                client,
                api_base,
                headers=headers,
            )
        }
        response: Final = fetched["response"]
        if response.status_code != 200:
            raise VertexAIError(
                status_code=response.status_code, message=f"Error: {response.status_code} {response.text}"
            )

        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_vertex_batch_payload(response)
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
            api_base=_gateway_api_base_or_none(api_base),
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

        limit_params: Final[dict[str, str]] = {"pageSize": str(limit)} if limit is not None else {}
        after_params: Final[dict[str, str]] = {"pageToken": after} if after is not None else {}
        params: Final = {**limit_params, **after_params}

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

        _json_response: Final = _vertex_batch_list_payload(response)
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
        params: dict[str, str],
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

        _json_response: Final = _vertex_batch_list_payload(response)
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
    ) -> LiteLLMBatch | Coroutine[object, object, LiteLLMBatch]:
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
            api_base=_gateway_api_base_or_none(api_base),
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

        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_vertex_batch_payload(retrieve_response)
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

        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_vertex_batch_payload(retrieve_response)
        )
        return vertex_batch_response

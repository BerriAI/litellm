from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import unquote

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.llms.vertex_ai.common_utils import (
    VertexAIError,
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.types.llms.openai import BatchJobStatus, CreateBatchRequest
from litellm.types.llms.vertex_ai import *
from litellm.types.llms.vertex_ai import GenerateContentResponseBody
from litellm.types.utils import LiteLLMBatch, ModelInfo, Usage

_NATIVE_VERTEX_RESPONSE: Final = TypeAdapter(GenerateContentResponseBody)


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value: Final = mapping.get(key)
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, str) and value.isdigit() else 0


def vertex_embedding_prompt_token_count(vertex_response: Mapping[str, object]) -> int:
    """
    Prompt tokens billed for one Vertex Gemini Embedding batch row.

    Live rows report usage under `usageMetadata`; the documented `tokenCount` is kept as
    a fallback.
    """
    usage_metadata: Final = vertex_response.get("usageMetadata")
    if isinstance(usage_metadata, Mapping):
        return _int_field(usage_metadata, "promptTokenCount")
    return _int_field(vertex_response, "tokenCount")


def is_vertex_embedding_batch_output_response(response_body: Mapping[str, object]) -> bool:
    return isinstance(response_body.get("embedding"), dict)


def is_native_vertex_batch_output_row(row: Mapping[str, object]) -> bool:
    return isinstance(row.get("request"), dict)


class NativeVertexBatchCostCalculator(Protocol):
    def __call__(
        self,
        usage: Usage,
        model: str,
        custom_llm_provider: str | None = None,
        model_info: ModelInfo | None = None,
    ) -> tuple[float, float]: ...


@dataclass(frozen=True, slots=True)
class NativeVertexBatchRowStats:
    usage: Usage
    total_tokens: int
    model: str | None
    prompt_cost: float
    completion_cost: float


def _native_vertex_row_usage(
    response_body: Mapping[str, object],
    calculate_usage: Callable[[GenerateContentResponseBody], Usage],
) -> Usage | None:
    if "usageMetadata" not in response_body:
        if not is_vertex_embedding_batch_output_response(response_body):
            return None
        prompt_tokens: Final = vertex_embedding_prompt_token_count(response_body)
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens)
    try:
        completion_response: Final = _NATIVE_VERTEX_RESPONSE.validate_python(response_body)
    except ValidationError as e:
        verbose_logger.debug("vertex_ai batch row response is not a GenerateContentResponse: %s", str(e))
        return None
    return calculate_usage(completion_response)


def native_vertex_batch_row_stats(
    row: Mapping[str, object],
    model_name: str | None,
    *,
    model_info: ModelInfo | None,
    calculate_usage: Callable[[GenerateContentResponseBody], Usage],
    cost_calculator: NativeVertexBatchCostCalculator,
) -> NativeVertexBatchRowStats | None:
    """
    Usage and cost of one native Vertex predictions.jsonl row, a
    `{"request": ..., "response": {"candidates": [...], "usageMetadata": {...}, "modelVersion": ...}}`
    generateContent object or a `{"request": ..., "response": {"embedding": {...}, "usageMetadata": {...}}}`
    embedding object (an embedding row without `usageMetadata` is billed from its documented `tokenCount`).
    `model_name` (the deployment model) prices the row unless it is a wildcard, else its own `modelVersion`
    does, else the wildcard name so explicit deployment prices still apply; a row without a response, a
    generateContent row without `response.usageMetadata`, and a row whose response fails validation are
    None (failed).
    """
    response_body: Final = row.get("response")
    if not isinstance(response_body, dict):
        return None
    usage: Final = _native_vertex_row_usage(response_body, calculate_usage)
    if usage is None:
        return None
    total_tokens: Final = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
    model_version: Final = response_body.get("modelVersion")
    deployment_model: Final = model_name if model_name and "*" not in model_name else None
    model: Final = deployment_model or (model_version if isinstance(model_version, str) else model_name)
    if model is None:
        verbose_logger.warning(
            "vertex_ai batch output row could not be costed, so it is billed at $0 and the rest of the batch "
            "is still billed: the row has no modelVersion and the batch has no deployment model"
        )
        return NativeVertexBatchRowStats(
            usage=usage, total_tokens=total_tokens, model=None, prompt_cost=0.0, completion_cost=0.0
        )
    try:
        prompt_cost, completion_cost = cost_calculator(
            usage=usage, model=model, custom_llm_provider="vertex_ai", model_info=model_info
        )
    except Exception as e:  # noqa: BLE001  # one unpriceable row must not abort the batch's cost accounting
        verbose_logger.warning(
            "vertex_ai batch output row could not be costed, so it is billed at $0 and the rest of the batch "
            "is still billed. model=%s error=%s",
            model,
            str(e),
        )
        return NativeVertexBatchRowStats(
            usage=usage, total_tokens=total_tokens, model=model, prompt_cost=0.0, completion_cost=0.0
        )
    return NativeVertexBatchRowStats(
        usage=usage, total_tokens=total_tokens, model=model, prompt_cost=prompt_cost, completion_cost=completion_cost
    )


class VertexAIBatchTransformation:
    """
    Transforms OpenAI Batch requests to Vertex AI Batch requests

    API Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
    """

    @classmethod
    def transform_openai_batch_request_to_vertex_ai_batch_request(
        cls,
        request: CreateBatchRequest,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> VertexAIBatchPredictionJob:
        """
        Transforms OpenAI Batch requests to Vertex AI Batch requests
        """
        request_display_name: Final = f"litellm-vertex-batch-{uuid.uuid4()}"
        input_file_id: Final = request.get("input_file_id")
        if input_file_id is None:
            raise ValueError("input_file_id is required, but not provided")
        input_config: InputConfig = InputConfig(gcsSource=GcsSource(uris=[input_file_id]), instancesFormat="jsonl")
        model: Final[str] = cls._get_batch_job_model(
            input_file_id=input_file_id,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
        output_config: Final[OutputConfig] = OutputConfig(
            predictionsFormat="jsonl",
            gcsDestination=GcsDestination(outputUriPrefix=cls._get_gcs_uri_prefix_from_file(input_file_id)),
        )
        return VertexAIBatchPredictionJob(
            inputConfig=input_config,
            outputConfig=output_config,
            model=model,
            displayName=request_display_name,
        )

    @classmethod
    def transform_vertex_ai_batch_response_to_openai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> LiteLLMBatch:
        return LiteLLMBatch(
            id=cls._get_batch_id_from_vertex_ai_batch_response(response),
            completion_window="24h",
            created_at=_convert_vertex_datetime_to_openai_datetime(vertex_datetime=response.get("createTime", "")),
            endpoint="",
            input_file_id=cls._get_input_file_id_from_vertex_ai_batch_response(response),
            object="batch",
            status=cls._get_batch_job_status_from_vertex_ai_batch_response(response),
            error_file_id=None,  # Vertex AI doesn't seem to have a direct equivalent
            output_file_id=cls._get_output_file_id_from_vertex_ai_batch_response(response),
        )

    @classmethod
    def transform_vertex_ai_batch_list_response_to_openai_list_response(
        cls, response: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Transforms Vertex AI batch list response into OpenAI-compatible list response.
        """

        batch_jobs: Final = response.get("batchPredictionJobs", []) or []
        data: Final = [cls.transform_vertex_ai_batch_response_to_openai_batch_response(job) for job in batch_jobs]

        first_id: Final = data[0].id if len(data) > 0 else None
        last_id: Final = data[-1].id if len(data) > 0 else None
        next_page_token: Final = response.get("nextPageToken")

        return {
            "object": "list",
            "data": data,
            "first_id": first_id,
            "last_id": last_id,
            "has_more": bool(next_page_token),
            "next_page_token": next_page_token,
        }

    @classmethod
    def _get_batch_id_from_vertex_ai_batch_response(cls, response: VertexBatchPredictionResponse) -> str:
        """
        Gets the batch id from the Vertex AI Batch response safely

        vertex response: `projects/510528649030/locations/us-central1/batchPredictionJobs/3814889423749775360`
        returns: `3814889423749775360`
        """
        _name: Final = response.get("name", "")
        if not _name:
            return ""

        # Split by '/' and get the last part if it exists
        parts: Final = _name.split("/")
        return parts[-1] if parts else _name

    @classmethod
    def _get_input_file_id_from_vertex_ai_batch_response(cls, response: VertexBatchPredictionResponse) -> str:
        """
        Gets the input file id from the Vertex AI Batch response
        """
        input_file_id: Final[str] = ""
        input_config: Final = response.get("inputConfig")
        if input_config is None:
            return input_file_id

        gcs_source: Final = input_config.get("gcsSource")
        if gcs_source is None:
            return input_file_id

        uris: Final = gcs_source.get("uris", "")
        if len(uris) == 0:
            return input_file_id

        return uris[0]

    @classmethod
    def _get_output_file_id_from_vertex_ai_batch_response(cls, response: VertexBatchPredictionResponse) -> str | None:
        """
        Gets the output file id from the Vertex AI Batch response, None until Vertex reports outputInfo
        """
        output_info: Final = response.get("outputInfo") or OutputInfo()
        gcs_output_directory: Final = output_info.get("gcsOutputDirectory", "").rstrip("/")
        if not gcs_output_directory:
            return None
        return f"{gcs_output_directory}/predictions.jsonl"

    @classmethod
    def _get_batch_job_status_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> BatchJobStatus:
        """
        Gets the batch job status from the Vertex AI Batch response

        ref: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/JobState
        """
        state_mapping: Final[dict[str, BatchJobStatus]] = {
            "JOB_STATE_UNSPECIFIED": "failed",
            "JOB_STATE_QUEUED": "validating",
            "JOB_STATE_PENDING": "validating",
            "JOB_STATE_RUNNING": "in_progress",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLING": "cancelling",
            "JOB_STATE_CANCELLED": "cancelled",
            "JOB_STATE_PAUSED": "in_progress",
            "JOB_STATE_EXPIRED": "expired",
            "JOB_STATE_UPDATING": "in_progress",
            "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
        }

        vertex_state: Final = response.get("state", "JOB_STATE_UNSPECIFIED")
        return state_mapping[vertex_state]

    @classmethod
    def _get_gcs_uri_prefix_from_file(cls, input_file_id: str) -> str:
        """
        Gets the gcs uri prefix from the input file id

        Example:
        input_file_id: "gs://litellm-testing-bucket/vtx_batch.jsonl"
        returns: "gs://litellm-testing-bucket"

        input_file_id: "gs://litellm-testing-bucket/batches/vtx_batch.jsonl"
        returns: "gs://litellm-testing-bucket/batches"
        """
        # Split the path and remove the filename
        path_parts: Final = input_file_id.rsplit("/", 1)
        return path_parts[0]

    @classmethod
    def _get_batch_job_model(
        cls,
        input_file_id: str,
        vertex_project: str | None,
        vertex_location: str | None,
    ) -> str:
        """
        Returns the `model` for the batchPredictionJobs request: the publisher model path as-is, or
        the full `projects/../locations/../endpoints/<id>` resource name for a fine-tuned endpoint.

        The v1 batch API only accepts Model resources, so the handler resolves an endpoint resource
        to its deployed tuned model (`projects/../locations/../models/<id>`) before sending the job.
        """
        parsed_model: Final = cls._get_model_from_gcs_file(input_file_id)
        if not parsed_model.startswith("endpoints/"):
            return parsed_model
        if not vertex_project:
            raise VertexAIError(
                status_code=400,
                message=(
                    f"Vertex AI batch jobs against a fine-tuned endpoint ('{parsed_model}') require "
                    "`vertex_project` to build the endpoint resource name"
                ),
            )
        return f"projects/{vertex_project}/locations/{vertex_location or 'us-central1'}/{parsed_model}"

    @classmethod
    def _get_model_from_gcs_file(cls, gcs_file_uri: str) -> str:
        """
        Extracts the model from the gcs file uri

        When files are uploaded using LiteLLM (/v1/files), the model is stored in the gcs file uri

        Why?
        - Because Vertex Requires the `model` param in create batch jobs request, but OpenAI does not require this


        gcs_file_uri format: gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/e9412502-2c91-42a6-8e61-f5c294cc0fc8
        returns: "publishers/google/models/gemini-1.5-flash-001"

        Fine-tuned Gemini endpoints are stored as `endpoints/<numeric id>` in the uri and returned
        in that form.

        Raises a 400 `VertexAIError` when the uri carries no parseable model path.
        """
        model: Final = cls._parse_model_from_gcs_file(gcs_file_uri)
        if model is None:
            raise VertexAIError(
                status_code=400,
                message=(
                    "Vertex AI batch creation requires the model to be part of `input_file_id`, but "
                    f"'{gcs_file_uri}' contains no 'publishers/<publisher>/models/<model>' or "
                    "'endpoints/<numeric endpoint id>' path segment. "
                    "Either upload the input file through LiteLLM (POST /v1/files with "
                    "custom_llm_provider=vertex_ai), which encodes the model into the returned file id, or "
                    "pass a uri of the form "
                    "gs://<bucket>/<prefix>/publishers/<publisher>/models/<model>/<file> "
                    "(or gs://<bucket>/<prefix>/endpoints/<numeric endpoint id>/<file> for fine-tuned models)"
                ),
            )
        return model

    @classmethod
    def _parse_model_from_gcs_file(cls, gcs_file_uri: str) -> str | None:
        """
        Returns the `publishers/<publisher>/models/<model>` or `endpoints/<numeric id>` path from a
        gcs uri, or None if the uri does not contain one.

        A publisher path wins over an `endpoints/` segment, and the last `endpoints/` occurrence is
        used, so a user-configured bucket prefix that happens to contain `endpoints/<digits>` cannot
        override the model path LiteLLM appended after it.
        """
        unquoted_uri: Final = unquote(gcs_file_uri)
        _, separator, model_path = unquoted_uri.partition("publishers/")
        if separator:
            parts: Final = model_path.split("/")
            if len(parts) >= 3 and parts[1] == "models" and parts[2]:
                return f"publishers/{'/'.join(parts[:3])}"

        _, endpoint_separator, endpoint_path = unquoted_uri.rpartition("endpoints/")
        endpoint_id: Final = endpoint_path.split("/")[0] if endpoint_separator else ""
        if endpoint_id.isdigit():
            return f"endpoints/{endpoint_id}"

        return None

    @classmethod
    def is_unmanaged_gcs_batch_input_file_id(cls, input_file_id: str | None) -> bool:
        """
        Returns True if `input_file_id` is a raw gs:// Vertex batch input file (i.e. not a
        LiteLLM-managed unified file id) with a `publishers/` model path that
        `_get_model_from_gcs_file` can parse.
        """
        return (
            input_file_id is not None
            and input_file_id.startswith("gs://")
            and cls._parse_model_from_gcs_file(input_file_id) is not None
        )

    @classmethod
    def get_bare_model_name_from_gcs_file(cls, gcs_file_uri: str) -> str:
        """
        Extracts the bare model name (e.g. "gemini-1.5-flash-001") from a gcs file uri.
        """
        return cls._get_model_from_gcs_file(gcs_file_uri).rsplit("/", 1)[-1]

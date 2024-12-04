import uuid
from typing import Any, Dict, Literal

from litellm.types.llms.openai import Batch, BatchJobStatus, CreateBatchRequest
from litellm.types.llms.vertex_ai import *


class VertexAIBatchTransformation:
    """
    Transforms OpenAI Batch requests to Vertex AI Batch requests

    API Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
    """

    @classmethod
    def transform_openai_batch_request_to_vertex_ai_batch_request(
        cls,
        request: CreateBatchRequest,
    ) -> VertexAIBatchPredictionJob:
        """
        Transforms OpenAI Batch requests to Vertex AI Batch requests
        """
        request_display_name = f"litellm-vertex-batch-{uuid.uuid4()}"
        input_file_id = request.get("input_file_id")
        if input_file_id is None:
            raise ValueError("input_file_id is required, but not provided")
        input_config: InputConfig = InputConfig(
            gcsSource=GcsSource(uris=input_file_id), instancesFormat="jsonl"
        )
        model: str = cls._get_model_from_gcs_file(input_file_id)
        output_config: OutputConfig = OutputConfig(
            predictionsFormat="jsonl",
            gcsDestination=GcsDestination(
                outputUriPrefix=cls._get_gcs_uri_prefix_from_file(input_file_id)
            ),
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
    ) -> Batch:
        return Batch(
            id=response.get("name", ""),
            completion_window="24hrs",
            created_at=cls._convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=response.get("createTime", "")
            ),
            endpoint="",
            input_file_id=cls._get_input_file_id_from_vertex_ai_batch_response(
                response
            ),
            object="batch",
            status=cls._get_batch_job_status_from_vertex_ai_batch_response(response),
            error_file_id=None,  # Vertex AI doesn't seem to have a direct equivalent
            output_file_id=cls._get_output_file_id_from_vertex_ai_batch_response(
                response
            ),
        )

    @classmethod
    def _get_input_file_id_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> str:
        """
        Gets the input file id from the Vertex AI Batch response
        """
        input_file_id: str = ""
        input_config = response.get("inputConfig")
        if input_config is None:
            return input_file_id

        gcs_source = input_config.get("gcsSource")
        if gcs_source is None:
            return input_file_id

        uris = gcs_source.get("uris", "")
        if len(uris) == 0:
            return input_file_id

        return uris[0]

    @classmethod
    def _get_output_file_id_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> str:
        """
        Gets the output file id from the Vertex AI Batch response
        """
        output_file_id: str = ""
        output_config = response.get("outputConfig")
        if output_config is None:
            return output_file_id

        gcs_destination = output_config.get("gcsDestination")
        if gcs_destination is None:
            return output_file_id

        output_uri_prefix = gcs_destination.get("outputUriPrefix", "")
        return output_uri_prefix

    @classmethod
    def _get_batch_job_status_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> BatchJobStatus:
        """
        Gets the batch job status from the Vertex AI Batch response

        ref: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/JobState
        """
        state_mapping: Dict[str, BatchJobStatus] = {
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

        vertex_state = response.get("state", "JOB_STATE_UNSPECIFIED")
        return state_mapping[vertex_state]

    @classmethod
    def _convert_vertex_datetime_to_openai_datetime(cls, vertex_datetime: str) -> int:
        """
        Converts a Vertex AI datetime string to an OpenAI datetime integer

        vertex_datetime: str = "2024-12-04T21:53:12.120184Z"
        returns: int = 1722729192
        """
        from datetime import datetime

        # Parse the ISO format string to datetime object
        dt = datetime.strptime(vertex_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
        # Convert to Unix timestamp (seconds since epoch)
        return int(dt.timestamp())

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
        path_parts = input_file_id.rsplit("/", 1)
        return path_parts[0]

    @classmethod
    def _get_model_from_gcs_file(cls, gcs_file_uri: str) -> str:
        return "publishers/google/models/gemini-1.5-flash-001"

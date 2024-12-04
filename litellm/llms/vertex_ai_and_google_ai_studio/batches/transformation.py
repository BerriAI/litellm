import uuid
from typing import Any, Dict

from litellm.types.llms.openai import CreateBatchRequest
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
        gcs_bucket_name: str,
    ) -> VertexAIBatchPredictionJob:
        """
        Transforms OpenAI Batch requests to Vertex AI Batch requests
        """
        gcs_bucket_name = "litellm-testing-bucket"
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
            gcsDestination=GcsDestination(outputUriPrefix=gcs_bucket_name),
        )
        return VertexAIBatchPredictionJob(
            inputConfig=input_config,
            outputConfig=output_config,
            model=model,
            displayName=request_display_name,
        )

    @classmethod
    def _get_model_from_gcs_file(cls, gcs_file_uri: str) -> str:
        return "publishers/google/models/gemini-1.5-flash-001"

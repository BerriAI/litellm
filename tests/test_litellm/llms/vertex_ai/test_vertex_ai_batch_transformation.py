from litellm.llms.vertex_ai.batches.transformation import VertexAIBatchTransformation


def test_output_file_id_uses_predictions_jsonl_with_output_info():
    response = {
        "outputInfo": {
            "gcsOutputDirectory": "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-123"
        }
    }

    output_file_id = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )

    assert (
        output_file_id
        == "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-123/predictions.jsonl"
    )


def test_output_file_id_falls_back_to_output_uri_prefix_with_predictions_jsonl():
    response = {
        "outputInfo": {},
        "outputConfig": {
            "gcsDestination": {
                "outputUriPrefix": "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-456"
            }
        },
    }

    output_file_id = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )

    assert (
        output_file_id
        == "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-456/predictions.jsonl"
    )

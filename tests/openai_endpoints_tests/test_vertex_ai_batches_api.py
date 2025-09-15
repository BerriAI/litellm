from openai import OpenAI
import pytest

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)


VERTEX_AI_BATCH_MODEL = "vertex_ai/gemini-2.5-flash-batches"


@pytest.mark.asyncio
async def test_vertex_ai_batches_api():
    """
    Test vertex ai batches api

    E2E Test Creating a File and a Batch on Vertex AI
    """
    # Upload file
    batch_input_file = client.files.create(
        file=open("tests/openai_endpoints_tests/vertex_ai_batch_completions.jsonl", "rb"),
        purpose="batch",
        extra_body={"target_model_names": VERTEX_AI_BATCH_MODEL}
    )
    print(batch_input_file)

    # Create batch
    batch = client.batches.create( 
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Test batch job"},
    )
    print(batch)

    assert batch.id is not None
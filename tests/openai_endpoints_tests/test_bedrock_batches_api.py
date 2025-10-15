from openai import OpenAI
import pytest

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",
)


BEDROCK_BATCH_MODEL = "bedrock/batch-anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.mark.asyncio
async def test_bedrock_batches_api():
    """
    Test bedrock batches api

    E2E Test Creating a File and a Batch on Bedrock
    """
    # Upload file
    batch_input_file = client.files.create(
        file=open("tests/openai_endpoints_tests/bedrock_batch_completions.jsonl", "rb"),
        purpose="batch",
        extra_body={"target_model_names": BEDROCK_BATCH_MODEL}
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
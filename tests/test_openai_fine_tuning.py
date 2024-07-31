from openai import AsyncOpenAI
import os
import pytest


@pytest.mark.asyncio
async def test_openai_fine_tuning():
    """
    [PROD Test] Ensures logprobs are returned correctly
    """
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    response = await client.files.create(
        extra_body={"custom_llm_provider": "azure"},
        file=open(file_path, "rb"),
        purpose="fine-tune",
    )

    print("response from files.create: {}".format(response))

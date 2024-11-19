import pytest

import litellm


def test_mlflow_logging():
    litellm.success_callback = ["mlflow"]
    litellm.failure_callback = ["mlflow"]

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "what llm are u"}],
        max_tokens=10,
        temperature=0.2,
        user="test-user",
    )

@pytest.mark.asyncio()
async def test_async_mlflow_logging():
    litellm.success_callback = ["mlflow"]
    litellm.failure_callback = ["mlflow"]

    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi test from local arize"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )

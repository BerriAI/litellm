import os

import pytest

import litellm


@pytest.fixture(autouse=True)
def isolate_host_aws_config(monkeypatch, isolated_aws_credentials_dir):
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", isolated_aws_credentials_dir["credentials"])
    monkeypatch.setenv("AWS_CONFIG_FILE", isolated_aws_credentials_dir["config"])
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)


@pytest.mark.asyncio
async def test_bedrock_messages_with_rust() -> None:
    response = await litellm.anthropic.messages.acreate(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        api_key=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
        aws_region_name="us-west-2",
        rust=True,
    )

    assert response["content"][0]["text"]
    assert response["_hidden_params"]["additional_headers"]["x-litellm-rust"] == "true"


@pytest.mark.asyncio
async def test_bedrock_messages_streaming_with_rust() -> None:
    response = await litellm.anthropic.messages.acreate(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        stream=True,
        api_key=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
        aws_region_name="us-west-2",
        rust=True,
    )

    chunks = [chunk async for chunk in response]

    assert chunks
    assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"

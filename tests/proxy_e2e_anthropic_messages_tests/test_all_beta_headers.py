"""
This test ensures that the proxy can passthrough anthropic requests
"""
from pathlib import Path
import pytest
import aiohttp
import json

def get_all_supported_anthropic_beta_headers(provider: str):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "litellm"
        / "anthropic_beta_headers_config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    anthropic_mapping = config.get(provider, {})

    # Only include headers that have a non-null mapping value
    return [
        header_name
        for header_name, provider_value in anthropic_mapping.items()
        if provider_value is not None
    ]


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.parametrize(
    "model_name,provider_name",
    [
        ("claude-sonnet-4-5-20250929", "anthropic"),
    ],
)
async def test_anthropic_messages_with_all_beta_headers(model_name, provider_name):
    """
    Test that v1/messages endpoint works with all non-null Anthropic beta headers
    and doesn't throw errors
    """
    print("Testing v1/messages with all non-null Anthropic beta headers")
    
    
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": ",".join(get_all_supported_anthropic_beta_headers(provider_name)),
    }
    
    payload = {
        "model": model_name,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else"}],
        "tools": [{
            "type": "code_execution_20250825",
            "name": "code_execution"
        }]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://0.0.0.0:4000/v1/messages", 
            json=payload, 
            headers=headers
        ) as response:
            response_text = await response.text()
            print(f"Response status: {response.status}")
            print(f"Response text: {response_text}")
            
            # The request should succeed without errors
            assert response.status == 200, f"Request should succeed, got status {response.status}: {response_text}"
            
            response_json = await response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=4, default=str)}")
            
            # Basic response validation
            assert "id" in response_json, "Response should have an id"
            assert "content" in response_json, "Response should have content"
            assert "model" in response_json, "Response should have model"
            assert "usage" in response_json, "Response should have usage"
            
            # Verify usage information
            usage = response_json["usage"]
            assert "input_tokens" in usage, "Usage should have input_tokens"
            assert "output_tokens" in usage, "Usage should have output_tokens"
            assert usage["input_tokens"] > 0, "Should have some input tokens"
            assert usage["output_tokens"] > 0, "Should have some output tokens"
            
            print(f"✅ Test passed: Request with all beta headers succeeded")
            print(f"   Model: {response_json['model']}")
            print(f"   Input tokens: {usage['input_tokens']}")
            print(f"   Output tokens: {usage['output_tokens']}")



@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.parametrize(
    "model_name,provider_name",
    [
        ("bedrock-claude-opus-4.5", "bedrock"),
        ("bedrock-converse-claude-sonnet-4.5", "bedrock_converse")
    ],
)
async def test_bedrock_invoke_messages_with_all_beta_headers(
    model_name, provider_name
):
    """
    Test that v1/messages endpoint works with all non-null Anthropic beta headers
    for both bedrock and bedrock_converse providers.
    """
    print(f"Testing v1/messages for model={model_name}, provider={provider_name}")

    beta_headers = get_all_supported_anthropic_beta_headers(provider_name)

    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": ",".join(beta_headers),
    }

    payload = {
        "model": model_name,
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Say 'hello' and nothing else"}
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://0.0.0.0:4000/v1/messages",
            json=payload,
            headers=headers,
        ) as response:

            response_text = await response.text()
            print(f"Response status: {response.status}")
            print(f"Response text: {response_text}")

            assert (
                response.status == 200
            ), f"{provider_name} request failed: {response.status}: {response_text}"

            response_json = await response.json()

            # Basic response validation
            assert "id" in response_json
            assert "content" in response_json
            assert "model" in response_json
            assert "usage" in response_json

            usage = response_json["usage"]

            assert "input_tokens" in usage
            assert "output_tokens" in usage
            assert usage["input_tokens"] > 0
            assert usage["output_tokens"] > 0

            print("✅ Test passed")
            print(f"   Provider: {provider_name}")
            print(f"   Model: {response_json['model']}")
            print(f"   Input tokens: {usage['input_tokens']}")
            print(f"   Output tokens: {usage['output_tokens']}")

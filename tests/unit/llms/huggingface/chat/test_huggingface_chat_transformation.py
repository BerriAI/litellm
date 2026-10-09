from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import respx

from litellm.llms.huggingface.common_utils import _fetch_inference_provider_mapping


@pytest.fixture(autouse=True)
def isolate_huggingface_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("HF_API_BASE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_BASE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    _fetch_inference_provider_mapping.cache_clear()
    yield
    _fetch_inference_provider_mapping.cache_clear()


PROVIDER_MAPPING_RESPONSE = {
    "fireworks-ai": {
        "status": "live",
        "providerId": "accounts/fireworks/models/llama-v3-8b-instruct",
        "task": "conversational",
    },
    "together": {
        "status": "live",
        "providerId": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "task": "conversational",
    },
    "hf-inference": {
        "status": "live",
        "providerId": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task": "conversational",
    },
}


@pytest.fixture
def mock_provider_mapping() -> Iterator[MagicMock]:
    with patch(
        "litellm.llms.huggingface.chat.transformation.fetch_inference_provider_mapping"
    ) as mock:
        mock.return_value = PROVIDER_MAPPING_RESPONSE
        yield mock


@pytest.mark.usefixtures("fake_provider_credentials")
def test_build_chat_completion_url_function():
    """Test the _build_chat_completion_url helper function"""
    from litellm.llms.huggingface.chat.transformation import (
        _build_chat_completion_url,
    )

    test_cases = [
        ("https://example.com", "https://example.com/v1/chat/completions"),
        ("https://example.com/", "https://example.com/v1/chat/completions"),
        ("https://example.com/v1", "https://example.com/v1/chat/completions"),
        ("https://example.com/v1/", "https://example.com/v1/chat/completions"),
        (
            "https://example.com/v1/chat/completions",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/",
            "https://example.com/custom/path/v1/chat/completions",
        ),
    ]

    for input_url, expected_url in test_cases:
        result = _build_chat_completion_url(input_url)
        assert (
            result == expected_url
        ), f"Failed for input: {input_url}, expected: {expected_url}, got: {result}"


@pytest.mark.parametrize(
    "model, expected_url",
    [
        (
            "meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/v1/chat/completions",
        ),
        (
            "together/meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/together/v1/chat/completions",
        ),
        (
            "novita/meta-llama/Llama-3-8B-Instruct",
            "https://router.huggingface.co/novita/v3/openai/chat/completions",
        ),
        (
            "http://custom-endpoint.com/v1/chat/completions",
            "http://custom-endpoint.com/v1/chat/completions",
        ),
    ],
)
def test_get_complete_url(model, expected_url):
    """Test that the complete URL is constructed correctly for different providers"""
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    url = config.get_complete_url(
        api_base=None,
        model=model,
        optional_params={},
        stream=False,
        api_key="test_api_key",
        litellm_params={},
    )
    assert url == expected_url


@pytest.mark.parametrize(
    "api_base, model, expected_url",
    [
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
            "huggingface/tgi",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "huggingface/tgi",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/v1/chat/completions",
            "huggingface/tgi",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/v1",
            "huggingface/tgi",
            "https://example.com/v1/chat/completions",
        ),
    ],
)
def test_get_complete_url_inference_endpoints(api_base, model, expected_url):
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    url = config.get_complete_url(
        api_base=api_base,
        model=model,
        optional_params={},
        stream=False,
        api_key="test_api_key",
        litellm_params={},
    )
    assert url == expected_url


@pytest.mark.usefixtures("mock_provider_mapping")
@pytest.mark.parametrize(
    "model, expected_model",
    [
        (
            "together/meta-llama/Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        ),
        (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ),
    ],
)
def test_transform_request(model, expected_model):
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()
    messages = [{"role": "user", "content": "Hello"}]

    transformed_request = config.transform_request(
        model=model,
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert transformed_request["model"] == expected_model
    assert transformed_request["messages"] == messages


@pytest.mark.usefixtures("fake_provider_credentials")
def test_validate_environment():
    """Test that the environment is validated correctly"""
    from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

    config = HuggingFaceChatConfig()

    headers = config.validate_environment(
        headers={},
        model="huggingface/fireworks-ai/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        api_key="test_api_key",
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer test_api_key"
    assert headers["content-type"] == "application/json"

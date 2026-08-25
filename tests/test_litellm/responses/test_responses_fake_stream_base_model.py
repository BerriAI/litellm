from unittest.mock import MagicMock, patch

import litellm


def _fake_stream_passed_to_handler(**extra_kwargs):
    with patch(
        "litellm.responses.main.base_llm_http_handler.response_api_handler",
        return_value=MagicMock(),
    ) as mock_handler:
        litellm.responses(
            model="azure/my-custom-ptu-deployment",
            input="hello",
            stream=True,
            api_key="fake-api-key",
            api_base="https://fake.openai.azure.com",
            api_version="2025-01-01-preview",
            **extra_kwargs,
        )
    return mock_handler.call_args.kwargs["fake_stream"]


def test_responses_streams_natively_when_model_info_base_model_streams():
    assert _fake_stream_passed_to_handler(model_info={"base_model": "azure/gpt-4o"}) is False


def test_responses_fake_streams_without_base_model():
    assert _fake_stream_passed_to_handler() is True

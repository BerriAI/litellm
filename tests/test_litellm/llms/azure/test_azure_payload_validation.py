import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.common_utils import validate_azure_request_payload

def test_validate_azure_request_payload():
    # Test valid dict
    valid_payload = {"key": "value"}
    assert validate_azure_request_payload(valid_payload) == valid_payload

    # Test surrogate string removal
    malformed_payload = {"input": "test\ud83d"}
    fixed = validate_azure_request_payload(malformed_payload)
    assert fixed["input"] == "test\ufffd"

@patch("litellm.llms.azure.azure.AzureChatCompletion.get_azure_openai_client")
def test_audio_speech_payload_validation(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.audio.speech.create.return_value = MagicMock(response=b"test")

    azure_model = AzureChatCompletion()
    
    azure_model.audio_speech(
        model="test-model",
        input="test\ud83d",
        voice="alloy",
        optional_params={"speed": 1.0},
        api_key="sk-test",
        api_base="https://test.api.cognitive.microsoft.com",
        api_version="2023-05-15",
        organization=None,
        max_retries=2,
        timeout=10.0,
    )
    
    mock_client.audio.speech.create.assert_called_once()
    kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert kwargs["input"] == "test\ufffd" # The surrogate should be stripped
    assert kwargs["model"] == "test-model"
    assert kwargs["voice"] == "alloy"

@patch("litellm.llms.azure.azure.AzureChatCompletion.get_azure_openai_client")
@pytest.mark.asyncio
async def test_async_audio_speech_payload_validation(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_create = AsyncMock()
    mock_create.return_value = MagicMock(response=b"test")
    mock_client.audio.speech.create = mock_create

    azure_model = AzureChatCompletion()
    
    await azure_model.async_audio_speech(
        model="test-model",
        input="test\ud83d",
        voice="alloy",
        optional_params={"speed": 1.0},
        api_key="sk-test",
        api_base="https://test.api.cognitive.microsoft.com",
        api_version="2023-05-15",
        azure_ad_token=None,
        azure_ad_token_provider=None,
        max_retries=2,
        timeout=10.0,
    )
    
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["input"] == "test\ufffd"
    assert kwargs["model"] == "test-model"

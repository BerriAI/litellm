import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion
from openai import AzureOpenAI


def test_azure_o4_streaming():
    """
    Test that o4 models support native streaming.
    """
    client = AzureOpenAI(
        api_key="my-fake-o1-key",
        base_url="https://openai-gpt-4-test-v-1.openai.azure.com",
        api_version="2024-02-15-preview",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_create:
        try:
            completion(
                model="azure/o4-mini",
                messages=[{"role": "user", "content": "Hello, world!"}],
                stream=True,
                client=client,
            )
        except (
            Exception
        ) as e:  # expect output translation error as mock response doesn't return a json
            print(e)
        assert mock_create.call_count == 1
        assert "stream" in mock_create.call_args.kwargs


def test_should_fake_stream_for_o4_models():
    """
    Test that should_fake_stream returns False for o4 models.
    """
    config = litellm.AzureOpenAIO1Config()
    
    # Test o4-mini
    assert config.should_fake_stream(model="azure/o4-mini", stream=True) is False
    
    # Test o4 model
    assert config.should_fake_stream(model="azure/o4", stream=True) is False
    
    # Test that o1 still uses fake streaming
    assert config.should_fake_stream(model="azure/o1", stream=True) is True
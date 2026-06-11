import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm


def test_azure_o1_override_fake_stream():
    """Test that native streaming is not supported for o1."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure/o1-preview",
                "litellm_params": {
                    "model": "azure/o1-preview",
                    "api_key": "my-fake-o1-key",
                    "api_base": "https://openai-gpt-4-test-v-1.openai.azure.com",
                },
                "model_info": {
                    "supports_native_streaming": True,
                },
            }
        ]
    )

    ## check model info

    model_info = litellm.get_model_info(
        model="azure/o1-preview", custom_llm_provider="azure"
    )
    assert model_info["supports_native_streaming"] is True

    fake_stream = litellm.AzureOpenAIO1Config().should_fake_stream(
        model="azure/o1-preview", stream=True
    )
    assert fake_stream is False

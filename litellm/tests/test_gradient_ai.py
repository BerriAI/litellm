import requests
from litellm import completion
import litellm
import pytest


def test_gradient_valid_config() -> None:
    litellm.gradient_workspace_id = "valid_workspace_id"
    litellm.gradient_key = "my_very_secret_and_trustworthy_key"
    with pytest.raises(litellm.llms.gradient.GradientAIError) or pytest.raises(
        requests.RequestException
    ):
        completion(
            # set model to the ID (base or fine-tuned adapter) in gradient.ai
            model="model",
            custom_llm_provider="gradient",
            messages=[{"content": "Please say: `bar`", "role": "user"}],
        )


def test_gradient_invalid_workspace() -> None:
    litellm.gradient_workspace_id = ""  # empty
    litellm.gradient_key = "my_very_secret_and_trustworthy_key"  # empty
    with pytest.raises(litellm.llms.gradient.GradientMissingSecretError):
        completion(
            # set model to the ID (base or fine-tuned adapter) in gradient.ai
            model="model",
            custom_llm_provider="gradient",
            messages=[{"content": "Please say: `bar`", "role": "user"}],
        )


def test_gradient_invalid_key() -> None:
    litellm.gradient_workspace_id = "valid_long_workspace"  # empty
    litellm.gradient_key = ""  # empty
    with pytest.raises(litellm.llms.gradient.GradientMissingSecretError):
        completion(
            # set model to the ID (base or fine-tuned adapter) in gradient.ai
            model="model",
            custom_llm_provider="gradient",
            messages=[{"content": "Please say: `bar`", "role": "user"}],
        )

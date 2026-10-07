import pytest

import litellm
import litellm.interactions as interactions


def test_missing_model_and_agent():
    with pytest.raises((ValueError, litellm.APIConnectionError)):
        interactions.create(input="Hello", api_key="test-api-key")

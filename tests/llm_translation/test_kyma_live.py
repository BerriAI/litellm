"""
Live integration tests for Kyma API provider.
These tests make real network calls and require KYMA_API_KEY to be set.
They are intentionally placed here (llm_translation/) to be run only in
CI environments with credentials configured, not in the standard unit test suite.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm

KYMA_API_KEY = os.environ.get("KYMA_API_KEY")


@pytest.mark.skipif(not KYMA_API_KEY, reason="KYMA_API_KEY not set")
def test_kyma_live_completion():
    """Live test: call kyma/llama-3.3-70b and verify a valid response"""
    response = litellm.completion(
        model="kyma/llama-3.3-70b",
        messages=[{"role": "user", "content": "Say 'test successful' and nothing else"}],
        max_tokens=10,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

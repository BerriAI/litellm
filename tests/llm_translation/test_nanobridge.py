"""
Integration tests for Nanobridge (real API calls).

Requires NANOBRIDGE_API_KEY. Not collected in mock-only CI for tests/test_litellm/.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


@pytest.mark.skipif(
    not os.environ.get("NANOBRIDGE_API_KEY"),
    reason="NANOBRIDGE_API_KEY not set",
)
def test_nanobridge_live_chat_completions():
    response = litellm.completion(
        model="nanobridge/deepseek-v4-flash",
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=16,
    )
    assert response.choices[0].message.content

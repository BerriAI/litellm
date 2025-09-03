
import os
import sys
import pytest
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths
import litellm

def test_completion_openrouter_reasoning_content():
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="openrouter/anthropic/claude-3.7-sonnet",
        messages=[{"role": "user", "content": "Hello world"}],
        reasoning={"effort": "high"},
    )
    print(resp)
    assert resp.choices[0].message.reasoning_content is not None

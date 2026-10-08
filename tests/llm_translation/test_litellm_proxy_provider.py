import re


import litellm
import pytest





























def test_litellm_gateway_from_sdk_with_thinking_param():
    with pytest.raises(Exception, match=re.escape("Connection error.")) as exc_info:
        response = litellm.completion(
            model="litellm_proxy/anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Hello world"}],
            api_base="http://0.0.0.0:4000",
            api_key="sk-PIp1h0RekR",
            # client=openai_client,
            thinking={"type": "enabled", "max_budget": 100},
        )
    e = exc_info.value
    assert "Connection error." in str(e)

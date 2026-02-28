import os
import sys
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import ProxyLogging


class MockProviderGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_hook_type = None
        self.last_data_messages = None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Optional[dict]:
        self.last_hook_type = "pre_call"
        # The hook should receive the filtered messages
        import copy

        self.last_data_messages = copy.deepcopy(data.get("messages"))

        # Simulate the guardrail modifying the message (e.g. masking PII)
        if self.last_data_messages and len(self.last_data_messages) > 0:
            self.last_data_messages[0]["content"] = self.last_data_messages[0][
                "content"
            ].upper()
            data["messages"] = self.last_data_messages

        return data


@pytest.mark.asyncio
async def test_guardrail_dispatch_message_filtering():
    """
    Test that _execute_guardrail_hook correctly filters messages when
    experimental_use_latest_role_message_only=True, and then merges the
    modifications back.
    """
    proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())
    guardrail = MockProviderGuardrail()
    guardrail.experimental_use_latest_role_message_only = True

    # Multi-turn conversation
    data = {
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Assistant reply"},
            {"role": "user", "content": "Second message"},
        ]
    }

    # Dispatch the pre_call hook
    result = await proxy_logging._execute_guardrail_hook(
        callback=guardrail,
        hook_type="pre_call",
        data=data,
        user_api_key_dict=MagicMock(),
        call_type="completion",
    )

    # 1. Verify the hook only saw the latest user message
    assert guardrail.last_data_messages is not None
    assert len(guardrail.last_data_messages) == 1
    # Inside the hook, it was uppercased
    assert guardrail.last_data_messages[0]["content"] == "SECOND MESSAGE"

    # 2. Verify the original data was correctly restored and merged
    assert "messages" in data
    assert len(data["messages"]) == 3
    assert data["messages"][0]["content"] == "First message"
    assert data["messages"][1]["content"] == "Assistant reply"
    # The last message should have the modification from the hook
    assert data["messages"][2]["content"] == "SECOND MESSAGE"

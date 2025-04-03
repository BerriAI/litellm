import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.llms.datarobot.chat.transformation import DataRobotConfig


class TestDataRobotConfig:

    @pytest.mark.parametrize(
        "api_base, expected_url",
        [
            ("http://localhost:5001", "http://localhost:5001/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com/api/v2/genai/llmgw/chat/completions", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://staging.datarobot.com", "https://staging.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
        ]
    )
    def test_get_complete_url(self, api_base, expected_url):
        handler = DataRobotConfig()
        assert handler.get_complete_url(api_base, "test_model", {}, {}, False) == expected_url

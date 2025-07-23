import os
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from unittest.mock import patch
from litellm.llms.heroku.chat.transformation import HerokuChatConfig

class TestHerokuChatConfig:
    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = HerokuChatConfig()
        headers = {}
        api_key = "fake-heroku-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="claude-3-5-haiku",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # set env var for api_base
        os.environ["HEROKU_API_BASE"] = "https://mia.heroku.com"

        print('****************************************')
        print(config.get_complete_url(api_base=None, api_key=api_key, model="claude-3-5-haiku", optional_params={}, litellm_params={}, stream=False))
        print('****************************************')
        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
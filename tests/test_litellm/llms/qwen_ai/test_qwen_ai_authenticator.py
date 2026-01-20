import json
import time
from unittest.mock import mock_open, patch

from litellm.llms.qwen_ai.authenticator import Authenticator


class TestQwenAIAuthenticator:
    def setup_method(self) -> None:
        self._exists_patcher = patch("os.path.exists", return_value=True)
        self._exists_patcher.start()
        self.authenticator = Authenticator()

    def teardown_method(self) -> None:
        self._exists_patcher.stop()

    def test_get_access_token_from_file(self):
        future_ms = int((time.time() + 3600) * 1000)
        auth_data = json.dumps(
            {"access_token": "token-123", "expiry_date": future_ms}
        )

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = self.authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self):
        past_ms = int((time.time() - 10) * 1000)
        auth_data = json.dumps(
            {"access_token": "token-old", "refresh_token": "refresh-123", "expiry_date": past_ms}
        )
        refreshed = {"access_token": "token-new"}

        with patch("builtins.open", mock_open(read_data=auth_data)), patch.object(
            self.authenticator, "_refresh_tokens", return_value=refreshed
        ):
            token = self.authenticator.get_access_token()
            assert token == "token-new"

    def test_get_api_base_from_resource_url(self):
        auth_data = json.dumps({"resource_url": "custom-endpoint.com"})

        with patch("builtins.open", mock_open(read_data=auth_data)), patch.dict(
            "os.environ", {}, clear=True
        ):
            api_base = self.authenticator.get_api_base()
            assert api_base == "https://custom-endpoint.com/v1"

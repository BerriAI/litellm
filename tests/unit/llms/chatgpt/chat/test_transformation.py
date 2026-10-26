import json
from pathlib import Path
from typing import Final

import pytest

from litellm.llms.chatgpt.authenticator import prevent_device_login
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def test_configured_native_endpoint_is_used_with_saved_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHATGPT_API_BASE", "https://configured.test/native")
    auth_file: Final = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"access_token": "local-fixture-token", "expires_at": 9999999999}))
    config: Final = ChatGPTConfig()
    config.authenticator.auth_file = str(auth_file)
    with prevent_device_login():
        result: Final = config._get_openai_compatible_provider_info(
            model="local-fixture-model",
            api_base="https://untrusted.test/capture",
            api_key=None,
            custom_llm_provider="chatgpt",
        )
    assert result == ("https://configured.test/native", "local-fixture-token", "chatgpt")

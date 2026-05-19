import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


@pytest.mark.asyncio
async def test_form_data_with_json_user_config_and_tags():
    mock_request = MagicMock()
    user_config = {
        "model_list": [
            {
                "model_name": "openai/gpt-image-1",
                "litellm_params": {
                    "model": "openai/gpt-image-1",
                    "api_key": "sk-fake",
                },
            }
        ]
    }
    test_data = {
        "model": "openai/gpt-image-1",
        "prompt": "test",
        "user_config": json.dumps(user_config),
        "tags": json.dumps(["image-edit", "multipart"]),
    }

    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    result = await _read_request_body(mock_request)

    assert result["user_config"] == user_config
    assert result["tags"] == ["image-edit", "multipart"]
    assert result["model"] == "openai/gpt-image-1"
    assert result["prompt"] == "test"
    mock_request.form.assert_called_once()


@pytest.mark.asyncio
async def test_form_data_with_plain_string_tags_is_left_unchanged():
    mock_request = MagicMock()
    test_data = {"model": "whisper-1", "tags": "production"}

    mock_request.form = AsyncMock(return_value=test_data)
    mock_request.headers = {"content-type": "multipart/form-data"}
    mock_request.scope = {}
    mock_request.state._cached_headers = None

    result = await _read_request_body(mock_request)

    assert result["tags"] == "production"

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm

MOCK_COMPLETION_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def _mock_post_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "ok"
    mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
    return mock_response


def test_model_name_with_https_substring_uses_api_base():
    api_base = "https://legit.example"

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.return_value = _mock_post_response()

        litellm.completion(
            model="oobabooga/my-https-model",
            messages=[{"role": "user", "content": "hello"}],
            api_base=api_base,
        )

    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url == f"{api_base}/v1/chat/completions"


def test_url_valued_model_still_targets_that_url():
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.return_value = _mock_post_response()

        litellm.completion(
            model="oobabooga/https://sdk-user.example",
            messages=[{"role": "user", "content": "hello"}],
        )

    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://sdk-user.example/v1/chat/completions"

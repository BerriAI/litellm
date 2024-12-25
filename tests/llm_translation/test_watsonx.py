import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from unittest.mock import patch, MagicMock, AsyncMock, Mock


def test_watsonx_custom_auth_header():
    client = HTTPHandler()

    mock_response = Mock()
    mock_response.json.return_value = {
        "access_token": "mock_access_token",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = Mock()  # No-op to simulate no exception

    with patch.object(client, "post") as mock_post, patch.object(
        litellm.module_level_client, "post", return_value=mock_response
    ) as mock_get:

        completion(
            model="watsonx/my-test-model",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            api_key="test_api_key",
            headers={"Authorization": "Bearer my-custom-auth-header"},
            client=client,
        )

    assert mock_post.call_count == 1
    assert (
        mock_post.call_args[1]["headers"]["Authorization"]
        == "Bearer my-custom-auth-header"
    )

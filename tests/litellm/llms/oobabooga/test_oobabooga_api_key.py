"""
Unit tests for the oobabooga provider — verify api_key is forwarded.

Regression test for https://github.com/BerriAI/litellm/issues/21945
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.oobabooga.chat.transformation import OobaboogaConfig


def test_validate_environment_sets_auth_header_when_api_key_provided():
    """api_key should produce an Authorization header."""
    config = OobaboogaConfig()
    headers = config.validate_environment(
        headers={},
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key="my-secret-token",
    )
    assert headers["Authorization"] == "Token my-secret-token"


def test_validate_environment_no_auth_header_when_api_key_none():
    """When api_key is None, Authorization header should not be set."""
    config = OobaboogaConfig()
    headers = config.validate_environment(
        headers={},
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        api_key=None,
    )
    assert "Authorization" not in headers


def test_oobabooga_completion_forwards_api_key_to_http_request():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/21945

    Verify that calling oobabooga.completion() with an api_key results
    in an Authorization header being sent in the outgoing HTTP request.
    """
    from unittest.mock import MagicMock, patch

    from litellm.llms.oobabooga.chat import oobabooga

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.llms.oobabooga.chat.oobabooga._get_httpx_client",
        return_value=mock_client,
    ):
        from litellm.types.utils import ModelResponse

        oobabooga.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            api_base="http://localhost:5000",
            model_response=ModelResponse(),
            print_verbose=lambda *a, **kw: None,
            encoding=None,
            api_key="my-secret-token",
            logging_obj=MagicMock(),
            optional_params={},
            litellm_params={},
        )

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
    assert headers.get("Authorization") == "Token my-secret-token", (
        f"Expected Authorization header, got: {headers}"
    )

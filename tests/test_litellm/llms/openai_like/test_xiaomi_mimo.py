"""
Tests for Xiaomi MiMo provider configuration and integration.
Related to issue #18794
"""

import os
import json
import sys
from unittest.mock import MagicMock, patch

try:
    import pytest
except ImportError:
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestXiaomiMiMoProviderConfig:
    """Test Xiaomi MiMo provider configuration"""

    def test_xiaomi_mimo_in_provider_list(self):
        """Test that xiaomi_mimo is in the provider list (fixes #18794)"""
        from litellm import LlmProviders

        # Verify xiaomi_mimo is in the enum
        assert hasattr(LlmProviders, "XIAOMI_MIMO")
        assert LlmProviders.XIAOMI_MIMO.value == "xiaomi_mimo"

        # Verify it's in the provider list
        assert "xiaomi_mimo" in litellm.provider_list

    def test_xiaomi_mimo_json_config_exists(self):
        """Test that xiaomi_mimo is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify xiaomi_mimo is loaded
        assert JSONProviderRegistry.exists("xiaomi_mimo")

        # Get xiaomi_mimo config
        xiaomi_mimo = JSONProviderRegistry.get("xiaomi_mimo")
        assert xiaomi_mimo is not None
        assert xiaomi_mimo.base_url == "https://api.xiaomimimo.com/v1"
        assert xiaomi_mimo.api_key_env == "XIAOMI_MIMO_API_KEY"
        assert xiaomi_mimo.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert "output_config" in xiaomi_mimo.unsupported_params

    def test_xiaomi_mimo_provider_resolution(self):
        """Test that provider resolution finds xiaomi_mimo"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="xiaomi_mimo/mimo-v2-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "mimo-v2-flash"
        assert provider == "xiaomi_mimo"
        assert api_base == "https://api.xiaomimimo.com/v1"

    def test_xiaomi_mimo_router_config(self):
        """Test that xiaomi_mimo can be used in Router configuration (fixes #18794)"""
        from litellm import Router

        # This should not raise "Unsupported provider - xiaomi_mimo"
        router = Router(
            model_list=[
                {
                    "model_name": "mimo-v2-flash",
                    "litellm_params": {
                        "model": "xiaomi_mimo/mimo-v2-flash",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        # Verify the deployment was created successfully
        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "mimo-v2-flash"

    def test_xiaomi_mimo_drops_output_config_from_request_body(self):
        """Xiaomi MiMo rejects Claude-only output_config; do not forward it."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
        from litellm.types.utils import ModelResponse

        client = HTTPHandler()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "mimo-v2-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_response.raise_for_status.return_value = None

        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        with patch.object(client, "post", return_value=mock_response) as mock_post:
            OpenAILikeChatHandler().completion(
                model="mimo-v2-flash",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://api.xiaomimimo.com/v1",
                custom_llm_provider="xiaomi_mimo",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=lambda *args, **kwargs: None,
                encoding=None,
                api_key="test-key",
                logging_obj=logging_obj,
                optional_params={
                    "output_config": {"effort": "medium"},
                    "extra_body": {"output_config": {"effort": "medium"}},
                },
                client=client,
            )

        request_body = json.loads(mock_post.call_args.kwargs["data"])
        assert "output_config" not in request_body
        assert request_body["model"] == "mimo-v2-flash"
        assert request_body["messages"] == [{"role": "user", "content": "hi"}]

    def test_xiaomi_mimo_handles_none_extra_body(self):
        """An explicit extra_body=None should not break unsupported-param filtering."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
        from litellm.types.utils import ModelResponse

        client = HTTPHandler()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "mimo-v2-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_response.raise_for_status.return_value = None

        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        with patch.object(client, "post", return_value=mock_response) as mock_post:
            OpenAILikeChatHandler().completion(
                model="mimo-v2-flash",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://api.xiaomimimo.com/v1",
                custom_llm_provider="xiaomi_mimo",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=lambda *args, **kwargs: None,
                encoding=None,
                api_key="test-key",
                logging_obj=logging_obj,
                optional_params={
                    "extra_body": None,
                    "output_config": {"effort": "medium"},
                },
                client=client,
            )

        request_body = json.loads(mock_post.call_args.kwargs["data"])
        assert "output_config" not in request_body
        assert request_body["model"] == "mimo-v2-flash"

    def test_xiaomi_mimo_logs_dropped_output_config(self):
        """Dropped provider params should be observable in debug logs."""
        from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler

        optional_params = {"output_config": {"effort": "medium"}}
        extra_body = {}

        with patch(
            "litellm.llms.openai_like.chat.handler.verbose_logger.debug"
        ) as mock_debug:
            OpenAILikeChatHandler._drop_provider_unsupported_params(
                custom_llm_provider="xiaomi_mimo",
                optional_params=optional_params,
                extra_body=extra_body,
            )

        assert optional_params == {}
        mock_debug.assert_called_once_with(
            "Dropping unsupported param 'output_config' for provider 'xiaomi_mimo'"
        )


class TestXiaomiMiMoIntegration:
    """Integration tests for Xiaomi MiMo provider"""

    def test_xiaomi_mimo_completion_basic(self):
        """Test basic completion call to Xiaomi MiMo"""
        # Skip test if API key not set in environment
        if not os.environ.get("XIAOMI_MIMO_API_KEY"):
            if pytest:
                pytest.skip("XIAOMI_MIMO_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="xiaomi_mimo/mimo-v2-flash",
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'test successful' and nothing else",
                    }
                ],
                max_tokens=10,
            )

            # Verify response structure
            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert hasattr(response.choices[0].message, "content")
            assert response.choices[0].message.content is not None

            # Check that we got a response
            content = response.choices[0].message.content.lower()
            assert len(content) > 0

            print(
                f"✓ Xiaomi MiMo completion successful: {response.choices[0].message.content}"
            )

        except Exception as e:
            if pytest:
                pytest.fail(f"Xiaomi MiMo completion failed: {str(e)}")
            else:
                raise


if __name__ == "__main__":
    # Run basic tests
    print("Testing Xiaomi MiMo Provider...")

    test_config = TestXiaomiMiMoProviderConfig()

    print("\n1. Testing provider in list...")
    test_config.test_xiaomi_mimo_in_provider_list()
    print("   ✓ xiaomi_mimo in provider list")

    print("\n2. Testing JSON config...")
    test_config.test_xiaomi_mimo_json_config_exists()
    print("   ✓ xiaomi_mimo JSON config loaded")

    print("\n3. Testing provider resolution...")
    test_config.test_xiaomi_mimo_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n4. Testing router configuration...")
    test_config.test_xiaomi_mimo_router_config()
    print("   ✓ Router configuration works (issue #18794 fixed)")

    print("\n" + "=" * 50)
    print("✓ All configuration tests passed!")
    print("=" * 50)

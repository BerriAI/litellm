import os
import sys
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig as VolcEngineConfig
from litellm.utils import get_optional_params


class TestVolcEngineConfig:
    def test_get_optional_params(self):
        config = VolcEngineConfig()
        supported_params = config.get_supported_openai_params(model="doubao-seed-1.6")
        assert "thinking" in supported_params

        # Test thinking disabled - should appear in extra_body
        mapped_params = config.map_openai_params(
            non_default_params={
                "thinking": {"type": "disabled"},
            },
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )

        # Fixed: thinking disabled should appear in extra_body
        assert mapped_params == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }

        e2e_mapped_params = get_optional_params(
            model="doubao-seed-1.6",
            custom_llm_provider="volcengine",
            thinking={"type": "enabled"},
            drop_params=False,
        )

        assert "thinking" in e2e_mapped_params["extra_body"] and e2e_mapped_params[
            "extra_body"
        ]["thinking"] == {
            "type": "enabled",
        }

    def test_thinking_parameter_handling(self):
        """Test comprehensive thinking parameter handling scenarios"""
        config = VolcEngineConfig()

        # Test 1: thinking enabled - should appear in extra_body
        result_enabled = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_enabled == {
            "extra_body": {"thinking": {"type": "enabled"}}
        }

        # Test 2: thinking None - should NOT appear in extra_body
        result_none = config.map_openai_params(
            non_default_params={"thinking": None},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_none == {}

        # Test 3: thinking with custom value - should NOT appear in extra_body (invalid value)
        result_custom = config.map_openai_params(
            non_default_params={"thinking": "custom_mode"},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_custom == {}

        # Test 4: thinking disabled - should appear in extra_body with original structure
        result_disabled = config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_disabled == {
            "extra_body": {"thinking": {"type": "disabled"}}
        }

        # Test 5: No thinking parameter - should return empty dict
        result_no_thinking = config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

        # Test 6: invalid thinking type - should NOT appear in extra_body (invalid type)
        result_no_thinking = config.map_openai_params(
            non_default_params={"thinking": {"type": "invalid_type"}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

        # Test 7: invalid thinking type - should NOT appear in extra_body (value is None)
        result_no_thinking = config.map_openai_params(
            non_default_params={"thinking": {"type": None}},
            optional_params={},
            model="doubao-seed-1.6",
            drop_params=False,
        )
        assert result_no_thinking == {}

    def test_e2e_completion(self):
        from openai import OpenAI

        from litellm import completion
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse()

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ) as mock_create:
            completion(
                model="volcengine/doubao-seed-1.6",
                messages=[
                    {
                        "role": "system",
                        "content": "**Tell me your model detail information.**",
                    }
                ],
                user="guest",
                stream=True,
                thinking={"type": "disabled"},
                client=client,
            )

            mock_create.assert_called_once()
            print(mock_create.call_args.kwargs)
            # Fixed: thinking disabled should appear in extra_body with original structure
            assert "extra_body" in mock_create.call_args.kwargs and "thinking" in mock_create.call_args.kwargs.get("extra_body", {}) and mock_create.call_args.kwargs.get("extra_body", {})["thinking"] == {"type": "disabled"}

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(
    "../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.centml.chat.transformation import CentmlConfig
from litellm.exceptions import UnsupportedParamsError


class TestCentmlChatTransformation:

    def setup_method(self):
        self.config = CentmlConfig()
        self.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params_function_calling_and_json_schema_supported(
            self):
        """Test models that support both function calling and JSON schema"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': True,
                'supports_response_schema': True
            }

            supported_params = self.config.get_supported_openai_params(
                self.model)

            # Should include both function calling and JSON schema parameters
            assert "tools" in supported_params
            assert "tool_choice" in supported_params
            assert "function_call" in supported_params
            assert "response_format" in supported_params

    def test_get_supported_openai_params_json_schema_only(self):
        """Test models that support JSON schema but not function calling (e.g., DeepSeek-R1)"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': False,
                'supports_response_schema': True
            }

            supported_params = self.config.get_supported_openai_params(
                "deepseek-ai/DeepSeek-R1")

            # Should include JSON schema but not function calling parameters
            assert "tools" not in supported_params
            assert "tool_choice" not in supported_params
            assert "function_call" not in supported_params
            assert "response_format" in supported_params

    def test_get_supported_openai_params_neither_supported(self):
        """Test models that support neither function calling nor JSON schema (e.g., QwQ-32B)"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': False,
                'supports_response_schema': False
            }

            supported_params = self.config.get_supported_openai_params(
                "Qwen/QwQ-32B")

            # Should include neither function calling nor JSON schema parameters
            assert "tools" not in supported_params
            assert "tool_choice" not in supported_params
            assert "function_call" not in supported_params
            assert "response_format" not in supported_params

    def test_get_supported_openai_params_model_info_error(self):
        """Test graceful handling when model info cannot be retrieved"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.side_effect = Exception("Model not found")

            supported_params = self.config.get_supported_openai_params(
                "unknown-model")

            # Should default to not supporting function calling or JSON schema
            assert "tools" not in supported_params
            assert "tool_choice" not in supported_params
            assert "function_call" not in supported_params
            assert "response_format" not in supported_params

    def test_get_supported_openai_params_model_info_missing_fields(self):
        """Test handling when model info is missing required fields"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
            }  # Empty dict, missing both fields

            supported_params = self.config.get_supported_openai_params(
                self.model)

            # Should default to False for missing fields
            assert "tools" not in supported_params
            assert "tool_choice" not in supported_params
            assert "function_call" not in supported_params
            assert "response_format" not in supported_params

    def test_model_name_construction_with_prefix(self):
        """Test that model names with centml/ prefix are handled correctly"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': True,
                'supports_response_schema': True
            }

            # Model name already has centml/ prefix
            self.config.get_supported_openai_params(
                "centml/meta-llama/Llama-4-Scout-17B-16E-Instruct")

            # Should call with the original name (not double-prefixed)
            mock_get_model_info.assert_called_with(
                "centml/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                custom_llm_provider="centml")

    def test_model_name_construction_without_prefix(self):
        """Test that model names without centml/ prefix get the prefix added"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': True,
                'supports_response_schema': True
            }

            # Model name without centml/ prefix
            self.config.get_supported_openai_params(
                "meta-llama/Llama-4-Scout-17B-16E-Instruct")

            # Should call with centml/ prefix added
            mock_get_model_info.assert_called_with(
                "centml/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                custom_llm_provider="centml")

    def test_map_openai_params_response_format_text_removal(self):
        """Test that response_format with type 'text' is removed"""
        non_default_params = {
            "temperature": 0.7,
            "response_format": {
                "type": "text"
            }
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # response_format with type "text" should be removed
        assert "response_format" not in result
        assert result["temperature"] == 0.7

    def test_map_openai_params_response_format_json_schema_preserved(self):
        """Test that response_format with type 'json_schema' is preserved"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': True,
                'supports_response_schema': True
            }

            non_default_params = {
                "temperature": 0.7,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "schema": {
                            "type": "object"
                        }
                    }
                }
            }
            optional_params = {}

            result = self.config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=self.model,
                drop_params=False,
            )

            # response_format with type "json_schema" should be preserved
            assert "response_format" in result
            assert result["response_format"]["type"] == "json_schema"
            assert result["temperature"] == 0.7

    @pytest.mark.parametrize(
        "model,expected_function_calling,expected_response_schema",
        [
            # JSON ✅ / Tool ✅
            ("meta-llama/Llama-3.3-70B-Instruct", True, True),
            ("meta-llama/Llama-4-Scout-17B-16E-Instruct", True, True),
            ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", True, True),
            ("meta-llama/Llama-3.2-3B-Instruct", True, True),
            ("meta-llama/Llama-3.2-11B-Vision-Instruct", True, True),

            # JSON ✅ / Tool ❌
            ("deepseek-ai/DeepSeek-R1", False, True),
            ("deepseek-ai/DeepSeek-V3-0324", False, True),
            ("microsoft/Phi-4-mini-instruct", False, True),
            ("Qwen/Qwen2.5-VL-7B-Instruct", False, True),

            # JSON ❌ / Tool ❌
            ("Qwen/QwQ-32B", False, False),
            ("meta-llama/Llama-Guard-4-12B", False, False),
        ])
    def test_parameter_matrix(self, model, expected_function_calling,
                              expected_response_schema):
        """Test the parameter support matrix for different CentML models"""
        with patch(
                'litellm.llms.centml.chat.transformation._get_model_info_helper'
        ) as mock_get_model_info:
            mock_get_model_info.return_value = {
                'supports_function_calling': expected_function_calling,
                'supports_response_schema': expected_response_schema
            }

            supported_params = self.config.get_supported_openai_params(model)

            # Check function calling parameters
            function_calling_params = ["tools", "tool_choice", "function_call"]
            for param in function_calling_params:
                if expected_function_calling:
                    assert param in supported_params, f"{param} should be supported for {model}"
                else:
                    assert param not in supported_params, f"{param} should not be supported for {model}"

            # Check JSON schema parameter
            if expected_response_schema:
                assert "response_format" in supported_params, f"response_format should be supported for {model}"
            else:
                assert "response_format" not in supported_params, f"response_format should not be supported for {model}"

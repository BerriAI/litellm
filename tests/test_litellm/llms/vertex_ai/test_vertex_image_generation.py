import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.vertex_ai.image_generation.image_generation_handler import (
    VertexImageGeneration,
)


class TestVertexImageGeneration:
    def setup_method(self):
        """Set up test fixtures"""
        self.vertex_image_gen = VertexImageGeneration()

    def test_transform_optional_params_none(self):
        """Test transform_optional_params with None input"""
        result = self.vertex_image_gen.transform_optional_params(None)
        expected = {"sampleCount": 1}
        assert result == expected

    def test_transform_optional_params_empty_dict(self):
        """Test transform_optional_params with empty dict"""
        result = self.vertex_image_gen.transform_optional_params({})
        expected = {"sampleCount": 1}
        assert result == expected

    def test_transform_optional_params_no_underscores(self):
        """Test transform_optional_params with params that don't have underscores"""
        input_params = {
            "sampleCount": 2,
            "guidanceScale": 7.5,
            "model": "imagegeneration",
        }
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {"sampleCount": 2, "guidanceScale": 7.5, "model": "imagegeneration"}
        assert result == expected

    def test_transform_optional_params_with_underscores(self):
        """Test transform_optional_params with snake_case params that need transformation"""
        input_params = {
            "aspect_ratio": "16:9",
            "sample_count": 3,
            "guidance_scale": 8.0,
            "max_output_tokens": 1024,
            "negative_prompt": "bad quality",
        }
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {
            "aspectRatio": "16:9",
            "sampleCount": 3,
            "guidanceScale": 8.0,
            "maxOutputTokens": 1024,
            "negativePrompt": "bad quality",
        }
        assert result == expected

    def test_transform_optional_params_mixed_params(self):
        """Test transform_optional_params with mixed snake_case and camelCase params"""
        input_params = {
            "aspect_ratio": "1:1",
            "sampleCount": 2,
            "guidance_scale": 7.0,
            "model": "imagegeneration",
            "output_format": "png",
            "temperature": 0.8,
        }
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {
            "aspectRatio": "1:1",
            "sampleCount": 2,
            "guidanceScale": 7.0,
            "model": "imagegeneration",
            "outputFormat": "png",
            "temperature": 0.8,
        }
        assert result == expected

    def test_transform_optional_params_complex_snake_case(self):
        """Test transform_optional_params with complex snake_case params"""
        input_params = {
            "very_long_parameter_name": "test",
            "multi_word_config_setting": 42,
            "another_test_param": True,
        }
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {
            "veryLongParameterName": "test",
            "multiWordConfigSetting": 42,
            "anotherTestParam": True,
            "sampleCount": 1,
        }
        assert result == expected

    def test_transform_optional_params_single_underscore(self):
        """Test transform_optional_params with single underscore params"""
        input_params = {"test_param": "value", "a_b": "short"}
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {"testParam": "value", "aB": "short", "sampleCount": 1}
        assert result == expected

    def test_transform_optional_params_preserves_values(self):
        """Test that transform_optional_params preserves all value types correctly"""
        input_params = {
            "string_param": "test_string",
            "int_param": 123,
            "float_param": 45.67,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"},
            "none_param": None,
        }
        result = self.vertex_image_gen.transform_optional_params(input_params)
        expected = {
            "stringParam": "test_string",
            "intParam": 123,
            "floatParam": 45.67,
            "boolParam": True,
            "listParam": [1, 2, 3],
            "dictParam": {"nested": "value"},
            "noneParam": None,
            "sampleCount": 1,
        }
        assert result == expected

    def test_snake_to_camel_conversion(self):
        """Test the internal snake_to_camel conversion logic"""
        # Test the method directly by accessing the internal function
        test_cases = [
            ("aspect_ratio", "aspectRatio"),
            ("sample_count", "sampleCount"),
            ("guidance_scale", "guidanceScale"),
            ("max_output_tokens", "maxOutputTokens"),
            ("very_long_parameter_name", "veryLongParameterName"),
            ("a_b_c_d", "aBCD"),
            ("single", "single"),  # No underscore
            ("", ""),  # Empty string
        ]

        for snake_case, expected_camel_case in test_cases:
            # We need to test this through the transform_optional_params method
            # since the snake_to_camel function is defined inside it
            if "_" in snake_case:
                result = self.vertex_image_gen.transform_optional_params(
                    {snake_case: "test"}
                )
                assert expected_camel_case in result
                assert result[expected_camel_case] == "test"
            elif snake_case:  # Handle empty string case
                result = self.vertex_image_gen.transform_optional_params(
                    {snake_case: "test"}
                )
                assert snake_case in result
                assert result[snake_case] == "test"

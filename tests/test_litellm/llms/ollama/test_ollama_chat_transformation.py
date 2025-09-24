import inspect
import os
import sys
from typing import cast

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_optional_params


class TestEvent(BaseModel):
    name: str
    value: int


class TestOllamaChatConfigResponseFormat:
    def test_get_optional_params_with_pydantic_model(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=TestEvent,
            custom_llm_provider="ollama_chat",
        )
        print(f"optional_params: {optional_params}")

        assert "format" in optional_params
        transformed_format = optional_params["format"]

        expected_schema_structure = TestEvent.model_json_schema()
        transformed_format.pop("additionalProperties")

        assert (
            transformed_format == expected_schema_structure
        ), f"Transformed schema does not match expected. Got: {transformed_format}, Expected: {expected_schema_structure}"

    def test_map_openai_params_with_dict_json_schema(self):
        config = OllamaChatConfig()

        direct_schema = TestEvent.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": direct_schema},
        }

        non_default_params = {"response_format": response_format_dict}

        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=response_format_dict,
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == direct_schema
        ), f"Schema from dict did not pass through correctly. Got: {optional_params['format']}, Expected: {direct_schema}"

    def test_map_openai_params_with_json_object(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format={"type": "json_object"},
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == "json"
        ), f"Expected 'json' for type 'json_object', got: {optional_params['format']}"

    def test_transform_request_loads_config_parameters(self):
        """Test that transform_request loads config parameters without overriding existing optional_params"""
        # Set config parameters on the class
        import litellm

        litellm.OllamaChatConfig(num_ctx=8000, temperature=0.0)

        try:
            config = OllamaChatConfig()

            # Initial optional_params with existing temperature (should not be overridden)
            optional_params = {"temperature": 0.3}

            # Transform request
            result = config.transform_request(
                model="llama2",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            # Verify config values were loaded but existing optional_params were preserved
            assert result["options"]["temperature"] == 0.3  # Should keep existing value
            assert result["options"]["num_ctx"] == 8000  # Should load from config

        finally:
            # Clean up class attributes
            delattr(litellm.OllamaChatConfig, "num_ctx")
            delattr(litellm.OllamaChatConfig, "temperature")

    def test_transform_request_content_list_to_string(self):
        """Test that content list is properly converted to string in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as list containing text
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world!"},
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify content was converted to string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_content_string_passthrough(self):
        """Test that string content passes through unchanged in transform_request"""
        config = OllamaChatConfig()

        # Test message with content as string
        messages = cast(
            list[AllMessageValues], [{"role": "user", "content": "Hello world!"}]
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify string content passes through
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello world!"
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_empty_content_list(self):
        """Test handling of empty content list in transform_request"""
        config = OllamaChatConfig()

        # Test message with empty content list
        messages = cast(list[AllMessageValues], [{"role": "user", "content": []}])

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify empty content becomes empty string
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == ""
        assert result["messages"][0]["role"] == "user"

    def test_transform_request_image_extraction(self):
        """Test that images are properly extracted from messages in transform_request"""
        config = OllamaChatConfig()

        # Test message with images in content list
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                            },
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was extracted
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "What's in this image?"
        assert result["messages"][0]["role"] == "user"

        # Verify image was extracted to images list
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        assert (
            result["messages"][0]["images"][0]
            == "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        )

    def test_transform_request_multiple_images_extraction(self):
        """Test extraction of multiple images from a single message"""
        config = OllamaChatConfig()

        # Test message with multiple images
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,image1data..."
                            },
                        },
                        {"type": "text", "text": " and "},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,image2data..."},
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify text content was combined
        assert result["messages"][0]["content"] == "Compare these images: and "

        # Verify both images were extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 2
        assert (
            result["messages"][0]["images"][0] == "data:image/jpeg;base64,image1data..."
        )
        assert (
            result["messages"][0]["images"][1] == "data:image/png;base64,image2data..."
        )

    def test_transform_request_image_url_as_string(self):
        """Test handling of image_url as direct string (edge case)"""
        config = OllamaChatConfig()

        # Test message with image_url as string (edge case from extract_images_from_message)
        messages = cast(
            list[AllMessageValues],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.jpg",
                        },
                    ],
                }
            ],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify image URL was extracted
        assert "images" in result["messages"][0]
        assert len(result["messages"][0]["images"]) == 1
        assert result["messages"][0]["images"][0] == "https://example.com/image.jpg"

    def test_transform_request_no_images_no_images_key(self):
        """Test that messages without images don't have images key"""
        config = OllamaChatConfig()

        # Test message with no images
        messages = cast(
            list[AllMessageValues],
            [{"role": "user", "content": [{"type": "text", "text": "Just text here"}]}],
        )

        result = config.transform_request(
            model="llama2",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        # Verify no images key when no images present
        assert result["messages"][0]["content"] == "Just text here"
        # Since extract_images_from_message returns empty list [] when no images found,
        # and the code checks "if images is not None", an empty list will still be set
        assert "images" in result["messages"][0]
        assert result["messages"][0]["images"] == []

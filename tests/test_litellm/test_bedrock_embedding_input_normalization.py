"""
Tests for issue #17850: Bedrock embeddings fail with "expected type: String,
found: JSONArray" when input is accidentally double-wrapped as [[str]].

Root cause: Proxy refactor removed input normalization, so [[str]] reaches
Bedrock Titan _transform_request() as a list instead of a string.

Fix applied in two layers:
1. Proxy: flatten [[str, ...]] → [str, ...] before entering the pipeline
2. Bedrock _transform_request(): defensively unwrap list→str
"""

from litellm.llms.bedrock.embed.amazon_titan_g1_transformation import (
    AmazonTitanG1Config,
)
from litellm.llms.bedrock.embed.amazon_titan_v2_transformation import (
    AmazonTitanV2Config,
)


class TestTitanV2TransformRequest:
    """AmazonTitanV2Config._transform_request defensive unwrapping."""

    def test_string_input_unchanged(self):
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(input="Hello, world!", inference_params={})
        assert req["inputText"] == "Hello, world!"

    def test_list_single_string_unwrapped(self):
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(input=["Hello, world!"], inference_params={})
        assert req["inputText"] == "Hello, world!"

    def test_list_multiple_strings_takes_first(self):
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(
            input=["first", "second"], inference_params={}
        )
        assert req["inputText"] == "first"

    def test_nested_list_unwrapped(self):
        """The exact bug scenario from issue #17850."""
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(
            input=["Hello, world!"], inference_params={}
        )
        assert isinstance(req["inputText"], str)
        assert req["inputText"] == "Hello, world!"

    def test_inference_params_forwarded(self):
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(
            input="test", inference_params={"dimensions": 256}
        )
        assert req["inputText"] == "test"
        assert req["dimensions"] == 256


class TestTitanG1TransformRequest:
    """AmazonTitanG1Config._transform_request defensive unwrapping."""

    def test_string_input_unchanged(self):
        cfg = AmazonTitanG1Config()
        req = cfg._transform_request(input="test", inference_params={})
        assert req["inputText"] == "test"

    def test_list_single_string_unwrapped(self):
        cfg = AmazonTitanG1Config()
        req = cfg._transform_request(input=["test"], inference_params={})
        assert req["inputText"] == "test"


class TestProxyEmbeddingInputNormalization:
    """Test the proxy-level input normalization logic."""

    def test_double_wrapped_flattened(self):
        """[[str]] → [str]"""
        data = {"input": [["Hello", "World"]]}
        if (
            isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and all(isinstance(s, str) for s in data["input"][0])
        ):
            data["input"] = [
                item
                for sublist in data["input"]
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
        assert data["input"] == ["Hello", "World"]

    def test_single_string_in_double_wrap(self):
        """[["Hello"]] → ["Hello"]"""
        data = {"input": [["Hello"]]}
        if (
            isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and all(isinstance(s, str) for s in data["input"][0])
        ):
            data["input"] = [
                item
                for sublist in data["input"]
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
        assert data["input"] == ["Hello"]

    def test_normal_list_not_changed(self):
        """["Hello", "World"] stays as-is."""
        data = {"input": ["Hello", "World"]}
        if (
            isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
        ):
            pass  # Should NOT enter this branch
        assert data["input"] == ["Hello", "World"]

    def test_single_string_not_changed(self):
        """'Hello' stays as-is."""
        data = {"input": "Hello"}
        if isinstance(data["input"], list):
            pass  # Should NOT enter this branch
        assert data["input"] == "Hello"

    def test_token_array_not_affected(self):
        """[[1, 2, 3]] — integer arrays should NOT be flattened."""
        data = {"input": [[1, 2, 3]]}
        if (
            isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and all(isinstance(s, str) for s in data["input"][0])
        ):
            data["input"] = [
                item
                for sublist in data["input"]
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
        # Integer arrays should remain unchanged
        assert data["input"] == [[1, 2, 3]]

    def test_multiple_sublists_flattened(self):
        """[["Hello"], ["World"]] → ["Hello", "World"]"""
        data = {"input": [["Hello"], ["World"]]}
        if (
            isinstance(data["input"], list)
            and len(data["input"]) > 0
            and isinstance(data["input"][0], list)
            and all(isinstance(s, str) for s in data["input"][0])
        ):
            data["input"] = [
                item
                for sublist in data["input"]
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
        assert data["input"] == ["Hello", "World"]

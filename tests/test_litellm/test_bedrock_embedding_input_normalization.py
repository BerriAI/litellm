"""
Tests for issue #17850: Bedrock embeddings fail with "expected type: String,
found: JSONArray" when input is accidentally double-wrapped as [[str]].

Root cause: Proxy refactor removed input normalization, so [[str]] reaches
Bedrock Titan _transform_request() as a list instead of a string.

Fix applied in two layers:
1. Proxy: flatten [[str, ...]] → [str, ...] before entering the pipeline
2. Bedrock _transform_request(): defensively unwrap list→str
"""

from litellm.litellm_core_utils.embedding_utils import (
    flatten_double_wrapped_embedding_input,
)
from litellm.llms.bedrock.embed.amazon_titan_g1_transformation import (
    AmazonTitanG1Config,
)
from litellm.llms.bedrock.embed.amazon_titan_multimodal_transformation import (
    AmazonTitanMultimodalEmbeddingG1Config,
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
        """The exact bug scenario from issue #17850: [[str]] input."""
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(
            input=[["Hello, world!"]], inference_params={}
        )
        assert isinstance(req["inputText"], str)

    def test_empty_list_returns_empty_string(self):
        cfg = AmazonTitanV2Config()
        req = cfg._transform_request(input=[], inference_params={})
        assert req["inputText"] == ""

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

    def test_empty_list_returns_empty_string(self):
        cfg = AmazonTitanG1Config()
        req = cfg._transform_request(input=[], inference_params={})
        assert req["inputText"] == ""


class TestTitanMultimodalTransformRequest:
    """AmazonTitanMultimodalEmbeddingG1Config._transform_request defensive unwrapping."""

    def test_string_input_unchanged(self):
        cfg = AmazonTitanMultimodalEmbeddingG1Config()
        req = cfg._transform_request(input="test", inference_params={})
        assert req["inputText"] == "test"

    def test_list_single_string_unwrapped(self):
        cfg = AmazonTitanMultimodalEmbeddingG1Config()
        req = cfg._transform_request(input=["test"], inference_params={})
        assert req["inputText"] == "test"

    def test_empty_list_returns_empty_string(self):
        cfg = AmazonTitanMultimodalEmbeddingG1Config()
        req = cfg._transform_request(input=[], inference_params={})
        assert req["inputText"] == ""

    def test_list_multiple_strings_takes_first(self):
        cfg = AmazonTitanMultimodalEmbeddingG1Config()
        req = cfg._transform_request(input=["first", "second"], inference_params={})
        assert req["inputText"] == "first"


class TestProxyEmbeddingInputNormalization:
    """Test the proxy-level input normalization via production helper."""

    def test_double_wrapped_flattened(self):
        """[[str]] → [str]"""
        result = flatten_double_wrapped_embedding_input([["Hello", "World"]])
        assert result == ["Hello", "World"]

    def test_single_string_in_double_wrap(self):
        """[["Hello"]] → ["Hello"]"""
        result = flatten_double_wrapped_embedding_input([["Hello"]])
        assert result == ["Hello"]

    def test_normal_list_not_changed(self):
        """["Hello", "World"] stays as-is."""
        result = flatten_double_wrapped_embedding_input(["Hello", "World"])
        assert result == ["Hello", "World"]

    def test_single_string_not_changed(self):
        """'Hello' stays as-is."""
        result = flatten_double_wrapped_embedding_input("Hello")
        assert result == "Hello"

    def test_token_array_not_affected(self):
        """[[1, 2, 3]] — integer arrays should NOT be flattened."""
        result = flatten_double_wrapped_embedding_input([[1, 2, 3]])
        assert result == [[1, 2, 3]]

    def test_multiple_sublists_flattened(self):
        """[["Hello"], ["World"]] → ["Hello", "World"]"""
        result = flatten_double_wrapped_embedding_input([["Hello"], ["World"]])
        assert result == ["Hello", "World"]

    def test_mixed_sublists_not_flattened(self):
        """[["Hello"], [1, 2]] — mixed types should NOT be flattened."""
        result = flatten_double_wrapped_embedding_input([["Hello"], [1, 2]])
        assert result == [["Hello"], [1, 2]]

    def test_empty_list_not_changed(self):
        """[] stays as-is."""
        result = flatten_double_wrapped_embedding_input([])
        assert result == []

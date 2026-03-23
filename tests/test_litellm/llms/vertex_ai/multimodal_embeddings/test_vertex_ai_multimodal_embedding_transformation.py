import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.multimodal_embeddings.transformation import (
    VertexAIMultimodalEmbeddingConfig,
)
from litellm.types.llms.vertex_ai import Instance, InstanceImage, InstanceVideo


class TestVertexMultimodalEmbedding:
    def setup_method(self):
        self.config = VertexAIMultimodalEmbeddingConfig()

    def test_process_openai_embedding_input(self):
        input_data = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=",
        ]
        expected_output = [
            Instance(
                image=InstanceImage(bytesBase64Encoded=input_data[0].split(",")[1])
            ),
            Instance(
                image=InstanceImage(bytesBase64Encoded=input_data[1].split(",")[1])
            ),
        ]
        assert self.config._process_input_element(input_data[0]) == expected_output[0]
        assert self.config._process_input_element(input_data[1]) == expected_output[1]

    def test_process_str_and_video_input(self):
        input_data = ["hi", "gs://my-bucket/embeddings/supermarket-video.mp4"]
        expected_output = [
            Instance(
                text="hi",
                video=InstanceVideo(
                    gcsUri="gs://my-bucket/embeddings/supermarket-video.mp4"
                ),
            ),
        ]
        assert self.config.process_openai_embedding_input(input_data) == expected_output

    def test_process_list_of_str_and_str_input(self):
        input_data = ["hi", "hello"]
        expected_output = [
            Instance(text="hi"),
            Instance(text="hello"),
        ]
        assert self.config.process_openai_embedding_input(input_data) == expected_output

    def test_process_list_of_str_and_video_input(self):
        input_data = [
            "hi",
            "hello",
            "gs://my-bucket/embeddings/supermarket-video.mp4",
            "hey",
        ]
        expected_output = [
            Instance(text="hi"),
            Instance(
                text="hello",
                video=InstanceVideo(
                    gcsUri="gs://my-bucket/embeddings/supermarket-video.mp4"
                ),
            ),
            Instance(text="hey"),
        ]
        assert (
            self.config.process_openai_embedding_input(input_data) == expected_output
        ), f"Expected {expected_output}, but got {self.config.process_openai_embedding_input(input_data)}"

    def test_process_text_and_base64_image_input(self):
        """Test that text + base64 image combinations are correctly merged into a single instance."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = ["describe this image", base64_image]
        expected_output = [
            Instance(
                text="describe this image",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"

    def test_process_multiple_text_and_base64_image_pairs(self):
        """Test multiple text + base64 image pairs in a single request."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = [
            "first description",
            base64_image,
            "second description",
            base64_image,
        ]
        expected_output = [
            Instance(
                text="first description",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
            Instance(
                text="second description",
                image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1]),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"

    def test_process_base64_image_only_in_list(self):
        """Test that standalone base64 images in a list are processed correctly."""
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        input_data = [base64_image, base64_image]
        expected_output = [
            Instance(image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1])),
            Instance(image=InstanceImage(bytesBase64Encoded=base64_image.split(",")[1])),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"

    def test_process_text_and_gcs_image_input(self):
        """Test that text + GCS image combinations are correctly merged."""
        gcs_uri = "gs://my-bucket/image.png"
        input_data = ["describe this image", gcs_uri]
        expected_output = [
            Instance(
                text="describe this image",
                image=InstanceImage(gcsUri=gcs_uri),
            ),
        ]
        result = self.config.process_openai_embedding_input(input_data)
        assert result == expected_output, f"Expected {expected_output}, but got {result}"

    def test_dimensions_parameter_forwarded_as_dimension(self):
        """Test that the dimensions parameter is forwarded as dimension
        in the request body parameters field.

        Fixes: https://github.com/BerriAI/litellm/issues/24392
        """
        optional_params = {}
        optional_params = self.config.map_openai_params(
            non_default_params={"dimensions": 128},
            optional_params=optional_params,
            model="multimodalembedding",
            drop_params=False,
        )
        request = self.config.transform_embedding_request(
            model="multimodalembedding",
            input="Hello world",
            optional_params=optional_params,
            headers={},
        )
        assert "parameters" in request, "Request should contain 'parameters' field"
        assert request["parameters"]["dimension"] == 128

    def test_no_parameters_field_when_dimensions_not_set(self):
        """Test that no parameters field is added when dimensions is not set."""
        request = self.config.transform_embedding_request(
            model="multimodalembedding",
            input="Hello world",
            optional_params={},
            headers={},
        )
        assert "parameters" not in request

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

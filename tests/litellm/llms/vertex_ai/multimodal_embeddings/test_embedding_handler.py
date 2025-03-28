import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.multimodal_embeddings.embedding_handler import (
    VertexMultimodalEmbedding,
)
from litellm.types.llms.vertex_ai import Instance, InstanceImage


class TestVertexMultimodalEmbedding:
    def setup_method(self):
        self.embedding_handler = VertexMultimodalEmbedding()

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
        assert (
            self.embedding_handler._process_input_element(input_data[0])
            == expected_output[0]
        )
        assert (
            self.embedding_handler._process_input_element(input_data[1])
            == expected_output[1]
        )

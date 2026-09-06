import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.vertex_ai.vertex_embeddings.transformation import (
    VertexAITextEmbeddingConfig,
)
from litellm.utils import get_optional_params_embeddings


def test_encoding_format_is_supported_and_not_forwarded():
    config = VertexAITextEmbeddingConfig()

    assert "encoding_format" in config.get_supported_openai_params()

    optional_params, _ = config.map_openai_params(
        non_default_params={"encoding_format": "float"},
        optional_params={},
        kwargs={},
    )

    assert "encoding_format" not in optional_params


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
def test_get_optional_params_embeddings_encoding_format_does_not_raise(monkeypatch, encoding_format):
    monkeypatch.setattr(litellm, "drop_params", False)

    optional_params = get_optional_params_embeddings(
        model="textembedding-gecko@003",
        encoding_format=encoding_format,
        custom_llm_provider="vertex_ai",
        drop_params=False,
    )

    assert "encoding_format" not in optional_params

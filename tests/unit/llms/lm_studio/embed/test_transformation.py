import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.lm_studio.embed.transformation import LmStudioEmbeddingConfig
from litellm.utils import get_optional_params_embeddings

MODEL = "text-embedding-nomic-embed-text-v1.5"


def test_default_construction_contributes_no_request_params():
    assert LmStudioEmbeddingConfig().get_config() == {}


def test_map_openai_params_returns_the_given_optional_params_and_ignores_openai_params():
    optional_params = {"extra_body": {"keep": True}}

    mapped = LmStudioEmbeddingConfig().map_openai_params({"dimensions": 256}, optional_params)

    assert mapped is optional_params
    assert mapped == {"extra_body": {"keep": True}}


def test_openai_embedding_params_are_rejected_for_lm_studio():
    with pytest.raises(UnsupportedParamsError, match="lm_studio does not support parameters"):
        get_optional_params_embeddings(model=MODEL, custom_llm_provider="lm_studio", dimensions=256)


def test_openai_embedding_params_are_dropped_for_lm_studio_when_drop_params_is_set():
    optional_params = get_optional_params_embeddings(
        model=MODEL, custom_llm_provider="lm_studio", dimensions=256, drop_params=True
    )

    assert optional_params == {}

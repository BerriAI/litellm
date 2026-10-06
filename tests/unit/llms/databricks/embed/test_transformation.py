import re
from typing import Final

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.databricks.embed.transformation import DatabricksEmbeddingConfig
from litellm.utils import get_optional_params_embeddings

EMBEDDING_MODEL: Final = "databricks-bge-large-en"
INSTRUCTION: Final = "Represent this sentence for searching relevant passages:"


def test_map_openai_params_returns_the_given_optional_params_without_adding_openai_params() -> None:
    optional_params = {"instruction": INSTRUCTION}

    mapped = DatabricksEmbeddingConfig().map_openai_params(
        non_default_params={"dimensions": 3}, optional_params=optional_params
    )

    assert mapped is optional_params
    assert mapped == {"instruction": INSTRUCTION}


def test_databricks_embeddings_reject_an_openai_param_the_endpoint_does_not_take() -> None:
    with pytest.raises(
        UnsupportedParamsError, match=re.escape("databricks does not support parameters: {'dimensions': 3}")
    ):
        get_optional_params_embeddings(model=EMBEDDING_MODEL, custom_llm_provider="databricks", dimensions=3)


def test_databricks_embeddings_drop_the_openai_param_and_keep_the_instruction_when_dropping_is_on() -> None:
    optional_params = get_optional_params_embeddings(
        model=EMBEDDING_MODEL,
        custom_llm_provider="databricks",
        dimensions=3,
        drop_params=True,
        instruction=INSTRUCTION,
    )

    assert optional_params == {"instruction": INSTRUCTION}

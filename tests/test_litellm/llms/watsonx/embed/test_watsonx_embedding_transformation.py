import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.llms.watsonx.embed.transformation import IBMWatsonXEmbeddingConfig


class TestIBMWatsonXEmbeddingConfig:
    def _config(self) -> IBMWatsonXEmbeddingConfig:
        return IBMWatsonXEmbeddingConfig()

    def test_transform_embedding_request_wraps_string_input(self):
        cfg = self._config()
        request = cfg.transform_embedding_request(
            model="intfloat/multilingual-e5-large",
            input="cdsc",
            optional_params={"project_id": "test-project-id"},
            headers={},
        )

        assert request["inputs"] == ["cdsc"]
        assert request["project_id"] == "test-project-id"

    def test_transform_embedding_request_preserves_list_input(self):
        cfg = self._config()
        request = cfg.transform_embedding_request(
            model="intfloat/multilingual-e5-large",
            input=["first", "second"],
            optional_params={"project_id": "test-project-id"},
            headers={},
        )

        assert request["inputs"] == ["first", "second"]

    def test_transform_embedding_request_rejects_token_input(self):
        cfg = self._config()

        with pytest.raises(ValueError, match="string or list of strings"):
            cfg.transform_embedding_request(
                model="intfloat/multilingual-e5-large",
                input=[[1, 2, 3]],
                optional_params={"project_id": "test-project-id"},
                headers={},
            )

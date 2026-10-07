from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError


from litellm.llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

JINA_KEY_ENV_NAMES = ("JINA_AI_API_KEY", "JINA_API_KEY", "JINA_AI_TOKEN")


@pytest.fixture
def no_jina_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in JINA_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class TestJinaAIEmbeddingTransform:
    def setup_method(self):
        self.config = JinaAIEmbeddingConfig()
        self.model = "jina-embeddings-v2-base-en"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self):
        """Test that 'dimensions' parameter is correctly mapped"""
        test_params = {"dimensions": 1024}
        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"dimensions": 1024}

    def test_transform_embedding_request_text_input(self):
        """Test transformation of a standard text embedding request"""
        input_data = ["hello world", "hello world again"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_result = {
            "model": self.model,
            "input": input_data,
        }
        assert result == expected_result

    def test_transform_embedding_request_image_input(self):
        """Test transformation of an image embedding request"""
        # a fake base64 string for testing purposes
        input_data = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        ]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_input = [
            {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="}
        ]
        expected_result = {
            "model": self.model,
            "input": expected_input,
        }
        assert result == expected_result

    @pytest.mark.parametrize("env_name", JINA_KEY_ENV_NAMES)
    def test_every_accepted_env_name_resolves_a_key(
        self,
        env_name: str,
        no_jina_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Each accepted spelling must be reachable on its own, not shadowed by a repeated read of another name."""
        sentinel = f"resolved-via-{env_name.lower()}"
        monkeypatch.setenv(env_name, sentinel)

        _, _, dynamic_api_key = self.config._get_openai_compatible_provider_info(api_base=None, api_key=None)

        assert dynamic_api_key == sentinel

    def test_env_name_precedence_is_stable(
        self,
        no_jina_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Earlier names in the chain win, so adding later fallbacks never re-points an already working install."""
        for name in JINA_KEY_ENV_NAMES:
            monkeypatch.setenv(name, f"resolved-via-{name.lower()}")

        for expected_name in JINA_KEY_ENV_NAMES:
            _, _, dynamic_api_key = self.config._get_openai_compatible_provider_info(api_base=None, api_key=None)
            assert dynamic_api_key == f"resolved-via-{expected_name.lower()}"
            monkeypatch.delenv(expected_name)

    def test_explicit_api_key_beats_every_env_name(
        self,
        no_jina_env: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """An api_key passed by the caller short-circuits the whole environment chain."""
        for name in JINA_KEY_ENV_NAMES:
            monkeypatch.setenv(name, f"resolved-via-{name.lower()}")

        _, _, dynamic_api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key="passed-in-by-caller"
        )

        assert dynamic_api_key == "passed-in-by-caller"

    def test_transform_embedding_request_mixed_input(self):
        """Test transformation of a mixed text and image embedding request"""
        # a fake base64 string for testing purposes
        base64_str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        input_data = ["hello world", base64_str]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_input = [
            {"text": "hello world"},
            {"image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="},
        ]
        expected_result = {
            "model": self.model,
            "input": expected_input,
        }
        assert result == expected_result


def _transform(raw_response: httpx.Response) -> EmbeddingResponse:
    return JinaAIEmbeddingConfig().transform_embedding_response(
        model="jina-embeddings-v3",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=MagicMock(),
        api_key="test-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )


def test_transform_embedding_response_builds_the_response_from_the_body():
    response = _transform(
        httpx.Response(
            200,
            json={
                "model": "jina-embeddings-v3",
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 2]}],
                "usage": {"prompt_tokens": 3, "total_tokens": 5},
                "unknown": "ignored",
            },
        )
    )

    assert response.model_dump() == {
        "model": "jina-embeddings-v3",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 2]}],
        "object": "list",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 3,
            "total_tokens": 5,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }


@pytest.mark.parametrize("body", [b"null", b"7", b'["leaked payload text"]', b'"leaked payload text"'])
def test_transform_embedding_response_rejects_a_body_that_is_not_an_object(body: bytes):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, content=body))

    assert "leaked payload text" not in str(exc_info.value)


@pytest.mark.parametrize("field", ["model", "data", "usage"])
def test_transform_embedding_response_invalid_field_is_reported_by_name(field: str):
    with pytest.raises(ValidationError) as exc_info:
        _transform(httpx.Response(200, json={field: 5}))

    assert exc_info.value.title == "EmbeddingResponse"
    assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]

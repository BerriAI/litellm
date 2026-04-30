from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig
from litellm.llms.huggingface.common_utils import HuggingFaceError
from litellm.llms.huggingface.embedding.handler import HuggingFaceEmbedding
from litellm.llms.huggingface.embedding.transformation import (
    HuggingFaceEmbeddingConfig,
)


def test_huggingface_chat_rejects_url_valued_model():
    config = HuggingFaceChatConfig()

    with pytest.raises(HuggingFaceError) as exc_info:
        config.get_complete_url(
            api_base=None,
            api_key="hf-secret",
            model="https://attacker.example/v1",
            optional_params={},
            litellm_params={},
        )

    assert exc_info.value.status_code == 400


def test_huggingface_chat_allows_legacy_url_model_when_rejection_disabled(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "reject_url_model_destinations", False)
    config = HuggingFaceChatConfig()

    complete_url = config.get_complete_url(
        api_base=None,
        api_key="hf-secret",
        model="https://trusted.example",
        optional_params={},
        litellm_params={},
    )

    assert complete_url == "https://trusted.example/v1/chat/completions"


def test_huggingface_chat_keeps_explicit_api_base_for_custom_endpoints():
    config = HuggingFaceChatConfig()

    complete_url = config.get_complete_url(
        api_base="https://admin-configured.example",
        api_key="hf-secret",
        model="huggingface/mistral",
        optional_params={},
        litellm_params={},
    )

    assert complete_url == "https://admin-configured.example/v1/chat/completions"


def test_huggingface_embedding_config_rejects_url_valued_model():
    config = HuggingFaceEmbeddingConfig()

    with pytest.raises(HuggingFaceError) as exc_info:
        config.get_api_base(
            api_base=None,
            model="prefixhttps://attacker.example/embeddings",
        )

    assert exc_info.value.status_code == 400


def test_huggingface_embedding_config_allows_legacy_url_model_when_rejection_disabled(
    monkeypatch,
):
    monkeypatch.setattr(litellm, "reject_url_model_destinations", False)
    config = HuggingFaceEmbeddingConfig()

    api_base = config.get_api_base(
        api_base=None,
        model="https://trusted.example/embeddings",
    )

    assert api_base == "https://trusted.example/embeddings"


def test_huggingface_embedding_handler_rejects_before_task_lookup():
    handler = HuggingFaceEmbedding()
    logging_obj = MagicMock()
    encoding = MagicMock()
    encoding.encode.return_value = []

    with patch(
        "litellm.llms.huggingface.embedding.handler.get_hf_task_embedding_for_model"
    ) as mock_task_lookup:
        with pytest.raises(HuggingFaceError) as exc_info:
            handler.embedding(
                model="https://attacker.example/embeddings",
                input=["hello"],
                model_response=MagicMock(),
                optional_params={},
                litellm_params={},
                logging_obj=logging_obj,
                encoding=encoding,
                api_key="hf-secret",
            )

    assert exc_info.value.status_code == 400
    mock_task_lookup.assert_not_called()

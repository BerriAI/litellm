from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.oobabooga.chat.oobabooga import completion, embedding
from litellm.llms.oobabooga.common_utils import OobaboogaError


def test_oobabooga_completion_rejects_url_valued_model_before_request():
    with patch("litellm.llms.oobabooga.chat.oobabooga._get_httpx_client") as mock_get:
        with pytest.raises(OobaboogaError) as exc_info:
            completion(
                model="https://attacker.example/v1",
                messages=[],
                api_base="https://admin-configured.example",
                model_response=MagicMock(),
                print_verbose=MagicMock(),
                encoding=MagicMock(),
                api_key="ooba-secret",
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    assert exc_info.value.status_code == 400
    mock_get.assert_not_called()


def test_oobabooga_completion_allows_legacy_url_model_when_host_is_allowlisted(
    monkeypatch,
):
    monkeypatch.setattr(
        litellm, "provider_url_destination_allowed_hosts", ["trusted.example"]
    )
    response = MagicMock()
    client = MagicMock()
    client.post.return_value = response

    with patch(
        "litellm.llms.oobabooga.chat.oobabooga._get_httpx_client",
        return_value=client,
    ):
        with patch(
            "litellm.llms.oobabooga.chat.oobabooga.oobabooga_config.transform_response",
            return_value="ok",
        ):
            result = completion(
                model="https://trusted.example",
                messages=[],
                api_base=None,
                model_response=MagicMock(),
                print_verbose=MagicMock(),
                encoding=MagicMock(),
                api_key="ooba-secret",
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    assert result == "ok"
    client.post.assert_called_once()
    assert (
        client.post.call_args.args[0] == "https://trusted.example/v1/chat/completions"
    )


def test_oobabooga_completion_rejects_url_model_when_host_is_not_allowlisted(
    monkeypatch,
):
    monkeypatch.setattr(
        litellm, "provider_url_destination_allowed_hosts", ["trusted.example"]
    )

    with patch("litellm.llms.oobabooga.chat.oobabooga._get_httpx_client") as mock_get:
        with pytest.raises(OobaboogaError) as exc_info:
            completion(
                model="https://other.example",
                messages=[],
                api_base=None,
                model_response=MagicMock(),
                print_verbose=MagicMock(),
                encoding=MagicMock(),
                api_key="ooba-secret",
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    assert exc_info.value.status_code == 400
    mock_get.assert_not_called()


def test_oobabooga_embedding_rejects_url_valued_model_before_request():
    with patch(
        "litellm.llms.oobabooga.chat.oobabooga.litellm.module_level_client.post"
    ) as mock_post:
        with pytest.raises(OobaboogaError) as exc_info:
            embedding(
                model="prefixhttps://attacker.example/embeddings",
                input=["hello"],
                model_response=MagicMock(),
                api_key="ooba-secret",
                api_base="https://admin-configured.example",
                logging_obj=MagicMock(),
                optional_params={},
                encoding=MagicMock(),
            )

    assert exc_info.value.status_code == 400
    mock_post.assert_not_called()

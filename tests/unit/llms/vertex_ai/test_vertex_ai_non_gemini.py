import logging

import pytest

import litellm

FILE_FAILURE = "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason."


def _complete_on_the_legacy_route(vertex_credentials: str) -> None:
    litellm.completion(
        model="vertex_ai/chat-bison",
        messages=[{"role": "user", "content": "hi"}],
        vertex_credentials=vertex_credentials,
        vertex_project="p",
        vertex_location="us-central1",
    )


def test_a_missing_key_file_is_reported_as_a_file_failure(tmp_path, caplog):
    missing = tmp_path / "vertexai.json"

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        with pytest.raises(litellm.APIConnectionError, match=FILE_FAILURE):
            _complete_on_the_legacy_route(str(missing))

    assert repr(str(missing)) in caplog.text
    assert "No such file or directory" in caplog.text


def test_a_key_file_is_read_and_handed_to_google_auth(tmp_path, caplog):
    not_a_key = tmp_path / "settings.json"
    not_a_key.write_text('{"hello": "world"}')

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        with pytest.raises(litellm.APIConnectionError, match=FILE_FAILURE):
            _complete_on_the_legacy_route(str(not_a_key))

    assert f"the credentials file at {str(not_a_key)!r} holds credentials that cannot be used" in caplog.text

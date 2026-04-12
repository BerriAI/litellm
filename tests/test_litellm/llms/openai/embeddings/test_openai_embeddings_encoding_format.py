from unittest.mock import patch

import litellm


@patch("litellm.main.openai_chat_completions.embedding", return_value={"ok": True})
def test_openai_embedding_does_not_send_encoding_format_when_unset(mock_embedding):
    """Regression test: do not send encoding_format=null to OpenAI-compatible APIs."""
    litellm.embedding(
        model="text-embedding-3-small",
        input=["hello"],
        api_base="https://example.com/v1",
        api_key="test-key",
        custom_llm_provider="openai",
    )

    optional_params = mock_embedding.call_args.kwargs["optional_params"]
    assert "encoding_format" not in optional_params


@patch("litellm.main.openai_chat_completions.embedding", return_value={"ok": True})
def test_openai_embedding_preserves_explicit_encoding_format(mock_embedding):
    """Explicit encoding_format should still be forwarded."""
    litellm.embedding(
        model="text-embedding-3-small",
        input=["hello"],
        api_base="https://example.com/v1",
        api_key="test-key",
        custom_llm_provider="openai",
        encoding_format="float",
    )

    optional_params = mock_embedding.call_args.kwargs["optional_params"]
    assert optional_params["encoding_format"] == "float"

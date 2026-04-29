from unittest.mock import MagicMock
from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _convert_mcp_content_to_openai,
    _convert_single_content,
    _resolve_model_from_preferences,
)


def test_convert_text_content():
    mock_text = MagicMock()
    mock_text.type = "text"
    mock_text.text = "hello world"

    result = _convert_single_content(mock_text)
    assert result == {"type": "text", "text": "hello world"}


def test_convert_image_content():
    mock_image = MagicMock()
    mock_image.type = "image"
    mock_image.data = "base64data"
    mock_image.mimeType = "image/jpeg"

    result = _convert_single_content(mock_image)
    assert result == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64data"}}


def test_convert_audio_content():
    mock_audio = MagicMock()
    mock_audio.type = "audio"
    mock_audio.data = "audiobase64"
    mock_audio.mimeType = "audio/mp3"

    result = _convert_single_content(mock_audio)
    assert result == {"type": "input_audio", "input_audio": {"data": "audiobase64", "format": "mp3"}}


def test_convert_list_content():
    mock_text = MagicMock()
    mock_text.type = "text"
    mock_text.text = "text"

    mock_image = MagicMock()
    mock_image.type = "image"
    mock_image.data = "img"
    mock_image.mimeType = "image/png"

    result = _convert_mcp_content_to_openai([mock_text, mock_image])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "text"}
    assert result[1]["type"] == "image_url"


def test_resolve_model_from_hints():
    import litellm.proxy.proxy_server as proxy_server

    mock_prefs = MagicMock()
    mock_hint = MagicMock()
    mock_hint.name = "claude"
    mock_prefs.hints = [mock_hint]

    # Save original
    original_router = proxy_server.llm_router
    try:
        proxy_server.llm_router = MagicMock()
        proxy_server.llm_router.get_model_names.return_value = ["gpt-4", "claude-3-5-sonnet"]
        result = _resolve_model_from_preferences(mock_prefs)
        assert result == "claude-3-5-sonnet"
    finally:
        proxy_server.llm_router = original_router


def test_resolve_model_fallback():
    result = _resolve_model_from_preferences(None, default_model="fallback-model")
    assert result == "fallback-model"

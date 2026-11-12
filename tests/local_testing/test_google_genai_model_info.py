from unittest.mock import MagicMock
from litellm.google_genai.main import GenerateContentHelper


def test_setup_generate_content_call_passes_model_info_to_logging():
    mock_logging_obj = MagicMock()
    expected_model_info = {"id": "deployment-123", "db_model": True}

    GenerateContentHelper.setup_generate_content_call(
        model="gemini/gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        litellm_logging_obj=mock_logging_obj,
        litellm_call_id="call-123",
        model_info=expected_model_info,
    )

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args.kwargs
    assert call_kwargs["litellm_params"]["model_info"] == expected_model_info


def test_setup_generate_content_call_omits_model_info_when_not_provided():
    mock_logging_obj = MagicMock()

    GenerateContentHelper.setup_generate_content_call(
        model="gemini/gemini-2.0-flash",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        litellm_logging_obj=mock_logging_obj,
        litellm_call_id="call-456",
    )

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args.kwargs
    assert "model_info" not in call_kwargs["litellm_params"]

from unittest.mock import patch

from litellm.llms.google_code_assist.transformation import GoogleCodeAssistConfig


def test_transform_request_uses_base_model_name_for_internal_gemini_helpers():
    class _CaptureConfig(GoogleCodeAssistConfig):
        def __init__(self):
            super().__init__()
            self.model_seen_by_map = None

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            messages: list,
        ) -> dict:
            self.model_seen_by_map = model
            return {"thinkingConfig": {"includeThoughts": True}}

    config = _CaptureConfig()
    with (
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._transform_system_message",
            return_value=(None, [{"role": "user", "content": "hi"}]),
        ),
        patch(
            "litellm.llms.vertex_ai.gemini.transformation._gemini_convert_messages_with_history",
            return_value=[{"role": "user", "parts": [{"text": "hi"}]}],
        ) as mock_convert,
    ):
        result = config.transform_request(
            model="google_code_assist/gemini-2.5-pro",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"thinkingConfig": {"includeThoughts": False}},
            litellm_params={},
        )

    assert config.model_seen_by_map == "gemini-2.5-pro"
    assert mock_convert.call_args.kwargs["model"] == "gemini-2.5-pro"
    assert result["model"] == "gemini-2.5-pro"

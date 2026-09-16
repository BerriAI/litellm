from litellm.llms.sap.chat.gemini_direct_transformation import SapDeploymentGeminiChatConfig
from litellm.llms.sap.chat.handler import GenAIHubOrchestration
from litellm.llms.sap.chat.openai_direct_transformation import SapDeploymentOpenAIChatConfig
from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig

_HANDLER = GenAIHubOrchestration()


def test_orchestration_body_enables_stream_in_config_envelope():
    data = {"config": {"modules": {"foo": "bar"}}}
    result = _HANDLER._add_stream_param_to_request_body(data, GenAIHubOrchestrationConfig(), fake_stream=False)
    assert result["config"]["stream"] == {"enabled": True}
    assert result["config"]["modules"] == {"foo": "bar"}


def test_orchestration_body_merges_into_existing_stream_block():
    data = {"config": {"stream": {"chunk_size": 10}}}
    result = _HANDLER._add_stream_param_to_request_body(data, GenAIHubOrchestrationConfig(), fake_stream=False)
    assert result["config"]["stream"] == {"chunk_size": 10, "enabled": True}


def test_direct_connect_body_without_stream_support_is_left_untouched():
    data = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {}}
    result = _HANDLER._add_stream_param_to_request_body(data, SapDeploymentGeminiChatConfig(), fake_stream=False)
    assert result == data
    assert "stream" not in result


def test_direct_connect_body_with_stream_support_gets_top_level_stream():
    data = {"messages": [{"role": "user", "content": "hi"}]}
    result = _HANDLER._add_stream_param_to_request_body(data, SapDeploymentOpenAIChatConfig(), fake_stream=False)
    assert result["stream"] is True
    assert "config" not in result

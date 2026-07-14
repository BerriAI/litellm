"""Regression test for https://github.com/BerriAI/litellm/issues/33221"""
from litellm.main import responses_api_bridge_check


def test_gpt56_tools_bridged_to_responses_without_reasoning_effort():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
    for model in ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6"]:
        model_info, _ = responses_api_bridge_check(model=model, custom_llm_provider="openai", tools=tools, reasoning_effort=None)
        assert model_info.get("mode") == "responses", f"{model} with tools should bridge to responses even without reasoning_effort"

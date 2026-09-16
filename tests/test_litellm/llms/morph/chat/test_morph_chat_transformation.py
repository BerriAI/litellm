import pytest

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.morph.chat.transformation import MorphChatConfig

NO_TOOLS_MODEL = "morph-v3-large"
TOOL_CALLING_MODEL = "morph-glm53flash"
UNMAPPED_MODEL = "morph-unreleased-model"
TOOLS = [
    {
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }
]


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.mark.parametrize("model", [NO_TOOLS_MODEL, f"morph/{NO_TOOLS_MODEL}", "morph-v3-fast"])
def test_tool_params_are_withheld_when_registry_disables_function_calling(model):
    supported_params = MorphChatConfig().get_supported_openai_params(model)

    assert "tools" not in supported_params
    assert "tool_choice" not in supported_params
    assert "response_format" in supported_params
    assert "temperature" in supported_params


@pytest.mark.parametrize("model", [TOOL_CALLING_MODEL, f"morph/{TOOL_CALLING_MODEL}", UNMAPPED_MODEL])
def test_tool_params_pass_through_unless_registry_disables_function_calling(model):
    supported_params = MorphChatConfig().get_supported_openai_params(model)

    assert "tools" in supported_params
    assert "tool_choice" in supported_params


def test_tools_for_model_without_function_calling_raise_unsupported_params():
    with pytest.raises(UnsupportedParamsError):
        litellm.get_optional_params(
            model=NO_TOOLS_MODEL,
            custom_llm_provider="morph",
            tools=TOOLS,
            drop_params=False,
        )


def test_tools_for_model_without_function_calling_are_dropped_with_drop_params():
    optional_params = litellm.get_optional_params(
        model=NO_TOOLS_MODEL,
        custom_llm_provider="morph",
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
        drop_params=True,
    )

    assert "tools" not in optional_params
    assert "tool_choice" not in optional_params
    assert optional_params["temperature"] == 0.2


def test_tools_for_tool_calling_model_are_forwarded():
    optional_params = litellm.get_optional_params(
        model=TOOL_CALLING_MODEL,
        custom_llm_provider="morph",
        tools=TOOLS,
        tool_choice="required",
    )

    assert optional_params["tools"] == TOOLS
    assert optional_params["tool_choice"] == "required"

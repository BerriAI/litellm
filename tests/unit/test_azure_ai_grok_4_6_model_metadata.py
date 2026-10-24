from typing import Final

import pytest

from litellm import cost_per_token, get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MODEL: Final = "azure_ai/grok-4.6"


@pytest.mark.usefixtures("local_model_cost_map")
def test_azure_ai_grok_4_6_is_priced_and_routed() -> None:
    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert (routed_model, provider) == ("grok-4.6", "azure_ai")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "chat"
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    prompt_cost, completion_cost = cost_per_token(model=MODEL, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert prompt_cost > 0
    assert completion_cost > 0

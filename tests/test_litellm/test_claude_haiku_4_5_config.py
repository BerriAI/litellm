"""
Test Claude Haiku 4.5 model configurations for Bedrock
https://github.com/BerriAI/litellm/issues/15818
"""

import json
import os




def test_bedrock_haiku_4_5_matches_sonnet_capabilities():
    """
    Test that Haiku 4.5 has same capabilities as Sonnet 4.5
    (including computer_use, vision, tools, etc.)
    """
    # Load model configuration
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        model_data = json.load(f)

    haiku_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    sonnet_model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    haiku_info = model_data[haiku_model]
    sonnet_info = model_data[sonnet_model]

    # Both should use bedrock_converse
    assert haiku_info["litellm_provider"] == "bedrock_converse"
    assert sonnet_info["litellm_provider"] == "bedrock_converse"

    # Shared capabilities that should match
    shared_capabilities = [
        "supports_vision",
        "supports_computer_use",
        "supports_function_calling",
        "supports_tool_choice",
        "supports_prompt_caching",
        "supports_response_schema",
        "supports_pdf_input",
        "supports_assistant_prefill",
        "supports_reasoning",
    ]

    for capability in shared_capabilities:
        assert haiku_info.get(capability) == sonnet_info.get(
            capability
        ), f"Capability {capability} mismatch: Haiku={haiku_info.get(capability)}, Sonnet={sonnet_info.get(capability)}"

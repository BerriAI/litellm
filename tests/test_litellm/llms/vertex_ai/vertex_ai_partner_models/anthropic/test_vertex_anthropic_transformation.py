import pytest
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig

def test_vertex_anthropic_unsupported_params_removal():
    """
    Test that unsupported params are removed for Claude on Vertex AI 
    to fix Issue #21407.
    """
    config = VertexAIAnthropicConfig()
    
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "output_config": {"effort": "high"},
        "output_format": {"type": "json_schema"}
    }
    
    transformed_data = config.transform_request(
        model="claude-3-sonnet",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    # Assertions: Verify they are GONE, not remapped
    assert "output_config" not in transformed_data
    assert "output_format" not in transformed_data
    assert "response_schema" not in transformed_data
import pytest
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig

def test_vertex_anthropic_output_config_transformation():
    """
    Test that output_config is correctly remapped to response_schema 
    for Claude on Vertex AI to fix Issue #21407.
    """
    config = VertexAIAnthropicConfig()
    
    # 1. Setup mock Anthropic-style request data
    messages = [{"role": "user", "content": "Generate a JSON response"}]
    optional_params = {
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}}
                }
            }
        }
    }
    
    # 2. Run transformation
    transformed_data = config.transform_request(
        model="claude-3-sonnet",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    # 3. Assertions
    assert "output_config" not in transformed_data, "output_config must be removed"
    assert "response_schema" in transformed_data, "response_schema must be present"
    assert transformed_data["response_schema"]["type"] == "object"
    assert "name" in transformed_data["response_schema"]["properties"]
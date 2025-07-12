import litellm
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body


def test_system_message_conversion_gemini():
    """Test that system-only messages are properly handled for Gemini"""
    # Case 1: Default behavior - duplicate system as user
    messages = [{"role": "system", "content": "You are a helpful assistant"}]
    
    # Create mock objects for the test
    model = "gemini-2.0-flash"
    custom_llm_provider = "gemini"
    optional_params = {}
    litellm_params = {}
    
    result = _transform_request_body(
        messages=messages, # type: ignore
        model=model,
        optional_params=optional_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=None
    )
    
    # Verify that contents has user message
    assert len(result["contents"]) > 0
    assert result["contents"][0]["role"] == "user" # type: ignore
    assert "system_instruction" in result
    
    # Case 2: Disable duplication
    optional_params = {"duplicate_system_as_user_for_gemini": False}
    
    # Save original modify_params value
    original_modify_params = litellm.modify_params
    litellm.modify_params = False
    
    result_no_duplicate = _transform_request_body(
        messages=messages.copy(), # type: ignore
        model=model,
        optional_params=optional_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params={},
        cached_content=None
    )
    
    # Restore original modify_params value
    litellm.modify_params = original_modify_params
    
    # With duplication disabled and modify_params False,
    # we'd expect an empty contents field
    # This might actually raise an exception in practice
    assert "system_instruction" in result_no_duplicate
    
    # Case 3: With litellm.modify_params=True it should duplicate even with parameter set to False
    litellm.modify_params = True
    
    result_with_modify_params = _transform_request_body(
        messages=messages.copy(), # type: ignore
        model=model,
        optional_params={"duplicate_system_as_user_for_gemini": False},
        custom_llm_provider=custom_llm_provider,
        litellm_params={},
        cached_content=None
    )
    
    # Restore original modify_params value
    litellm.modify_params = original_modify_params
    
    # Verify that contents has user message due to modify_params=True
    assert len(result_with_modify_params["contents"]) > 0
    assert result_with_modify_params["contents"][0]["role"] == "user" # type: ignore

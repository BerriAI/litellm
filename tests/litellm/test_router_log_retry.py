import pytest
from unittest.mock import Mock
from litellm import Router

def test_log_retry_cross_request_leakage():
    """
    Test that log_retry does not leak previous_models across different requests
    that happen to use the same router instance.
    """
    router = Router(model_list=[{"model_name": "test", "litellm_params": {"model": "gpt-3.5-turbo"}}])
    
    kwargs_1 = {
        "litellm_call_id": "call_1",
        "model": "test",
        "api_key": "USER_1_SECRET_KEY_123",
        "litellm_metadata": {"user": "user1"}
    }
    e1 = Exception("HTTP 429 RateLimit")
    out_1 = router.log_retry(kwargs_1, e1)
    
    kwargs_2 = {
        "litellm_call_id": "call_2",
        "model": "test",
        "api_key": "USER_2_SECRET_KEY_456",
        "litellm_metadata": {"user": "user2"}
    }
    e2 = Exception("HTTP 500 Internal Error")
    out_2 = router.log_retry(kwargs_2, e2)
    
    prev_models_1 = out_1.get("litellm_metadata", {}).get("previous_models", [])
    prev_models_2 = out_2.get("litellm_metadata", {}).get("previous_models", [])
    
    # Assert each request only has 1 previous_model
    assert len(prev_models_1) == 1
    assert len(prev_models_2) == 1
    
    # Assert leakage is prevented
    assert prev_models_2[0]["litellm_call_id"] == "call_2"
    assert "call_1" not in str(prev_models_2)

def test_log_retry_bounded_growth():
    """
    Test that log_retry caps the previous_models list at 4 entries maximum.
    """
    router = Router(model_list=[{"model_name": "test", "litellm_params": {"model": "gpt-3.5-turbo"}}])
    
    kwargs = {
        "litellm_call_id": "call_bounce",
        "model": "test",
        "litellm_metadata": {}
    }
    
    # Call log_retry 10 times consecutively on the SAME request (kwargs object)
    for i in range(10):
        out = router.log_retry(kwargs, Exception(f"Error {i}"))
        kwargs = out
        
    prev_models = kwargs.get("litellm_metadata", {}).get("previous_models", [])
    
    # It checks `if len > 3: pop(0)`, meaning it pops until it's 3, then appends 1 -> max 4 entries
    assert len(prev_models) == 4
    
    # Ensure it kept the most recent 4
    exception_strings = [pm.get("exception_string") for pm in prev_models]
    assert exception_strings == ["Error 6", "Error 7", "Error 8", "Error 9"]

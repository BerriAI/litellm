"""
Test filter_out_litellm_params helper function.
"""
from litellm.utils import filter_out_litellm_params


def test_filter_out_litellm_params():
    """
    Test that filter_out_litellm_params removes LiteLLM internal parameters 
    while keeping provider-specific parameters.
    """
    kwargs = {
        "query": "test query",
        "max_results": 10,
        "shared_session": "mock_session_object",
        "metadata": {"key": "value"},
        "litellm_trace_id": "trace-123",
        "proxy_server_request": {"url": "http://example.com"},
        "secret_fields": {"api_key": "secret"},
        "custom_param": "should_be_kept",
    }
    
    filtered = filter_out_litellm_params(kwargs=kwargs)
    
    # Provider-specific params are kept
    assert filtered["query"] == "test query"
    assert filtered["max_results"] == 10
    assert filtered["custom_param"] == "should_be_kept"
    
    # LiteLLM internal params are removed
    assert "shared_session" not in filtered
    assert "metadata" not in filtered
    assert "litellm_trace_id" not in filtered
    assert "proxy_server_request" not in filtered
    assert "secret_fields" not in filtered


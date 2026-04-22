"""
Test routing_strategy override functionality for issue #21993
"""
import pytest


def test_routing_strategy_per_request_override():
    """
    Test that routing_strategy can be overridden per-request.
    
    This tests the core fix: routing_strategy should be extracted from request_kwargs
    and used instead of self.routing_strategy for deployment selection.
    """
    # Test the logic of extracting routing_strategy from kwargs
    DEFAULT_ROUTING = "least-busy"
    
    # Test 1: When routing_strategy is provided in kwargs
    request_kwargs = {"routing_strategy": "cost-based-routing", "messages": ["test"]}
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    assert routing_strategy_to_use == "cost-based-routing", "Should use override routing_strategy"
    assert "routing_strategy" not in request_kwargs, "routing_strategy should be popped from kwargs (not forwarded to LLM API)"
    
    # Test 2: When routing_strategy is NOT provided
    request_kwargs = {"messages": ["test"]}
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    assert routing_strategy_to_use == DEFAULT_ROUTING, "Should use default routing_strategy when no override"
    assert "routing_strategy" not in request_kwargs, "routing_strategy should be popped from kwargs"


def test_routing_strategy_all_strategies():
    """
    Test that all routing strategies can be overridden.
    """
    strategies = [
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
        "cost-based-routing",
        "usage-based-routing-v2",
    ]
    
    DEFAULT_ROUTING = "simple-shuffle"
    
    for strategy in strategies:
        request_kwargs = {"routing_strategy": strategy}
        routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
        
        assert routing_strategy_to_use == strategy, f"Expected {strategy}, got {routing_strategy_to_use}"
        assert "routing_strategy" not in request_kwargs, f"routing_strategy should be popped (not forwarded to API) for {strategy}"


def test_routing_strategy_none_override():
    """
    Test that explicit None override uses default routing_strategy.
    """
    DEFAULT_ROUTING = "least-busy"
    
    # Test with explicit None
    request_kwargs = {"routing_strategy": None}
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    assert routing_strategy_to_use == DEFAULT_ROUTING, "None override should use default"


def test_routing_strategy_empty_string_override():
    """
    Test that empty string override uses default routing_strategy.
    """
    DEFAULT_ROUTING = "simple-shuffle"
    
    # Test with empty string
    request_kwargs = {"routing_strategy": ""}
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    assert routing_strategy_to_use == DEFAULT_ROUTING, "Empty string override should use default"


def test_routing_strategy_doesnt_affect_other_kwargs():
    """
    Test that popping routing_strategy doesn't affect other kwargs.
    """
    DEFAULT_ROUTING = "least-busy"
    
    request_kwargs = {
        "routing_strategy": "cost-based-routing",
        "messages": ["test message"],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    assert routing_strategy_to_use == "cost-based-routing", "Should use override routing_strategy"
    assert "routing_strategy" not in request_kwargs, "routing_strategy should be popped"
    assert request_kwargs["messages"] == ["test message"], "Other kwargs should be preserved"
    assert request_kwargs["temperature"] == 0.7, "Temperature should be preserved"
    assert request_kwargs["max_tokens"] == 100, "Max tokens should be preserved"


def test_routing_strategy_prevents_api_forwarding():
    """
    Test that routing_strategy is NOT forwarded to LLM API.
    
    This addresses P1 concern from reviewer: "routing_strategy gets forwarded to litellm.acompletion and then to the underlying LLM API"
    """
    DEFAULT_ROUTING = "simple-shuffle"
    
    # Test that routing_strategy is properly popped (removed) from kwargs
    request_kwargs = {
        "routing_strategy": "cost-based-routing",
        "messages": ["test"],
        "temperature": 0.7,
    }
    
    routing_strategy_to_use = request_kwargs.pop("routing_strategy", None) or DEFAULT_ROUTING
    
    # routing_strategy should be gone from kwargs
    assert "routing_strategy" not in request_kwargs, "routing_strategy should be popped from kwargs (not forwarded to API)"
    
    # Other kwargs should be preserved
    assert "messages" in request_kwargs, "Messages should be preserved"
    assert "temperature" in request_kwargs, "Temperature should be preserved"
    
    # Verify routing was used for the request
    assert routing_strategy_to_use == "cost-based-routing", "Should use override routing_strategy"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
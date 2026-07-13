from litellm.router_utils.search_api_router import SearchAPIRouter

def test_expand_search_tools():
    # Setup test data with multiple keys
    search_tools_config = [
        {
            "search_tool_name": "tavily-pool",
            "litellm_params": {
                "search_provider": "tavily",
                "api_keys": ["key1", "key2"]
            }
        },
        {
            "search_tool_name": "single-tavily",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "key3"
            }
        }
    ]

    # Run the expansion logic
    expanded = SearchAPIRouter._expand_search_tools(search_tools_config)

    # Assertions
    assert len(expanded) == 3, f"Expected 3 tools, got {len(expanded)}"
    
    pool_tools = [t for t in expanded if t["search_tool_name"] == "tavily-pool"]
    assert len(pool_tools) == 2, "Expected 2 tools from the pool"
    
    assert pool_tools[0]["litellm_params"]["api_key"] == "key1"
    assert pool_tools[0]["search_tool_id"] == "tavily-pool_key_0"
    assert "api_keys" not in pool_tools[0]["litellm_params"], "api_keys should be removed from expanded dict"

    assert pool_tools[1]["litellm_params"]["api_key"] == "key2"
    assert pool_tools[1]["search_tool_id"] == "tavily-pool_key_1"

    single_tool = [t for t in expanded if t["search_tool_name"] == "single-tavily"][0]
    assert single_tool["litellm_params"]["api_key"] == "key3"
    assert single_tool["search_tool_id"] == "single-tavily"

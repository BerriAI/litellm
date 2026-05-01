def test_vertex_ai_web_search_options_dropped_when_function_tools_are_present():
    """Test that search tools are dropped even when web_search_options is mapped first."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()

    non_default_params = {
        "web_search_options": {},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "notion_search",
                    "description": "Search Notion",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    }

    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="gemini-2.5-pro",
        drop_params=True,
    )

    assert len(result["tools"]) == 1
    assert "function_declarations" in result["tools"][0]
    assert "googleSearch" not in result["tools"][0]

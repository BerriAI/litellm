from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

def test_vertex_ai_audio_supported_params():
    config = VertexGeminiConfig()
    
    # Test that 'audio' is in supported_params
    assert "audio" in config.supported_params
    
def test_map_function_google_search_retrieval_snake_case():
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "googleSearchRetrieval" in result[0]

def test_map_function_enterprise_web_search_snake_case():
    config = VertexGeminiConfig()
    optional_params = {}

    tools = [{"enterprise_web_search": {}}]
    result = config._map_function(tools, optional_params)

    assert len(result) == 1
    assert "enterpriseWebSearch" in result[0]

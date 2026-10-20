from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

UNPREFIXED = "gemini-3.1-flash-lite-image"
GEMINI = "gemini/gemini-3.1-flash-lite-image"
VERTEX = "vertex_ai/gemini-3.1-flash-lite-image"


def test_gemini_prefix_routes_to_gemini():
    routed_model, provider, _, _ = get_llm_provider(model=GEMINI)
    assert routed_model == UNPREFIXED
    assert provider == "gemini"


def test_vertex_prefix_routes_to_vertex():
    routed_model, provider, _, _ = get_llm_provider(model=VERTEX)
    assert routed_model == UNPREFIXED
    assert provider == "vertex_ai"

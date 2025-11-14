from litellm.llms.vertex_ai.gemini.transformation import (
    _transform_request_body,
)

# Tests for check_if_part_exists_in_parts removed - function not yet in carto/main
# Will be re-added when upstream-sync-resolver/24 is merged


# Tests for issue #14556: Labels field provider-aware filtering
def test_google_genai_excludes_labels():
    """Test that Google GenAI/AI Studio endpoints exclude labels when custom_llm_provider='gemini'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="gemini",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Google GenAI/AI Studio should NOT include labels
    assert "labels" not in result
    assert "contents" in result


def test_vertex_ai_includes_labels():
    """Test that Vertex AI endpoints include labels when custom_llm_provider='vertex_ai'"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {"labels": {"project": "test", "team": "ai"}}
    litellm_params = {}

    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params,
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params,
        cached_content=None,
    )

    # Vertex AI SHOULD include labels
    assert "labels" in result
    assert result["labels"] == {"project": "test", "team": "ai"}



def test_metadata_to_labels_vertex_only():
    """Test that metadata->labels conversion only happens for Vertex AI"""
    messages = [{"role": "user", "content": "test"}]
    optional_params = {}
    litellm_params = {
        "metadata": {
            "requester_metadata": {
                "user": "john_doe",
                "project": "test-project"
            }
        }
    }

    # Google GenAI/AI Studio should not include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="gemini",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" not in result

    # Vertex AI should include labels from metadata
    result = _transform_request_body(
        messages=messages,
        model="gemini-2.5-pro",
        optional_params=optional_params.copy(),
        custom_llm_provider="vertex_ai",
        litellm_params=litellm_params.copy(),
        cached_content=None,
    )
    assert "labels" in result
    assert result["labels"] == {"user": "john_doe", "project": "test-project"}

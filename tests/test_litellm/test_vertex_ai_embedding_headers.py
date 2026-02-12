import pytest
from litellm import embedding

def test_vertex_ai_embedding_extra_headers():
    # Test that extra_headers are passed without crashing
    try:
        response = embedding(
            model="vertex_ai/text-embedding-004",
            input=["hello"],
            extra_headers={"X-Custom-Header": "test-value"}
        )
    except Exception as e:
        # We expect a 401/404 if no real creds, 
        # but we are checking that it doesn't fail with a TypeError
        if "extra_headers" in str(e):
            pytest.fail("extra_headers not accepted by vertex_ai embedding")

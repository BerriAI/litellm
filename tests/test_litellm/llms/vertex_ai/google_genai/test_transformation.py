from litellm.llms.vertex_ai.google_genai.transformation import (
    VertexAIGoogleGenAIConfig,
)


def test_transform_generate_content_request_defaults_omitted_role():
    config = VertexAIGoogleGenAIConfig()
    contents = [
        {"parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi"}]},
        "Raw content",
    ]

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict={},
    )

    assert result["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi"}]},
        "Raw content",
    ]
    assert "role" not in contents[0]

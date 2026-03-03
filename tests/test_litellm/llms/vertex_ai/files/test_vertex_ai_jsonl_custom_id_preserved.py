import json

from litellm.llms.vertex_ai.files.transformation import VertexAIJsonlFilesTransformation


def test_should_preserve_custom_id_in_vertex_jsonl_transformation():
    """Ensure custom_id is preserved in Vertex JSONL batch uploads."""
    transformer = VertexAIJsonlFilesTransformation()
    openai_jsonl_content = [
        {
            "custom_id": "req-1",
            "body": {"model": "gemini-2.5-flash-lite", "messages": []},
        },
        {
            "custom_id": "req-2",
            "body": {"model": "gemini-2.5-flash-lite", "messages": []},
        },
    ]

    vertex_jsonl_content = transformer._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
        openai_jsonl_content
    )

    assert [item.get("custom_id") for item in vertex_jsonl_content] == [
        "req-1",
        "req-2",
    ]
    assert all("request" in item for item in vertex_jsonl_content)

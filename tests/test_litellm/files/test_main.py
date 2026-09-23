import pytest

import litellm


@pytest.mark.parametrize(
    "custom_llm_provider, purpose",
    [("openai", "batch"), ("vertex_ai", "assistants")],
    ids=["non-vertex-provider", "non-batch-purpose"],
)
def test_create_file_passthrough_is_rejected_outside_a_vertex_batch(custom_llm_provider, purpose):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.create_file(
            file=("batch.jsonl", b'{"request": {"contents": []}}\n', "application/jsonl"),
            purpose=purpose,
            custom_llm_provider=custom_llm_provider,
            passthrough=True,
            api_key="sk-test",
            api_base="http://127.0.0.1:9",
        )

    assert "vertex_ai" in str(exc_info.value)
    assert "batch" in str(exc_info.value)

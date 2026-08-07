import litellm

from litellm.llms.bedrock.files.transformation import BedrockFilesConfig


def test_bedrock_batch_resolves_model_alias_before_provider_mapping(monkeypatch):
    monkeypatch.setitem(
        litellm.model_alias_map,
        "bedrock-batch",
        "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
    )

    result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
        [
            {
                "custom_id": "req-1",
                "body": {
                    "model": "bedrock-batch",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            }
        ]
    )

    assert result == [
        {
            "recordId": "req-1",
            "modelInput": {
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "max_tokens": 16,
                "anthropic_version": "bedrock-2023-05-31",
            },
        }
    ]


def test_bedrock_batch_resolves_model_alias_before_embedding_mapping(monkeypatch):
    monkeypatch.setitem(
        litellm.model_alias_map,
        "bedrock-embedding-batch",
        "bedrock/amazon.titan-embed-text-v2:0",
    )

    result = BedrockFilesConfig()._transform_openai_jsonl_content_to_bedrock_jsonl_content(
        [
            {
                "custom_id": "embedding-1",
                "url": "/v1/embeddings",
                "body": {
                    "model": "bedrock-embedding-batch",
                    "input": "hello",
                },
            }
        ]
    )

    assert result == [
        {
            "recordId": "embedding-1",
            "modelInput": {"inputText": "hello"},
        }
    ]

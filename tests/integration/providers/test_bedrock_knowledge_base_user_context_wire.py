import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

ACCESS_KEY: Final = "AKIAINTEGRATION000003"
USER_CONTEXT: Final = {"userId": "reader@example.com"}
QUERY: Final = "synthetic knowledge base question"
RETRIEVE_RESPONSE: Final = json.dumps(
    {
        "retrievalResults": [
            {
                "content": {"text": "permitted document text"},
                "score": 0.87,
                "metadata": {
                    "x-amz-bedrock-kb-source-uri": "s3://synthetic-bucket/permitted.pdf",
                    "x-amz-bedrock-kb-chunk-id": "chunk-1",
                },
            }
        ]
    }
).encode()


def retrieve_peer(knowledge_base_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == f"/knowledgebases/{knowledge_base_id}/retrieve", (
            request.target
        )
        assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/")
        assert json.loads(request.body) == {
            "retrievalQuery": {"text": QUERY},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 3}},
            "userContext": USER_CONTEXT,
        }, request.body
        return Reply(body=RETRIEVE_RESPONSE)

    return respond


@pytest.mark.covers("providers.bedrock_knowledge_base.search_forwards_user_context_to_retrieve")
def test_vector_store_search_user_context_reaches_bedrock_retrieve_body(gateway: Gateway) -> None:
    knowledge_base_id: Final = f"KB{uuid.uuid4().hex[:8].upper()}"
    with wire_server(retrieve_peer(knowledge_base_id)) as wire, gateway.scenario() as scenario:
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": knowledge_base_id,
                "custom_llm_provider": "bedrock",
                "litellm_params": {
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": ACCESS_KEY,
                    "aws_secret_access_key": "synthetic-knowledge-base-secret-key",
                    "aws_bedrock_runtime_endpoint": wire.url,
                },
            },
        )
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": knowledge_base_id})
        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{knowledge_base_id}/search",
            {"query": QUERY, "max_num_results": 3, "userContext": USER_CONTEXT},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"] == [
            {
                "score": 0.87,
                "content": [{"text": "permitted document text", "type": "text"}],
                "file_id": "s3://synthetic-bucket/permitted.pdf",
                "filename": "permitted.pdf",
                "attributes": {
                    "x-amz-bedrock-kb-source-uri": "s3://synthetic-bucket/permitted.pdf",
                    "x-amz-bedrock-kb-chunk-id": "chunk-1",
                },
            }
        ], response.text
        assert len(wire.drain()) == 1

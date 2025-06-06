import json
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import Mock, patch

import pytest
import litellm
from litellm.integrations.vector_stores.qdrant_vector_store import QdrantVectorStore
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry, LiteLLM_ManagedVectorStore
import os


@pytest.fixture
def setup_vector_store_registry():
    litellm.vector_store_registry = VectorStoreRegistry(
        vector_stores=[
            LiteLLM_ManagedVectorStore(
                vector_store_id="test_collection",
                custom_llm_provider="qdrant",
            )
        ]
    )


@pytest.mark.asyncio
async def test_basic_qdrant_query():
    """
    Verify a basic qdrant query request works.
    """
    litellm._turn_on_debug()
    vector_store = QdrantVectorStore(
        qdrant_api_base=os.getenv("QDRANT_API_BASE"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model="text-embedding-3-small"
    )
    response = await vector_store.make_qdrant_query_request(
        collection_name="qdrant-docs", 
        query="how to get started with qdrant",
        vector_dimension=384
    )
    print("response: ", response)
    response = response["result"]

    # validate the response is a list of results
    assert "points" in response, "Response should contain 'points' key"
    points = response["points"]
    assert isinstance(points, list), "Points should be a list"
    
    # Validate the structure of each point
    for point in points:
        assert "id" in point, "Each point should have an 'id'"
        assert "score" in point, "Each point should have a 'score'"
        assert "payload" in point, "Each point should have a 'payload'"
        assert isinstance(point["payload"], dict), "Payload should be a dictionary"
        
    # Print the first payload for verification
    if len(points) > 0:
        print("First result payload:", points[0]["payload"])


@pytest.mark.asyncio
async def test_basic_qdrant_query_request(setup_vector_store_registry):
    vector_store = QdrantVectorStore(qdrant_api_base="http://localhost:6333")
    with patch.object(vector_store.async_handler, "post") as mock_post:
        with patch("litellm.aembedding") as mock_embed:
            mock_embed.return_value = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": []}
            mock_response.text = json.dumps({"result": []})
            mock_post.return_value = mock_response
            response = await vector_store.make_qdrant_query_request(
                collection_name="test_collection", query="hello"
            )
            assert response == {"result": []}
            mock_post.assert_called_once()
            sent_json = mock_post.call_args.kwargs["json"]
            assert sent_json["query"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_e2e_qdrant_retrieval_with_completion(setup_vector_store_registry):
    litellm._turn_on_debug()
    client = AsyncHTTPHandler()

    async def mock_query(*args, **kwargs):
        return {"result": [{"score": 0.9, "payload": {"text": "context"}}]}

    with patch.object(QdrantVectorStore, "make_qdrant_query_request", side_effect=mock_query):
        with patch.object(client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json = lambda: json.loads(mock_response.text) if mock_response.text else {}
            mock_post.return_value = mock_response
            await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "hello"}],
                vector_store_ids=["test_collection"],
                client=client,
            )
            mock_post.assert_called_once()
            request_body = mock_post.call_args.kwargs["json"]
            content = request_body["messages"][0]["content"]
            assert len(content) == 2
            assert QdrantVectorStore.CONTENT_PREFIX_STRING in content[1]["text"]

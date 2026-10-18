import json
import uuid
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
QUERY: Final = "team provider vector search"
TEAM_ACCESS_KEY: Final = "AKIAINTEGRATIONTEAM01"
STORE_ACCESS_KEY: Final = "AKIASTOREKEY00000001"
_RETRIEVE_RESPONSE: Final = json.dumps(
    {
        "retrievalResults": [
            {
                "content": {"text": "matching Bedrock content"},
                "score": 0.87,
                "metadata": {
                    "x-amz-bedrock-kb-source-uri": "s3://synthetic-bucket/vector-store.txt",
                    "x-amz-bedrock-kb-chunk-id": "chunk-1",
                },
            }
        ]
    }
).encode()
_OPENAI_RESULT: Final[dict[str, JsonValue]] = {
    "file_id": "file-search-wire",
    "filename": "search.txt",
    "score": 0.91,
    "attributes": {"tenant": "a"},
    "content": [{"type": "text", "text": "matching OpenAI content"}],
}
_BEDROCK_RESULT: Final[dict[str, JsonValue]] = {
    "score": 0.87,
    "content": [{"text": "matching Bedrock content", "type": "text"}],
    "file_id": "s3://synthetic-bucket/vector-store.txt",
    "filename": "vector-store.txt",
    "attributes": {
        "x-amz-bedrock-kb-source-uri": "s3://synthetic-bucket/vector-store.txt",
        "x-amz-bedrock-kb-chunk-id": "chunk-1",
    },
}


def _bedrock_retrieve_peer(knowledge_base_id: str, access_key: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"/knowledgebases/{knowledge_base_id}/retrieve", request.target
        assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={access_key}/")
        assert JSON_OBJECT.validate_json(request.body) == {
            "retrievalQuery": {"text": QUERY},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 3}},
        }
        return Reply(body=_RETRIEVE_RESPONSE)

    return respond


def _openai_search_peer(store_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"/v1/vector_stores/{store_id}/search", request.target
        assert request.headers["authorization"] == "Bearer team-openai-key"
        assert JSON_OBJECT.validate_json(request.body)["query"] == QUERY
        return Reply(
            body=json.dumps(
                {
                    "object": "vector_store.search_results.page",
                    "search_query": QUERY,
                    "data": [_OPENAI_RESULT],
                    "has_more": False,
                    "next_page": None,
                }
            ).encode()
        )

    return respond


def test_bedrock_vector_store_search_uses_bedrock_team_model_credentials(gateway: Gateway) -> None:
    knowledge_base_id: Final = f"KB{uuid.uuid4().hex[:8].upper()}"
    with wire_server(_bedrock_retrieve_peer(knowledge_base_id, TEAM_ACCESS_KEY)) as wire, gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.model(
            model="langgraph/repro-agent",
            api_base="http://127.0.0.1:1",
            model_info={"team_id": team},
        )
        scenario.model(
            model="bedrock/us.anthropic.claude-sonnet-5-5",
            api_base=wire.url,
            aws_access_key_id=TEAM_ACCESS_KEY,
            aws_secret_access_key="synthetic-team-secret-key",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            model_info={"team_id": team},
        )
        team_key: Final = scenario.key(team_id=team)
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": knowledge_base_id,
                "custom_llm_provider": "bedrock",
                "litellm_params": {
                    "aws_region_name": "us-east-1",
                    "aws_bedrock_runtime_endpoint": wire.url,
                },
            },
        )
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": knowledge_base_id})

        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{knowledge_base_id}/search",
            {"query": QUERY, "max_num_results": 3},
            key=team_key,
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["data"] == [_BEDROCK_RESULT], response.text
        assert len(wire.drain()) == 1


def test_bedrock_vector_store_search_uses_store_credentials_without_matching_team_model(
    gateway: Gateway,
) -> None:
    knowledge_base_id: Final = f"KB{uuid.uuid4().hex[:8].upper()}"
    with wire_server(_bedrock_retrieve_peer(knowledge_base_id, STORE_ACCESS_KEY)) as wire, gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.model(
            model="langgraph/repro-agent",
            api_base="http://127.0.0.1:1",
            model_info={"team_id": team},
        )
        team_key: Final = scenario.key(team_id=team)
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": knowledge_base_id,
                "custom_llm_provider": "bedrock",
                "litellm_params": {
                    "aws_access_key_id": STORE_ACCESS_KEY,
                    "aws_secret_access_key": "synthetic-store-secret-key",
                    "aws_region_name": "us-east-1",
                    "aws_bedrock_runtime_endpoint": wire.url,
                },
            },
        )
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": knowledge_base_id})

        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{knowledge_base_id}/search",
            {"query": QUERY, "max_num_results": 3},
            key=team_key,
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["data"] == [_BEDROCK_RESULT], response.text
        assert len(wire.drain()) == 1


def test_openai_vector_store_search_without_store_key_uses_team_byok_key(gateway: Gateway) -> None:
    store_id: Final = f"vs_team_{uuid.uuid4().hex}"
    with wire_server(_openai_search_peer(store_id)) as wire, gateway.scenario() as scenario:
        team: Final = scenario.team()
        scenario.model(
            model="openai/gpt-4o-mini",
            api_key="team-openai-key",
            api_base=f"{wire.url}/v1",
            model_info={"team_id": team},
        )
        team_key: Final = scenario.key(team_id=team)
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": store_id,
                "custom_llm_provider": "openai",
                "litellm_params": {},
            },
        )
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})

        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{store_id}/search",
            {"query": QUERY},
            key=team_key,
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["data"] == [_OPENAI_RESULT], response.text
        assert len(wire.drain()) == 1

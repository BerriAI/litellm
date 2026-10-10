from dataclasses import dataclass, field
from typing import Final

import pytest

import litellm
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    VectorStorePreCallHook,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)
from litellm.vector_stores.vector_store_registry import (
    LiteLLM_ManagedVectorStore,
    VectorStoreRegistry,
)
import json
from unittest.mock import patch, Mock
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@dataclass(slots=True)
class _LoggingObject:
    model_call_details: dict[str, object] = field(
        default_factory=lambda: {"litellm_params": {"metadata": {}}}
    )


@dataclass(frozen=True, slots=True)
class _NoProxyRuntime:
    router: "_RecordingRouter"

    def llm_router(self) -> "_RecordingRouter":
        return self.router

    def prisma_client(self) -> None:
        return None


@dataclass
class _RecordingRouter:
    calls: list[dict[str, object]] = field(default_factory=list)

    async def avector_store_search(self, **kwargs: object) -> VectorStoreSearchResponse:
        self.calls.append(kwargs)
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query="what is in the knowledge base?",
            data=[
                VectorStoreSearchResult(
                    score=1.0,
                    content=[VectorStoreResultContent(text="registered context", type="text")],
                )
            ],
        )


@pytest.fixture
def setup_vector_store_registry(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "vector_store_registry",
        VectorStoreRegistry(
            vector_stores=[LiteLLM_ManagedVectorStore(vector_store_id="T37J8R4WTM", custom_llm_provider="bedrock")]
        ),
        raising=False,
    )


@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_without_vector_store_registry(
    setup_vector_store_registry,
):
    litellm.turn_on_debug()
    client = AsyncHTTPHandler()
    litellm.vector_store_registry = None

    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        # Provide proper JSON response content
        mock_response.text = json.dumps(
            {
                "id": "msg_01ABC123",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "LiteLLM is a library that simplifies LLM API access.",
                    }
                ],
                "model": "claude-3.5-sonnet",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        try:
            response = await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids=["T37J8R4WTM"],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the LLM request was made
        mock_post.assert_called_once()

        # Verify the request body
        print("call args:", mock_post.call_args)
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4, default=str))

        # Assert content from the knowedge base was applied to the request

        # 1. we should have 1 content block, the first is the user message
        # There should only be one since there is no initialized vector store registry
        content = request_body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"


@pytest.mark.usefixtures("fake_provider_credentials")
@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_vector_store_not_in_registry(
    setup_vector_store_registry,
):
    """
    No vector store request is made for vector store ids that are not in the registry

    In this test newUnknownVectorStoreId is not in the registry, so no vector store request is made
    """
    litellm.turn_on_debug()
    client = AsyncHTTPHandler()

    if litellm.vector_store_registry is not None:
        print("Registry iniitalized:", litellm.vector_store_registry.vector_stores)
    else:
        print("Registry is None")

    with patch.object(client, "post") as mock_post:
        # Mock the response for the LLM call
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        # Provide proper JSON response content
        mock_response.text = json.dumps(
            {
                "id": "msg_01ABC123",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "LiteLLM is a library that simplifies LLM API access.",
                    }
                ],
                "model": "claude-3.5-sonnet",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response
        try:
            response = await litellm.acompletion(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": "what is litellm?"}],
                vector_store_ids=["newUnknownVectorStoreId"],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        # Verify the LLM request was made
        mock_post.assert_called_once()

        # Verify the request body
        print("call args:", mock_post.call_args)
        request_body = mock_post.call_args.kwargs["json"]
        print("Request body:", json.dumps(request_body, indent=4, default=str))

        # Assert content from the knowedge base was applied to the request

        # 1. we should have 1 content block, the first is the user message
        # There should only be one since there is no initialized vector store registry
        content = request_body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

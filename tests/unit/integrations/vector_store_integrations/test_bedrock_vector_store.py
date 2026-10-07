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


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_without_vector_store_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "vector_store_registry", None)
    messages: Final[list[AllMessageValues]] = [
        {"role": "user", "content": "what is in the knowledge base?"}
    ]
    logging_obj: Final = _LoggingObject()
    router: Final = _RecordingRouter()

    result: Final = await VectorStorePreCallHook(
        proxy_runtime=_NoProxyRuntime(router=router)
    ).async_get_chat_completion_prompt(
        model="chat-model",
        messages=messages,
        non_default_params={"vector_store_ids": ["registered-store"]},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
        litellm_logging_obj=logging_obj,
    )

    assert result[0] == "chat-model"
    assert result[1] is messages
    assert result[1] == [{"role": "user", "content": "what is in the knowledge base?"}]
    assert router.calls == []


@pytest.mark.asyncio
async def test_e2e_bedrock_knowledgebase_retrieval_with_vector_store_not_in_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        litellm,
        "vector_store_registry",
        VectorStoreRegistry(
            vector_stores=[
                LiteLLM_ManagedVectorStore(
                    vector_store_id="registered-store",
                    custom_llm_provider="bedrock",
                )
            ]
        ),
    )
    messages: Final[list[AllMessageValues]] = [
        {"role": "user", "content": "what is in the knowledge base?"}
    ]
    logging_obj: Final = _LoggingObject()
    router: Final = _RecordingRouter()

    result: Final = await VectorStorePreCallHook(
        proxy_runtime=_NoProxyRuntime(router=router)
    ).async_get_chat_completion_prompt(
        model="chat-model",
        messages=messages,
        non_default_params={"vector_store_ids": ["unknown-store"]},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
        litellm_logging_obj=logging_obj,
    )

    assert result[0] == "chat-model"
    assert result[1] is messages
    assert result[1] == [{"role": "user", "content": "what is in the knowledge base?"}]
    assert logging_obj.model_call_details == {"litellm_params": {"metadata": {}}}
    assert router.calls == []

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    VectorStorePreCallHook,
)
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)
from litellm.vector_stores.vector_store_registry import (
    LiteLLM_ManagedVectorStore,
    VectorStoreRegistry,
)


def _search_response(text: str) -> VectorStoreSearchResponse:
    return VectorStoreSearchResponse(
        object="vector_store.search_results.page",
        search_query="what is litellm?",
        data=[
            VectorStoreSearchResult(
                score=1.0,
                content=[VectorStoreResultContent(text=text, type="text")],
            )
        ],
    )


@dataclass
class RecordingRouter:
    failing_vector_store_ids: frozenset[str] = frozenset()
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def avector_store_search(self, **kwargs: Any) -> VectorStoreSearchResponse:
        self.calls.append(kwargs)
        vector_store_id = kwargs["vector_store_id"]
        if vector_store_id in self.failing_vector_store_ids:
            raise litellm.BadRequestError(
                message=f"no healthy deployments for {vector_store_id}",
                model="text-embedding-3-small",
                llm_provider="openai",
            )
        return _search_response(f"context from {vector_store_id}")


@dataclass(frozen=True)
class FakeProxyRuntime:
    router: RecordingRouter | None

    def llm_router(self) -> RecordingRouter | None:
        return self.router

    def prisma_client(self) -> None:
        return None


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def registry_with(monkeypatch: pytest.MonkeyPatch):
    def _register(*vector_store_ids: str, custom_llm_provider: str = "bedrock") -> None:
        monkeypatch.setattr(
            litellm,
            "vector_store_registry",
            VectorStoreRegistry(
                vector_stores=[
                    LiteLLM_ManagedVectorStore(vector_store_id=vector_store_id, custom_llm_provider=custom_llm_provider)
                    for vector_store_id in vector_store_ids
                ],
            ),
        )

    return _register


@pytest.fixture
def warnings():
    handler = RecordingHandler()
    verbose_logger.addHandler(handler)
    yield handler.records
    verbose_logger.removeHandler(handler)


class FakeLoggingObj:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.model_call_details: dict[str, Any] = {"litellm_params": {"metadata": metadata}}


async def _run_hook(hook: VectorStorePreCallHook, vector_store_ids: list[str], logging_obj: FakeLoggingObj):
    return await hook.async_get_chat_completion_prompt(
        model="chat-model",
        messages=[{"role": "user", "content": "what is litellm?"}],
        non_default_params={"vector_store_ids": vector_store_ids},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
        litellm_logging_obj=logging_obj,
    )


@pytest.mark.asyncio
async def test_hook_searches_through_the_injected_router_with_the_request_metadata(registry_with):
    """Regression (LIT-6752): the hook must reach the Router through its injected runtime, not a proxy_server import."""
    registry_with("vs-router")
    router = RecordingRouter()
    logging_obj = FakeLoggingObj({"user_api_key_team_id": "team-a"})

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=router)),
        ["vs-router"],
        logging_obj,
    )

    assert router.calls == [
        {
            "vector_store_id": "vs-router",
            "query": "what is litellm?",
            "custom_llm_provider": "bedrock",
            "metadata": {"user_api_key_team_id": "team-a"},
        }
    ]
    assert messages[0]["content"] == "Context:\n\ncontext from vs-router\n\n"


@pytest.mark.asyncio
async def test_hook_falls_back_to_the_sdk_when_the_runtime_has_no_router(registry_with, warnings):
    registry_with("vs-sdk", custom_llm_provider="lit6752-not-a-provider")

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)),
        ["vs-sdk"],
        FakeLoggingObj({"user_api_key_team_id": "team-a"}),
    )

    assert messages == [{"role": "user", "content": "what is litellm?"}]
    assert len(warnings) == 1
    assert (
        warnings[0]
        .getMessage()
        .startswith("Vector store search failed for vector_store_id=vs-sdk, continuing without its context: ")
    )
    assert "is not a valid LlmProviders" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_every_healthy_vector_store_contributes_its_own_context(registry_with):
    """Regression (LIT-6752): each store appended its context to the original messages, so only the last one survived."""
    registry_with("vs-one", "vs-two")
    router = RecordingRouter()

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=router)),
        ["vs-one", "vs-two"],
        FakeLoggingObj({}),
    )

    assert [message["content"] for message in messages] == [
        "Context:\n\ncontext from vs-one\n\n",
        "Context:\n\ncontext from vs-two\n\n",
        "what is litellm?",
    ]


@pytest.mark.asyncio
async def test_a_failing_vector_store_warns_with_its_id_and_the_other_stores_still_answer(registry_with, warnings):
    """Regression (LIT-6752): one unreachable store must not silently drop every other store's context."""
    registry_with("vs-broken", "vs-healthy")
    router = RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"}))
    logging_obj = FakeLoggingObj({"user_api_key_team_id": "team-a"})

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=router)),
        ["vs-broken", "vs-healthy"],
        logging_obj,
    )

    assert [call["vector_store_id"] for call in router.calls] == ["vs-broken", "vs-healthy"]
    assert messages[0]["content"] == "Context:\n\ncontext from vs-healthy\n\n"
    assert len(logging_obj.model_call_details["search_results"]) == 1
    assert [record.getMessage() for record in warnings] == [
        "Vector store search failed for vector_store_id=vs-broken, continuing without its context: "
        "litellm.BadRequestError: no healthy deployments for vs-broken"
    ]


@pytest.mark.asyncio
async def test_the_only_vector_store_failing_leaves_the_messages_untouched(registry_with, warnings):
    registry_with("vs-broken")
    original_messages = [{"role": "user", "content": "what is litellm?"}]

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(
            proxy_runtime=FakeProxyRuntime(router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"})))
        ),
        ["vs-broken"],
        FakeLoggingObj({}),
    )

    assert messages == original_messages
    assert [(record.levelname, record.getMessage()) for record in warnings] == [
        (
            "WARNING",
            "Vector store search failed for vector_store_id=vs-broken, continuing without its context: "
            "litellm.BadRequestError: no healthy deployments for vs-broken",
        )
    ]

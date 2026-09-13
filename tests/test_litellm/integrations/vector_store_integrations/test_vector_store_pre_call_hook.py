import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    ProxyServerRuntime,
    VectorStorePreCallHook,
)
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIResponse
from litellm.types.utils import (
    CallTypes,
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
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


def _first_message(response: ModelResponse) -> Message:
    choice = response.choices[0]
    assert isinstance(choice, Choices)
    return choice.message


@dataclass(frozen=True)
class ExplodingRegistry:
    async def pop_vector_stores_to_run_with_db_fallback(self, **kwargs: object) -> list[LiteLLM_ManagedVectorStore]:
        raise RuntimeError("the registry blew up")


@dataclass
class RecordingRouter:
    failing_vector_store_ids: frozenset[str] = frozenset()
    calls: list[dict[str, object]] = field(default_factory=list)

    async def avector_store_search(self, **kwargs: object) -> VectorStoreSearchResponse:
        self.calls.append(kwargs)
        vector_store_id = str(kwargs["vector_store_id"])
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


class RegisterStores(Protocol):
    def __call__(self, *vector_store_ids: str, custom_llm_provider: str = "bedrock") -> None: ...


@pytest.fixture
def registry_with(monkeypatch: pytest.MonkeyPatch) -> RegisterStores:
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
def warnings() -> Iterator[list[logging.LogRecord]]:
    handler = RecordingHandler()
    verbose_logger.addHandler(handler)
    yield handler.records
    verbose_logger.removeHandler(handler)


class FakeLoggingObj:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.model_call_details: dict[str, object] = {"litellm_params": {"metadata": metadata}}


async def _run_hook(
    hook: VectorStorePreCallHook,
    vector_store_ids: list[str],
    logging_obj: FakeLoggingObj,
) -> tuple[str, list[AllMessageValues], dict[str, object]]:
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
async def test_hook_searches_through_the_injected_router_with_the_request_metadata(
    registry_with: RegisterStores,
) -> None:
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
async def test_hook_falls_back_to_the_sdk_when_the_runtime_has_no_router(
    registry_with: RegisterStores,
    warnings: list[logging.LogRecord],
) -> None:
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
async def test_every_healthy_vector_store_contributes_its_own_context(registry_with: RegisterStores) -> None:
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
async def test_a_failing_vector_store_warns_with_its_id_and_the_other_stores_still_answer(
    registry_with: RegisterStores,
    warnings: list[logging.LogRecord],
) -> None:
    """Regression (LIT-6752): one unreachable store must not silently drop every other store's context."""
    registry_with("vs-broken", "vs-healthy")
    router = RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"}))
    logging_obj = FakeLoggingObj({"user_api_key_team_id": "team-a"})

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=router)),
        ["vs-broken", "vs-healthy"],
        logging_obj,
    )

    search_results = logging_obj.model_call_details["search_results"]

    assert [call["vector_store_id"] for call in router.calls] == ["vs-broken", "vs-healthy"]
    assert messages[0]["content"] == "Context:\n\ncontext from vs-healthy\n\n"
    assert isinstance(search_results, list)
    assert len(search_results) == 1
    assert [record.getMessage() for record in warnings] == [
        "Vector store search failed for vector_store_id=vs-broken, continuing without its context: "
        "litellm.BadRequestError: no healthy deployments for vs-broken"
    ]


@pytest.mark.asyncio
async def test_the_only_vector_store_failing_leaves_the_messages_untouched(
    registry_with: RegisterStores,
    warnings: list[logging.LogRecord],
) -> None:
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


@pytest.mark.asyncio
async def test_the_default_hook_reaches_the_proxy_router_through_its_runtime(
    registry_with: RegisterStores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (LIT-6752): a hook built with no arguments must still search through the proxy's own Router."""
    from litellm.proxy import proxy_server

    registry_with("vs-default")
    router = RecordingRouter()
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(),
        ["vs-default"],
        FakeLoggingObj({"user_api_key_team_id": "team-a"}),
    )

    assert [call["vector_store_id"] for call in router.calls] == ["vs-default"]
    assert messages[0]["content"] == "Context:\n\ncontext from vs-default\n\n"


def test_the_default_runtime_follows_the_proxy_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    runtime = ProxyServerRuntime()
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert runtime.llm_router() is None
    assert runtime.prisma_client() is None

    router = RecordingRouter()
    prisma = object()
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    assert runtime.llm_router() is router
    assert runtime.prisma_client() is prisma


@pytest.mark.asyncio
async def test_a_failing_vector_store_is_reported_back_to_the_caller(
    registry_with: RegisterStores,
) -> None:
    """Regression (LIT-6809): a silently dropped store left the caller with an un-augmented answer and no signal."""
    registry_with("vs-broken", "vs-healthy")
    logging_obj = FakeLoggingObj({})

    await _run_hook(
        VectorStorePreCallHook(
            proxy_runtime=FakeProxyRuntime(router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"})))
        ),
        ["vs-broken", "vs-healthy"],
        logging_obj,
    )

    response = ModelResponse(choices=[Choices(message=Message(content="an answer"))])
    await VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)).async_post_call_success_deployment_hook(
        request_data={"litellm_logging_obj": logging_obj},
        response=response,
        call_type=CallTypes.acompletion,
    )

    provider_specific_fields = _first_message(response).provider_specific_fields or {}
    assert provider_specific_fields["vector_store_search_failures"] == (
        {
            "vector_store_id": "vs-broken",
            "custom_llm_provider": "bedrock",
            "error": "litellm.BadRequestError: no healthy deployments for vs-broken",
        },
    )
    assert len(provider_specific_fields["search_results"]) == 1


@pytest.mark.asyncio
async def test_a_healthy_vector_store_alone_reports_no_failures(registry_with: RegisterStores) -> None:
    registry_with("vs-healthy")
    logging_obj = FakeLoggingObj({})

    await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=RecordingRouter())),
        ["vs-healthy"],
        logging_obj,
    )

    response = ModelResponse(choices=[Choices(message=Message(content="an answer"))])
    await VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)).async_post_call_success_deployment_hook(
        request_data={"litellm_logging_obj": logging_obj},
        response=response,
        call_type=CallTypes.acompletion,
    )

    assert "vector_store_search_failures" not in (_first_message(response).provider_specific_fields or {})


@pytest.mark.asyncio
async def test_a_failing_vector_store_is_reported_on_the_responses_api_response(
    registry_with: RegisterStores,
) -> None:
    """Regression (LIT-6809): /v1/responses answered 200 with no sign the knowledge base was missing."""
    registry_with("vs-broken")
    logging_obj = FakeLoggingObj({})

    await _run_hook(
        VectorStorePreCallHook(
            proxy_runtime=FakeProxyRuntime(router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"})))
        ),
        ["vs-broken"],
        logging_obj,
    )

    response = ResponsesAPIResponse(id="resp-lit6809", created_at=0, output=[])
    await VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)).async_post_call_success_deployment_hook(
        request_data={"litellm_logging_obj": logging_obj},
        response=response,
        call_type=CallTypes.aresponses,
    )

    assert response.model_dump()["vector_store_search_failures"] == [
        {
            "vector_store_id": "vs-broken",
            "custom_llm_provider": "bedrock",
            "error": "litellm.BadRequestError: no healthy deployments for vs-broken",
        }
    ]


@pytest.mark.asyncio
async def test_a_healthy_vector_store_leaves_the_responses_api_response_alone(registry_with: RegisterStores) -> None:
    registry_with("vs-healthy")
    logging_obj = FakeLoggingObj({})

    await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=RecordingRouter())),
        ["vs-healthy"],
        logging_obj,
    )

    response = ResponsesAPIResponse(id="resp-lit6809", created_at=0, output=[])
    await VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)).async_post_call_success_deployment_hook(
        request_data={"litellm_logging_obj": logging_obj},
        response=response,
        call_type=CallTypes.aresponses,
    )

    assert "vector_store_search_failures" not in response.model_dump()


@pytest.mark.asyncio
async def test_a_failing_vector_store_is_reported_on_the_streaming_chunk(registry_with: RegisterStores) -> None:
    registry_with("vs-broken")
    logging_obj = FakeLoggingObj({})

    await _run_hook(
        VectorStorePreCallHook(
            proxy_runtime=FakeProxyRuntime(router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"})))
        ),
        ["vs-broken"],
        logging_obj,
    )

    chunk = ModelResponseStream(choices=[StreamingChoices(delta=Delta(content="an answer"))])
    await VectorStorePreCallHook(
        proxy_runtime=FakeProxyRuntime(router=None)
    ).async_post_call_streaming_deployment_hook(
        request_data=logging_obj.model_call_details,
        response_chunk=chunk,
        call_type=CallTypes.acompletion,
    )

    assert (chunk.choices[0].delta.provider_specific_fields or {})["vector_store_search_failures"] == (
        {
            "vector_store_id": "vs-broken",
            "custom_llm_provider": "bedrock",
            "error": "litellm.BadRequestError: no healthy deployments for vs-broken",
        },
    )


@pytest.mark.asyncio
async def test_error_mode_fails_the_request_instead_of_answering_without_the_knowledge_base(
    registry_with: RegisterStores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (LIT-6809): opting in must turn an ungrounded answer into a 400 the caller can act on."""
    registry_with("vs-broken", "vs-healthy")
    monkeypatch.setattr(litellm, "vector_store_search_failure_mode", "error")

    with pytest.raises(litellm.VectorStoreSearchError) as raised:
        await _run_hook(
            VectorStorePreCallHook(
                proxy_runtime=FakeProxyRuntime(
                    router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"}))
                )
            ),
            ["vs-broken", "vs-healthy"],
            FakeLoggingObj({}),
        )

    assert raised.value.status_code == 400
    assert raised.value.failures == (
        {
            "vector_store_id": "vs-broken",
            "custom_llm_provider": "bedrock",
            "error": "litellm.BadRequestError: no healthy deployments for vs-broken",
        },
    )
    assert "vs-broken: litellm.BadRequestError: no healthy deployments for vs-broken" in raised.value.message


@pytest.mark.asyncio
async def test_a_misspelled_failure_mode_annotates_instead_of_erroring_the_request(
    registry_with: RegisterStores,
    monkeypatch: pytest.MonkeyPatch,
    warnings: list[logging.LogRecord],
) -> None:
    """Regression (LIT-6809): litellm_settings takes any value, so a typo must not become a 500."""
    registry_with("vs-broken")
    monkeypatch.setattr(litellm, "vector_store_search_failure_mode", "erorr")

    logging_obj = FakeLoggingObj({})
    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(
            proxy_runtime=FakeProxyRuntime(router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"})))
        ),
        ["vs-broken"],
        logging_obj,
    )

    assert messages[0]["content"] == "what is litellm?"
    assert logging_obj.model_call_details["vector_store_search_failures"] == (
        {
            "vector_store_id": "vs-broken",
            "custom_llm_provider": "bedrock",
            "error": "litellm.BadRequestError: no healthy deployments for vs-broken",
        },
    )
    assert any("erorr" in record.getMessage() for record in warnings)


@pytest.mark.asyncio
async def test_error_mode_leaves_a_fully_healthy_request_alone(
    registry_with: RegisterStores,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_with("vs-healthy")
    monkeypatch.setattr(litellm, "vector_store_search_failure_mode", "error")

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=RecordingRouter())),
        ["vs-healthy"],
        FakeLoggingObj({}),
    )

    assert messages[0]["content"] == "Context:\n\ncontext from vs-healthy\n\n"


@pytest.mark.asyncio
async def test_error_mode_does_not_swallow_the_raise_in_the_hooks_own_catch_all(
    registry_with: RegisterStores,
    monkeypatch: pytest.MonkeyPatch,
    warnings: list[logging.LogRecord],
) -> None:
    """Regression (LIT-6809): the catch-all around the hook must not turn the opted-in failure back into a 200."""
    registry_with("vs-broken")
    monkeypatch.setattr(litellm, "vector_store_search_failure_mode", "error")

    with pytest.raises(litellm.VectorStoreSearchError):
        await _run_hook(
            VectorStorePreCallHook(
                proxy_runtime=FakeProxyRuntime(
                    router=RecordingRouter(failing_vector_store_ids=frozenset({"vs-broken"}))
                )
            ),
            ["vs-broken"],
            FakeLoggingObj({}),
        )

    assert [record.levelname for record in warnings] == ["WARNING"]


@pytest.mark.asyncio
async def test_a_crash_outside_the_search_names_the_requested_vector_stores(
    monkeypatch: pytest.MonkeyPatch,
    warnings: list[logging.LogRecord],
) -> None:
    """Regression (LIT-6809): the catch-all logged no store id, so an operator could not tell which store broke."""
    monkeypatch.setattr(litellm, "vector_store_registry", ExplodingRegistry())

    _, messages, _ = await _run_hook(
        VectorStorePreCallHook(proxy_runtime=FakeProxyRuntime(router=None)),
        ["vs-one", "vs-two"],
        FakeLoggingObj({}),
    )

    assert messages == [{"role": "user", "content": "what is litellm?"}]
    assert [record.getMessage() for record in warnings] == [
        "Error in VectorStorePreCallHook for vector_store_ids=('vs-one', 'vs-two'): the registry blew up"
    ]

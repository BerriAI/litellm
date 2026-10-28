import datetime
import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import ModelResponse

API_BASE: Final = "https://agents-test.services.ai.azure.com/api/projects/litellm-test"
API_VERSION: Final = "2025-05-01"
AGENT_ID: Final = "asst_hbnoK9BOCcHhC3lC4MDroVGG"


@pytest.fixture(autouse=True)
def _httpx_only_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


def _assistants_api(thread_id: str, answer: str) -> respx.MockRouter:
    thread_path: Final = f"/api/projects/litellm-test/threads/{thread_id}"
    router: Final = respx.mock(assert_all_called=False)
    router.post(path="/api/projects/litellm-test/threads").respond(
        200, json={"id": thread_id, "object": "thread", "created_at": 1767225600, "metadata": {}}
    )
    router.post(path=f"{thread_path}/messages").respond(
        200, json={"id": "msg_user", "object": "thread.message", "thread_id": thread_id, "role": "user", "content": []}
    )
    router.post(path=f"{thread_path}/runs").respond(
        200,
        json={
            "id": "run_1",
            "object": "thread.run",
            "thread_id": thread_id,
            "assistant_id": AGENT_ID,
            "status": "queued",
        },
    )
    router.get(path=f"{thread_path}/runs/run_1").respond(
        200, json={"id": "run_1", "object": "thread.run", "thread_id": thread_id, "status": "completed"}
    )
    router.get(path=f"{thread_path}/messages").respond(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "id": "msg_assistant",
                    "object": "thread.message",
                    "thread_id": thread_id,
                    "role": "assistant",
                    "content": [{"type": "text", "text": {"value": answer, "annotations": []}}],
                },
                {
                    "id": "msg_user",
                    "object": "thread.message",
                    "thread_id": thread_id,
                    "role": "user",
                    "content": [{"type": "text", "text": {"value": "earlier", "annotations": []}}],
                },
            ],
            "first_id": "msg_assistant",
            "last_id": "msg_user",
            "has_more": False,
        },
    )
    return router


def _recorded(router: respx.MockRouter) -> list[tuple[str, str, str, object]]:
    return [
        (
            call.request.method,
            f"{call.request.url.path}?{call.request.url.query.decode()}",
            call.request.headers["authorization"],
            json.loads(call.request.content) if call.request.content else None,
        )
        for call in router.calls
    ]


@pytest.mark.asyncio
async def test_acompletion_creates_thread_posts_message_runs_and_returns_the_reply() -> None:
    router: Final = _assistants_api("thread_new", "25 * 4 = 100")

    with router:
        response: Final = await litellm.acompletion(
            model=f"azure_ai/agents/{AGENT_ID}",
            messages=[{"role": "user", "content": "Hi Agent, what is 25 * 4?"}],
            api_base=API_BASE,
            api_key="azure-ad-token",
            stream=False,
        )
        recorded: Final = _recorded(router)

    query: Final = f"api-version={API_VERSION}"
    auth: Final = "Bearer azure-ad-token"
    assert recorded == [
        ("POST", f"/api/projects/litellm-test/threads?{query}", auth, None),
        (
            "POST",
            f"/api/projects/litellm-test/threads/thread_new/messages?{query}",
            auth,
            {"role": "user", "content": "Hi Agent, what is 25 * 4?"},
        ),
        ("POST", f"/api/projects/litellm-test/threads/thread_new/runs?{query}", auth, {"assistant_id": AGENT_ID}),
        ("GET", f"/api/projects/litellm-test/threads/thread_new/runs/run_1?{query}", auth, None),
        ("GET", f"/api/projects/litellm-test/threads/thread_new/messages?{query}", auth, None),
    ]
    assert response.choices[0].message.content == "25 * 4 = 100"
    assert response._hidden_params["thread_id"] == "thread_new"


@pytest.mark.asyncio
async def test_handler_acompletion_with_thread_id_reuses_that_thread() -> None:
    router: Final = _assistants_api("thread_existing", "Your name is Alice.")
    messages: Final[list[dict[str, object]]] = [{"role": "user", "content": "What is my name?"}]
    logging_obj: Final = Logging(
        model=f"azure_ai/agents/{AGENT_ID}",
        messages=messages,
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1),
        litellm_call_id="azure-agents-thread-reuse",
        function_id="azure-agents-thread-reuse",
    )

    with router:
        response: Final = await AzureAIAgentsHandler().acompletion(
            model=f"agents/{AGENT_ID}",
            messages=messages,
            api_base=API_BASE,
            api_key="azure-ad-token",
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            optional_params={"thread_id": "thread_existing"},
            litellm_params={},
            timeout=30.0,
            client=AsyncHTTPHandler(),
        )
        recorded: Final = _recorded(router)

    assert [(method, url.split("?")[0], body) for method, url, _, body in recorded] == [
        ("POST", "/api/projects/litellm-test/threads/thread_existing/messages", messages[0]),
        ("POST", "/api/projects/litellm-test/threads/thread_existing/runs", {"assistant_id": AGENT_ID}),
        ("GET", "/api/projects/litellm-test/threads/thread_existing/runs/run_1", None),
        ("GET", "/api/projects/litellm-test/threads/thread_existing/messages", None),
    ]
    assert response.choices[0].message.content == "Your name is Alice."
    assert response._hidden_params["thread_id"] == "thread_existing"


def _sse(event: str, data: dict[str, object] | str) -> str:
    return f"event: {event}\ndata: {data if isinstance(data, str) else json.dumps(data)}\n\n"


def _message_delta(text: str) -> dict[str, object]:
    return {
        "id": "msg_stream",
        "object": "thread.message.delta",
        "delta": {"content": [{"index": 0, "type": "text", "text": {"value": text}}]},
    }


@pytest.mark.asyncio
async def test_acompletion_stream_yields_agent_message_deltas_then_stop(respx_mock: respx.MockRouter) -> None:
    thread: Final = {"id": "thread_stream", "object": "thread", "created_at": 1767225600, "metadata": {}}
    run: Final = {"id": "run_stream", "object": "thread.run", "thread_id": "thread_stream", "assistant_id": AGENT_ID}
    message: Final = {"id": "msg_stream", "object": "thread.message", "thread_id": "thread_stream", "role": "assistant"}
    completed_content: Final = [{"type": "text", "text": {"value": "10 + 5 = 15", "annotations": []}}]
    stream_body: Final = "".join(
        (
            _sse("thread.created", thread),
            _sse("thread.run.created", {**run, "status": "queued"}),
            _sse("thread.message.created", {**message, "status": "in_progress", "content": []}),
            _sse("thread.message.delta", _message_delta("10 + 5")),
            _sse("thread.message.delta", _message_delta(" = 15")),
            _sse("thread.message.completed", {**message, "status": "completed", "content": completed_content}),
            _sse("thread.run.completed", {**run, "status": "completed"}),
            _sse("done", "[DONE]"),
        )
    )
    route: Final = respx_mock.post(f"{API_BASE}/threads/runs").respond(
        200, text=stream_body, headers={"content-type": "text/event-stream"}
    )

    response: Final = await litellm.acompletion(
        model=f"azure_ai/agents/{AGENT_ID}",
        messages=[{"role": "user", "content": "Hi Agent, what is 10 + 5?"}],
        api_base=API_BASE,
        api_key="azure-ad-token",
        stream=True,
    )
    chunks: Final = [(chunk.choices[0].delta.content, chunk.choices[0].finish_reason) async for chunk in response]

    assert route.calls.last.request.url.params["api-version"] == API_VERSION
    assert json.loads(route.calls.last.request.content) == {
        "assistant_id": AGENT_ID,
        "stream": True,
        "thread": {"messages": [{"role": "user", "content": "Hi Agent, what is 10 + 5?"}]},
    }
    assert chunks == [("10 + 5", None), (" = 15", None), (None, "stop")]

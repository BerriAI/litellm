"""Azure AI Foundry Agents behind `/v1/chat/completions`: the thread, message, run and poll call sequence.

An `azure_ai/agents/<agent_id>` model is detected by `AzureFoundryModelInfo.get_azure_ai_route` and served by
`AzureAIAgentsHandler` (`litellm/llms/azure_ai/agents/handler.py`), which turns one chat call into the Agent
Service's Assistants-style flow, all against `{api_base}/...` with `?api-version=2025-05-01` by default:

1. `POST threads` (body `{}`), unless the caller passes a `thread_id`.
2. One `POST threads/{thread}/messages` per `user` or `system` message, always sent with `role: user`;
   assistant turns in the history are not posted.
3. `POST threads/{thread}/runs` with `{"assistant_id": <agent_id>}` (plus `instructions` if given).
4. `GET threads/{thread}/runs/{run}` until `completed`; `failed`, `cancelled` or `expired` raise
   `Run <status>: <last_error.message>` as a 500.
5. `GET threads/{thread}/messages`, and the first assistant text becomes the chat answer.

The deployment key is sent as `Authorization: Bearer <key>` on every call (the service expects an Entra
token). With `stream: true` the handler instead posts once to `threads/runs` with `stream: true` and relays
`thread.message.delta` events as chat chunks. Usage is estimated with the gpt-3.5 tokenizer, since the service
reports none. A Rust port must keep this whole sequence; any reordering changes what the agent sees.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Final
from urllib.parse import urlsplit

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure3_support import (
    AGENTS_DEFAULT_API_VERSION,
    agent_messages,
    agent_run,
    agent_sse,
    agent_thread,
    full_peer,
)
from integration.providers.dec1_azure_support import (
    FOUNDRY_KEY,
    error_message,
    json_reply,
    query_of,
    sse_events,
    streamed_text,
)

AGENT_ID: Final = "asst_dec1synthetic"
MODEL: Final = f"azure_ai/agents/{AGENT_ID}"


def _agent_peer(
    *, answer: str = "agent answer", run_status: str = "completed", annotations: tuple[object, ...] = ()
) -> tuple[Callable[[Request], Reply], str]:
    thread_id: Final = f"thread_{uuid.uuid4().hex[:12]}"
    run_id: Final = f"run_{uuid.uuid4().hex[:12]}"

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path.endswith("/threads"):
            return json_reply(agent_thread(thread_id))
        if request.method == "POST" and path.endswith("/messages"):
            return json_reply({"id": f"msg_{uuid.uuid4().hex}", "object": "thread.message"})
        if request.method == "POST" and path.endswith("/runs"):
            return json_reply(agent_run(run_id, thread_id, AGENT_ID, "queued"))
        if request.method == "GET" and path.endswith(f"/runs/{run_id}"):
            extra = (
                {"last_error": {"code": "server_error", "message": "tool crashed"}} if run_status == "failed" else {}
            )
            return json_reply(agent_run(run_id, thread_id, AGENT_ID, run_status, **extra))  # pyright: ignore[reportArgumentType]
        if request.method == "GET" and path.endswith("/messages"):
            return json_reply(agent_messages(answer, annotations=annotations))  # pyright: ignore[reportArgumentType]
        return json_reply({"error": {"message": f"unexpected {request.method} {path}"}}, status=404)

    return respond, thread_id


def _agents(gateway: Gateway, respond: Callable[[Request], Reply], *, model: str, **params: object):  # noqa: ANN202
    """An `azure_ai/agents/...` deployment whose every request (run polls are GETs) reaches `respond`."""
    return full_peer(gateway, respond, model=model, api_key=FOUNDRY_KEY, **params)  # pyright: ignore[reportArgumentType]


def _calls(requests: tuple[Request, ...]) -> list[tuple[str, str]]:
    return [(request.method, urlsplit(request.target).path) for request in requests]


def _all_requests(rig: object) -> tuple[Request, ...]:
    return rig.wire.drain()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


def test_one_chat_call_runs_the_full_thread_message_run_poll_sequence(gateway: Gateway) -> None:
    """An `azure_ai/agents/asst_dec1synthetic` deployment; the caller sends one user message.

    The peer must see exactly, in order: `POST /threads`, `POST /threads/<t>/messages`, `POST /threads/<t>/runs`,
    `GET /threads/<t>/runs/<r>` (completed on the first poll), `GET /threads/<t>/messages`. A probe `GET` the
    proxy may send is not a `/threads` path and is ignored.
    """
    respond, thread_id = _agent_peer()
    with _agents(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat("hello agent")
        assert response.status_code == 200, response.text
        calls: Final = [call for call in _calls(_all_requests(rig)) if "/threads" in call[1]]
        assert [method for method, _ in calls] == ["POST", "POST", "POST", "GET", "GET"], calls
        assert calls[0][1] == "/threads"
        assert calls[1][1] == f"/threads/{thread_id}/messages"
        assert calls[2][1] == f"/threads/{thread_id}/runs"
        assert calls[3][1].startswith(f"/threads/{thread_id}/runs/run_")
        assert calls[4][1] == f"/threads/{thread_id}/messages"


def test_every_agent_call_carries_the_default_api_version_and_the_key_as_bearer(gateway: Gateway) -> None:
    """The deployment sets no `api_version`.

    Every `/threads` call must carry query exactly `{"api-version": "2025-05-01"}` and
    `Authorization: Bearer synthetic-foundry-key`, never an `api-key` header.
    """
    respond, _ = _agent_peer()
    with _agents(gateway, respond, model=MODEL) as rig:
        assert rig.chat().status_code == 200
        calls: Final = [r for r in _all_requests(rig) if "/threads" in urlsplit(r.target).path]
        assert {json.dumps(query_of(r)) for r in calls} == {json.dumps({"api-version": AGENTS_DEFAULT_API_VERSION})}
        assert {r.headers.get("authorization") for r in calls} == {f"Bearer {FOUNDRY_KEY}"}
        assert all("api-key" not in r.headers for r in calls)


def test_the_run_names_the_agent_id_from_the_model(gateway: Gateway) -> None:
    """The run creation body must be exactly `{"assistant_id": "asst_dec1synthetic"}`, the part of the model after
    `agents/` (`get_agent_id_from_model`). The thread creation is posted with no body at all: the handler sends
    `data=json.dumps(json_data) if json_data else None`, and the `{}` it passes is falsy
    (litellm/llms/azure_ai/agents/handler.py:331, :361 and :457 on main).
    """
    respond, _ = _agent_peer()
    with _agents(gateway, respond, model=MODEL) as rig:
        rig.chat()
        posts: Final = {urlsplit(r.target).path.rsplit("/", 1)[-1]: r for r in _all_requests(rig) if r.method == "POST"}
        assert json.loads(posts["runs"].body) == {"assistant_id": AGENT_ID}
        assert posts["threads"].body == b""


def test_system_messages_are_posted_as_user_messages_and_assistant_turns_are_skipped(gateway: Gateway) -> None:
    """The caller sends system `be brief`, user `first`, assistant `earlier answer`, user `second`.

    The handler posts one thread message per system or user message, always with `role: user`, so the message
    bodies must be exactly `[{"role": "user", "content": "be brief"}, {..."first"}, {..."second"}]` in that
    order, and the assistant turn must not be posted.
    """
    respond, _ = _agent_peer()
    with _agents(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "earlier answer"},
                    {"role": "user", "content": "second"},
                ]
            }
        )
        assert response.status_code == 200, response.text
        posted: Final = [
            json.loads(r.body)
            for r in _all_requests(rig)
            if r.method == "POST" and urlsplit(r.target).path.endswith("/messages")
        ]
        assert posted == [
            {"role": "user", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]


def test_the_first_assistant_text_is_the_chat_answer_with_finish_reason_stop(gateway: Gateway) -> None:
    """The messages page lists the assistant answer `agent answer` before an older user turn.

    The caller must get one choice with `message.role == "assistant"`, `message.content == "agent answer"` and
    `finish_reason == "stop"`.
    """
    respond, _ = _agent_peer(answer="agent answer")
    with _agents(gateway, respond, model=MODEL) as rig:
        choice: Final = rig.chat().json()["choices"][0]
        assert (choice["message"]["role"], choice["message"]["content"], choice["finish_reason"]) == (
            "assistant",
            "agent answer",
            "stop",
        )


def test_usage_is_estimated_because_the_agent_service_reports_none(gateway: Gateway) -> None:
    """The agent service returns no token counts.

    The handler estimates usage with the gpt-3.5 tokenizer, so the answer must carry `usage` with
    `prompt_tokens > 0`, `completion_tokens > 0` and `total_tokens == prompt_tokens + completion_tokens`.
    """
    respond, _ = _agent_peer(answer="a reasonably long agent answer")
    with _agents(gateway, respond, model=MODEL) as rig:
        usage: Final = rig.chat("count my tokens please").json()["usage"]
        assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_url_citation_annotations_are_reshaped_into_openai_annotations(gateway: Gateway) -> None:
    """The assistant text carries a Foundry annotation with `start_index`/`end_index` beside `url_citation`.

    `_transform_annotations` moves the indexes inside `url_citation`, so `message.annotations` must be exactly
    `[{"type": "url_citation", "url_citation": {"url": "https://example.invalid/doc", "title": "Doc",
    "start_index": 10, "end_index": 13}}]`.
    """
    annotation: Final = {
        "type": "url_citation",
        "text": "[1]",
        "start_index": 10,
        "end_index": 13,
        "url_citation": {"url": "https://example.invalid/doc", "title": "Doc"},
    }
    respond, _ = _agent_peer(answer="see source [1]", annotations=(annotation,))
    with _agents(gateway, respond, model=MODEL) as rig:
        message: Final = rig.chat().json()["choices"][0]["message"]
        assert message["annotations"] == [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.invalid/doc",
                    "title": "Doc",
                    "start_index": 10,
                    "end_index": 13,
                },
            }
        ]


def test_a_failed_run_is_a_500_naming_the_run_error(gateway: Gateway) -> None:
    """The run poll answers `status: failed` with `last_error.message: tool crashed`.

    The caller must get HTTP 500 and the error message must contain `Run failed: tool crashed`, and the handler
    must not fetch the messages page afterwards.
    """
    respond, thread_id = _agent_peer(run_status="failed")
    with _agents(gateway, respond, model=MODEL, num_retries=0) as rig:
        response: Final = rig.chat_body({"messages": [{"role": "user", "content": "x"}], "num_retries": 0})
        assert response.status_code == 500, response.text
        assert "Run failed: tool crashed" in error_message(response)
        assert ("GET", f"/threads/{thread_id}/messages") not in _calls(_all_requests(rig))


def test_a_refused_thread_creation_keeps_the_service_status(gateway: Gateway) -> None:
    """The service answers 401 to `POST /threads`.

    `_check_response` raises `AzureAIAgentsError` with the upstream status (litellm/llms/azure_ai/agents/handler.py:286-292
    on main), so the caller must get HTTP 401 and no run must be created. The error is mapped to an
    `AuthenticationError` whose message carries the service's own error body (`bad token`), not the handler's
    `Failed to create thread` prefix, which round 2 showed is dropped on the way out.
    """

    def refuse(request: Request) -> Reply:
        return json_reply({"error": {"code": "Unauthorized", "message": "bad token"}}, status=401)

    with _agents(gateway, refuse, model=MODEL) as rig:
        response: Final = rig.chat_body({"messages": [{"role": "user", "content": "x"}], "num_retries": 0})
        assert response.status_code == 401, response.text
        assert "bad token" in error_message(response)
        assert not any(path.endswith("/runs") for _, path in _calls(_all_requests(rig)))


def test_a_streamed_call_posts_once_to_threads_runs_with_the_thread_inline(gateway: Gateway) -> None:
    """The caller sends `stream: true` with one user message `hi`.

    The handler must make exactly one `POST /threads/runs?api-version=2025-05-01` whose body is
    `{"assistant_id": "asst_dec1synthetic", "stream": true, "thread": {"messages": [{"role": "user",
    "content": "hi"}]}}`, and no separate thread, message or poll calls.
    """
    with _agents(gateway, lambda _: agent_sse("thread_stream", ("Hel", "lo")), model=MODEL) as rig:
        response: Final = rig.chat("hi", stream=True)
        assert response.status_code == 200, response.text
        posts: Final = [r for r in _all_requests(rig) if r.method == "POST"]
        assert [urlsplit(r.target).path for r in posts] == ["/threads/runs"]
        assert query_of(posts[0]) == {"api-version": AGENTS_DEFAULT_API_VERSION}
        assert json.loads(posts[0].body) == {
            "assistant_id": AGENT_ID,
            "stream": True,
            "thread": {"messages": [{"role": "user", "content": "hi"}]},
        }


def test_streamed_message_deltas_reach_the_caller_as_chat_chunks_ending_in_stop(gateway: Gateway) -> None:
    """The SSE stream carries `thread.message.delta` events `Hel` and `lo`, then `[DONE]`.

    The caller's chunks must concatenate to `Hello`, the last chunk with a choice must have
    `finish_reason: stop`, and the stream must end with `data: [DONE]`.
    """
    with _agents(gateway, lambda _: agent_sse("thread_stream", ("Hel", "lo")), model=MODEL) as rig:
        response: Final = rig.chat("hi", stream=True)
        events: Final = sse_events(response)
        assert streamed_text(events) == "Hello"
        finishes: Final = [c.get("finish_reason") for e in events for c in (e.get("choices") or [])]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
        assert finishes[-1] == "stop", events
        assert response.text.rstrip().endswith("data: [DONE]")

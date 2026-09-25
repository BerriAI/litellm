import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import JsonValue
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from liteadmin.agent import ToolContext, run_agent
from liteadmin.models import Action, ActionResult, ChatRequest


def tool(name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "role": "assistant",
        "content": "I will perform the requested action.",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


@dataclass
class Transcript:
    model_requests: list[dict[str, JsonValue]] = field(default_factory=list)
    operations: list[httpx.Request] = field(default_factory=list)
    approvals: list[Action] = field(default_factory=list)
    results: list[ActionResult] = field(default_factory=list)


async def run(
    first: dict[str, JsonValue],
    response: Callable[[httpx.Request], httpx.Response],
    approved: bool = True,
) -> Transcript:
    transcript = Transcript()

    def model_request(request: httpx2.Request) -> httpx2.Response:
        transcript.model_requests.append(json.loads(request.content))
        message = first if len(transcript.model_requests) == 1 else {"role": "assistant", "content": "Done."}
        return httpx2.Response(
            200,
            json={
                "id": "reply",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(transcript.model_requests) == 1 else "stop",
                    }
                ],
            },
        )

    def management_request(request: httpx.Request) -> httpx.Response:
        transcript.operations.append(request)
        return response(request)

    async def confirm(action: Action) -> bool:
        assert transcript.operations == []
        transcript.approvals.append(action)
        return approved

    async def result(action: Action, outcome: ActionResult) -> None:
        assert action.id == transcript.approvals[-1].id
        transcript.results.append(outcome)

    async with (
        httpx.AsyncClient(
            base_url="http://gateway",
            headers={"Authorization": "Bearer private-session"},
            transport=httpx.MockTransport(management_request),
        ) as management,
        httpx2.AsyncClient(transport=httpx2.MockTransport(model_request)) as inference,
        AsyncOpenAI(api_key="test", base_url="http://model", http_client=inference, max_retries=0) as model_client,
    ):
        model = OpenAIChatModel("test", provider=OpenAIProvider(openai_client=model_client))
        request = ChatRequest(
            model="test",
            messages=[{"role": "user", "content": "Perform the requested gateway action"}],
            inference_base_url="",
        )
        context = ToolContext(management, ("private-session",), confirm, result, asyncio.Lock())
        answers = [message async for message in run_agent(request, context, model)]
        assert answers == ["Done."]
    return transcript


@pytest.mark.asyncio
async def test_real_sdk_reads_the_gateway_and_removes_credentials_from_model_results() -> None:
    transcript = await run(
        tool("team_info", {"team_id": "team-1"}),
        lambda request: httpx.Response(
            200,
            json={
                "team_info": {
                    "team_id": "team-1",
                    "max_budget": 40,
                    "metadata": {"credential": "private-session"},
                    "team_alias": "private-session sk-secret",
                }
            },
        ),
    )
    assert len(transcript.operations) == 1
    assert transcript.operations[0].url == "http://gateway/team/info?team_id=team-1&key_limit=20"
    assert transcript.operations[0].headers["Authorization"] == "Bearer private-session"
    assert transcript.approvals == []
    tool_result = json.dumps(transcript.model_requests[1]["messages"])
    assert "private-session" not in tool_result
    assert "sk-secret" not in tool_result
    assert "40" in tool_result


@pytest.mark.asyncio
async def test_real_sdk_executes_the_exact_approved_arguments_and_sends_key_only_to_action_card() -> None:
    transcript = await run(
        tool("key_create", {"key_alias": "SDK key", "team_id": "team-1", "max_budget": 25, "models": None}),
        lambda request: httpx.Response(200, json={"key": "sk-created", "key_alias": "SDK key"}),
    )
    assert len(transcript.operations) == 1
    expected = {"key_alias": "SDK key", "team_id": "team-1", "max_budget": 25}
    assert transcript.approvals[0].arguments == expected
    assert json.loads(transcript.operations[0].content) == expected
    assert transcript.operations[0].url.path == "/key/generate"
    assert transcript.results[0].model_dump() == {"status": "completed", "key": "sk-created"}
    assert "sk-created" not in json.dumps(transcript.model_requests)


@pytest.mark.asyncio
async def test_declining_a_write_never_calls_the_management_api() -> None:
    transcript = await run(
        tool("team_update", {"team_id": "team-1", "max_budget": 50}),
        lambda request: httpx.Response(200, json={}),
        approved=False,
    )
    assert len(transcript.approvals) == 1
    assert transcript.operations == []
    assert transcript.results == []
    assert "cancelled" in json.dumps(transcript.model_requests[1])


@pytest.mark.asyncio
async def test_an_uncertain_write_is_not_retried_and_is_reported_as_unknown() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Lost response after the server applied the write", request=request)

    transcript = await run(tool("team_update", {"team_id": "team-1", "max_budget": 50}), timeout)
    assert len(transcript.operations) == 1
    assert transcript.results[0].status == "unknown"
    assert "Do not retry" in json.dumps(transcript.model_requests[1])

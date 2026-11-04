import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.llms.qca.interactions.handler import create_qca_interaction
from litellm.types.interactions import InteractionsAPIResponse


def _sse_event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


@pytest.mark.asyncio
async def test_create_qca_interaction_maps_session_events_to_completed_response() -> None:
    requests: list[httpx.Request] = []

    def route(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json={"id": "sess_test"})
        if request.url.path.endswith("/events"):
            return httpx.Response(202, json={"data": [{"id": "evt_input", "type": "user.message"}]})
        body: Final = b"".join(
            (
                _sse_event(
                    {
                        "id": "evt_answer",
                        "type": "agent.message",
                        "content": [{"type": "text", "text": "done"}],
                    }
                ),
                _sse_event(
                    {
                        "id": "evt_idle",
                        "type": "session.status_idle",
                        "stop_reason": {"type": "end_turn"},
                        "total_credits": 3.5,
                    }
                ),
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        logging_obj: Final = MagicMock(dispatch_success_handlers=AsyncMock())
        response: Final = await create_qca_interaction(
            agent="agent_test",
            input="finish the task",
            environment="env_test",
            previous_interaction_id=None,
            stream=False,
            api_key="secret",
            api_base="https://qca.test/api/v1/cloud",
            timeout=30,
            unsupported_params={},
            logging_obj=logging_obj,
            client=client,
        )

    assert isinstance(response, InteractionsAPIResponse)
    assert response.id == "qca1.sess_test.evt_input"
    assert response.status == "completed"
    assert response.steps == [
        {
            "type": "message",
            "id": "evt_answer",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
        }
    ]
    assert response.usage == {
        "qca_session_credits": {
            "model_credits": None,
            "sandbox_runtime_credits": None,
            "total_credits": 3.5,
        },
        "qca_usage_scope": "session_cumulative",
        "qca_usage_is_final": True,
    }
    assert requests[-1].headers["last-event-id"] == "evt_input"


@pytest.mark.asyncio
async def test_create_qca_interaction_continues_session_and_streams_normalized_events() -> None:
    requests: list[httpx.Request] = []

    def route(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/sessions/sess_existing"):
            return httpx.Response(
                200,
                json={"id": "sess_existing", "agent": {"id": "agent_test"}, "environment_id": "env_test"},
            )
        if request.url.path.endswith("/events"):
            return httpx.Response(202, json={"data": [{"id": "evt_next", "type": "user.message"}]})
        body: Final = b"".join(
            (
                _sse_event(
                    {
                        "id": "evt_action",
                        "type": "session.status_idle",
                        "stop_reason": {"type": "requires_action"},
                    }
                ),
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        logging_obj: Final = MagicMock(dispatch_success_handlers=AsyncMock())
        result: Final = await create_qca_interaction(
            agent="agent_test",
            input="continue",
            environment="env_test",
            previous_interaction_id="qca1.sess_existing.evt_old",
            stream=True,
            api_key="secret",
            api_base="https://qca.test/api/v1/cloud",
            timeout=30,
            unsupported_params={},
            logging_obj=logging_obj,
            client=client,
        )
        assert not isinstance(result, InteractionsAPIResponse)
        chunks: Final = [chunk async for chunk in result]

    assert [chunk.event_type for chunk in chunks] == ["interaction.created", "interaction.requires_action"]
    assert chunks[-1].interaction_id == "qca1.sess_existing.evt_next"
    assert all(not request.url.path.endswith("/sessions") for request in requests)


@pytest.mark.asyncio
async def test_create_qca_interaction_rejects_unsupported_features_before_network_call() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        logging_obj: Final = MagicMock(dispatch_success_handlers=AsyncMock())
        with pytest.raises(ValueError, match="background"):
            await create_qca_interaction(
                agent="agent_test",
                input="hello",
                environment="env_test",
                previous_interaction_id=None,
                stream=False,
                api_key="secret",
                api_base=None,
                timeout=30,
                unsupported_params={"background": True},
                logging_obj=logging_obj,
                client=client,
            )


@pytest.mark.asyncio
async def test_create_qca_interaction_rejects_continuation_for_another_agent() -> None:
    requests: list[httpx.Request] = []

    def route(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": "sess_existing", "agent": {"id": "agent_other"}, "environment_id": "env_test"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        logging_obj: Final = MagicMock(dispatch_success_handlers=AsyncMock())
        with pytest.raises(ValueError, match="configured agent and environment"):
            await create_qca_interaction(
                agent="agent_test",
                input="continue",
                environment="env_test",
                previous_interaction_id="qca1.sess_existing.evt_old",
                stream=False,
                api_key="secret",
                api_base=None,
                timeout=30,
                unsupported_params={},
                logging_obj=logging_obj,
                client=client,
            )

    assert [request.method for request in requests] == ["GET"]

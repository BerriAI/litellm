import asyncio
import json
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import BadRequestError, RateLimitError
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.qca.interactions.handler import create_qca_interaction
from litellm.types.interactions import InteractionsAPIResponse, InteractionSseEvent


def _sse_event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class _SuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        self.responses: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.responses.append(response_obj)


def _stream_logging(recorder: _SuccessRecorder, include_sync_callback: bool = False) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="qca/agent_test",
        messages=[],
        stream=True,
        call_type="acreate_interaction",
        start_time=datetime.now(),
        litellm_call_id="qca-test-call",
        function_id="qca-test-function",
        dynamic_async_success_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="qca/agent_test", user="", optional_params={}, litellm_params={"api_base": "https://qca.test"}
    )
    if include_sync_callback:
        logging_obj.dynamic_success_callbacks = ["s3"]
    return logging_obj


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
        logging_obj: Final = MagicMock()
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
    assert response.created is not None
    assert response.updated is not None
    assert datetime.fromisoformat(response.updated) >= datetime.fromisoformat(response.created)
    assert response.steps == [
        {
            "type": "model_output",
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
    session_request: Final = requests[0]
    assert json.loads(session_request.content) == {
        "agent": "agent_test",
        "environment_id": "env_test",
        "metadata": {"source": "litellm"},
    }


@pytest.mark.asyncio
async def test_create_qca_interaction_continues_session_and_streams_normalized_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    sync_logger: Final = MagicMock()
    monkeypatch.setattr(litellm_logging, "s3Logger", sync_logger)

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
                        "id": "evt_answer",
                        "type": "agent.message",
                        "content": [{"type": "text", "text": "continued"}],
                    }
                ),
                _sse_event(
                    {
                        "id": "evt_action",
                        "type": "session.status_idle",
                        "stop_reason": {"type": "end_turn"},
                    }
                ),
            )
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        recorder: Final = _SuccessRecorder()
        logging_obj: Final = _stream_logging(recorder, include_sync_callback=True)
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

    assert [chunk.event_type for chunk in chunks] == [
        "interaction.created",
        "step.start",
        "step.delta",
        "step.stop",
        "interaction.completed",
    ]
    assert chunks[1].index == chunks[2].index == chunks[3].index == 0
    assert chunks[2].delta == {"type": "text", "text": "continued"}
    assert chunks[3].status == "done"
    assert chunks[-1].interaction_id == "qca1.sess_existing.evt_next"
    assert len(recorder.responses) == 1
    logged: Final = recorder.responses[0]
    assert isinstance(logged, InteractionsAPIResponse)
    assert logged.steps == [{"type": "model_output", "content": [{"type": "text", "text": "continued"}]}]
    for _ in range(200):
        if sync_logger.log_event.call_count == 1:
            break
        await asyncio.sleep(0.01)
    sync_logger.log_event.assert_called_once()
    assert all(not request.url.path.endswith("/sessions") for request in requests)


@pytest.mark.asyncio
async def test_create_qca_interaction_stream_emits_schema_valid_error_event() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json={"id": "sess_test"})
        if request.url.path.endswith("/events"):
            return httpx.Response(202, json={"data": [{"id": "evt_input", "type": "user.message"}]})
        return httpx.Response(
            200,
            content=_sse_event({"id": "evt_error", "type": "session.error"}),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        result: Final = await create_qca_interaction(
            agent="agent_test",
            input="fail",
            environment="env_test",
            previous_interaction_id=None,
            stream=True,
            api_key="secret",
            api_base="https://qca.test/api/v1/cloud",
            timeout=30,
            unsupported_params={},
            logging_obj=_stream_logging(_SuccessRecorder()),
            client=client,
        )
        assert not isinstance(result, InteractionsAPIResponse)
        chunks: Final = [chunk async for chunk in result]

    assert chunks[-1].event_type == "error"
    validated: Final = InteractionSseEvent.model_validate(chunks[-1].model_dump(exclude_none=True))
    assert validated.root.event_type == "error"


@pytest.mark.asyncio
async def test_create_qca_interaction_rejects_unsupported_features_before_network_call() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        logging_obj: Final = MagicMock()
        with pytest.raises(BadRequestError, match="background") as error:
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
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_create_qca_interaction_rejects_store_false_before_network_call() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        with pytest.raises(BadRequestError, match="store") as error:
            await create_qca_interaction(
                agent="agent_test",
                input="hello",
                environment="env_test",
                previous_interaction_id=None,
                stream=False,
                api_key="secret",
                api_base=None,
                timeout=30,
                unsupported_params={"store": False},
                logging_obj=MagicMock(),
                client=client,
            )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_create_qca_interaction_preserves_rate_limit_status() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        with pytest.raises(RateLimitError) as error:
            await create_qca_interaction(
                agent="agent_test",
                input="hello",
                environment="env_test",
                previous_interaction_id=None,
                stream=False,
                api_key="secret",
                api_base=None,
                timeout=30,
                unsupported_params={},
                logging_obj=MagicMock(),
                client=client,
            )
    assert error.value.status_code == 429


@pytest.mark.asyncio
async def test_create_qca_interaction_reads_stream_error_before_mapping_status() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json={"id": "sess_test"})
        if request.url.path.endswith("/events"):
            return httpx.Response(202, json={"data": [{"id": "evt_input", "type": "user.message"}]})
        return httpx.Response(429, text="stream rate limited")

    async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
        with pytest.raises(RateLimitError, match="stream rate limited") as error:
            await create_qca_interaction(
                agent="agent_test",
                input="hello",
                environment="env_test",
                previous_interaction_id=None,
                stream=False,
                api_key="secret",
                api_base=None,
                timeout=30,
                unsupported_params={},
                logging_obj=MagicMock(),
                client=client,
            )
    assert error.value.status_code == 429


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
        logging_obj: Final = MagicMock()
        with pytest.raises(BadRequestError, match="configured agent and environment") as error:
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
    assert error.value.status_code == 400

    assert [request.method for request in requests] == ["GET"]

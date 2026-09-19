import time

import pytest
import respx
from httpx import Response

import litellm
from litellm.interactions.sessions.http_handler import SessionInteractionsHTTPHandler
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.interactions.transformation import AnthropicSessionsInteractionsConfig
from litellm.llms.gemini.interactions.transformation import GoogleAIStudioInteractionsConfig
from litellm.types.router import GenericLiteLLMParams

SESSION_ID = "sesn_011CZkZAtmR3yMPDzynEDxu7"
SESSION_URL = f"https://api.anthropic.com/v1/sessions/{SESSION_ID}"
SESSION = {
    "id": SESSION_ID,
    "status": "idle",
    "agent": {"id": "agent_1", "model": {"id": "claude-haiku-4-5"}},
    "usage": {"input_tokens": 1, "output_tokens": 1, "list_cost": {"amount": "3", "currency": "USD"}},
}
EVENTS = {
    "data": [
        {"id": "sevt_idle", "type": "session.status_idle", "stop_reason": {"type": "end_turn"}},
        {
            "id": "sevt_usage",
            "type": "session.usage",
            "usage": {"input_tokens": 10, "output_tokens": 5, "list_cost": {"amount": "12", "currency": "USD"}},
        },
        {"id": "sevt_msg", "type": "agent.message", "content": [{"type": "text", "text": "ok"}]},
    ],
    "next_page": None,
}


@pytest.fixture(autouse=True)
def _http_transport(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


def _logging() -> Logging:
    return Logging(
        model="",
        messages=[],
        stream=False,
        call_type="aget_interaction",
        start_time=time.time(),
        litellm_call_id="call-1",
        function_id="fn-1",
    )


def _handler() -> SessionInteractionsHTTPHandler:
    return SessionInteractionsHTTPHandler()


def _params() -> GenericLiteLLMParams:
    return GenericLiteLLMParams(api_key="sk-ant-test")


@respx.mock
def test_get_reads_the_session_then_its_newest_events():
    session = respx.get(SESSION_URL).mock(return_value=Response(200, json=SESSION))
    events = respx.get(f"{SESSION_URL}/events").mock(return_value=Response(200, json=EVENTS))

    interaction = _handler().get_interaction(
        interaction_id=SESSION_ID,
        interactions_api_config=AnthropicSessionsInteractionsConfig(),
        custom_llm_provider="anthropic",
        litellm_params=_params(),
        logging_obj=_logging(),
    )

    assert interaction.status == "completed"
    assert interaction.usage["total_output_tokens"] == 5
    assert interaction._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.12
    assert interaction.steps == [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]
    assert session.calls.last.request.headers["anthropic-beta"] == "managed-agents-2026-04-01"
    assert dict(events.calls.last.request.url.params) == {"order": "desc", "limit": "100"}


@respx.mock
@pytest.mark.asyncio
async def test_async_get_reads_the_session_then_its_newest_events():
    respx.get(SESSION_URL).mock(return_value=Response(200, json=SESSION))
    events = respx.get(f"{SESSION_URL}/events").mock(return_value=Response(200, json=EVENTS))

    interaction = await _handler().async_get_interaction(
        interaction_id=SESSION_ID,
        interactions_api_config=AnthropicSessionsInteractionsConfig(),
        custom_llm_provider="anthropic",
        litellm_params=_params(),
        logging_obj=_logging(),
    )

    assert interaction.status == "completed"
    assert events.calls.last.request.url.params["order"] == "desc"


@respx.mock
def test_get_raises_the_upstream_error_for_an_unknown_session():
    respx.get(SESSION_URL).mock(return_value=Response(404, json={"type": "error"}))
    respx.get(f"{SESSION_URL}/events").mock(return_value=Response(404, json={"type": "error"}))

    with pytest.raises(AnthropicError) as excinfo:
        _handler().get_interaction(
            interaction_id=SESSION_ID,
            interactions_api_config=AnthropicSessionsInteractionsConfig(),
            custom_llm_provider="anthropic",
            litellm_params=_params(),
            logging_obj=_logging(),
        )
    assert excinfo.value.status_code == 404


@respx.mock
def test_cancel_names_the_session_it_interrupted():
    events = respx.post(f"{SESSION_URL}/events").mock(
        return_value=Response(200, json={"data": [{"id": "sevt_i", "type": "user.interrupt", "processed_at": None}]})
    )

    result = _handler().cancel_interaction(
        interaction_id=SESSION_ID,
        interactions_api_config=AnthropicSessionsInteractionsConfig(),
        custom_llm_provider="anthropic",
        litellm_params=_params(),
        logging_obj=_logging(),
    )

    assert result.id == SESSION_ID
    assert result.status == "in_progress"
    assert events.calls.last.request.content == b'{"events":[{"type":"user.interrupt"}]}'


@respx.mock
@pytest.mark.asyncio
async def test_async_cancel_names_the_session_it_interrupted():
    respx.post(f"{SESSION_URL}/events").mock(return_value=Response(200, json={"data": []}))

    result = await _handler().async_cancel_interaction(
        interaction_id=SESSION_ID,
        interactions_api_config=AnthropicSessionsInteractionsConfig(),
        custom_llm_provider="anthropic",
        litellm_params=_params(),
        logging_obj=_logging(),
    )

    assert result.id == SESSION_ID


def test_get_refuses_a_single_request_config():
    with pytest.raises(TypeError, match="session-backed"):
        _handler().get_interaction(
            interaction_id="int_1",
            interactions_api_config=GoogleAIStudioInteractionsConfig(),
            custom_llm_provider="gemini",
            litellm_params=GenericLiteLLMParams(api_key="AIza"),
            logging_obj=_logging(),
        )

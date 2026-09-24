import json

import pytest
import respx
from httpx import Response

import litellm
from litellm.interactions.background_cost_polling import is_pollable_background_interaction

SESSION_ID = "sesn_011CZkZAtmR3yMPDzynEDxu7"
AGENT_ID = "agent_011CZkYpogX7uDKUyvBTophP"
ENV_ID = "env_011CZkZ9X2dpNyB7HsEFoRfW"
SESSIONS_URL = "https://api.anthropic.com/v1/sessions"
SESSION_URL = f"{SESSIONS_URL}/{SESSION_ID}"


def _session(status: str) -> dict[str, object]:
    return {
        "type": "session",
        "id": SESSION_ID,
        "status": status,
        "agent": {"id": AGENT_ID, "model": {"id": "claude-haiku-4-5"}},
        "environment_id": ENV_ID,
        "created_at": "2026-03-15T10:00:00Z",
        "updated_at": "2026-03-15T10:00:00Z",
        "usage": {"input_tokens": 0, "output_tokens": 0, "list_cost": {"amount": "0", "currency": "USD"}},
    }


SETTLED_EVENTS = {
    "data": [
        {"id": "sevt_idle", "type": "session.status_idle", "stop_reason": {"type": "end_turn"}},
        {
            "id": "sevt_usage",
            "type": "session.usage",
            "usage": {"input_tokens": 120, "output_tokens": 9, "list_cost": {"amount": "1", "currency": "USD"}},
        },
        {"id": "sevt_msg", "type": "agent.message", "content": [{"type": "text", "text": "ok"}]},
        {"id": "sevt_run", "type": "session.status_running"},
        {"id": "sevt_user", "type": "user.message", "content": [{"type": "text", "text": "Reply with ok."}]},
    ],
    "next_page": None,
}


@pytest.fixture(autouse=True)
def _http_transport(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@respx.mock
def test_create_starts_a_session_and_hands_back_an_in_progress_interaction():
    create = respx.post(SESSIONS_URL).mock(return_value=Response(200, json=_session("running")))

    interaction = litellm.interactions.create(
        agent=AGENT_ID,
        input="Reply with ok.",
        environment=ENV_ID,
        custom_llm_provider="anthropic",
        api_key="sk-ant-test",
    )

    assert interaction.id == SESSION_ID
    assert interaction.status == "in_progress"
    assert interaction.model == "claude-haiku-4-5"
    assert is_pollable_background_interaction(interaction) is True
    sent = create.calls.last.request
    assert sent.headers["x-api-key"] == "sk-ant-test"
    assert sent.headers["anthropic-beta"] == "managed-agents-2026-04-01"
    assert json.loads(sent.content) == {
        "agent": {"type": "agent", "id": AGENT_ID},
        "environment_id": ENV_ID,
        "initial_events": [{"type": "user.message", "content": [{"type": "text", "text": "Reply with ok."}]}],
    }


@respx.mock
def test_create_with_a_model_keeps_bridging_to_the_responses_api():
    responses = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "4"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )
    )
    sessions = respx.post(SESSIONS_URL)

    interaction = litellm.interactions.create(model="anthropic/claude-haiku-4-5", input="2 + 2?", api_key="sk-ant-test")

    assert interaction.status == "completed"
    assert responses.called
    assert not sessions.called


@respx.mock
def test_get_settles_the_session_with_the_provider_price():
    respx.get(SESSION_URL).mock(return_value=Response(200, json=_session("idle")))
    respx.get(f"{SESSION_URL}/events").mock(return_value=Response(200, json=SETTLED_EVENTS))

    interaction = litellm.interactions.get(SESSION_ID, custom_llm_provider="anthropic", api_key="sk-ant-test")

    assert interaction.status == "completed"
    assert interaction.usage == {
        "total_input_tokens": 120,
        "total_cached_tokens": 0,
        "total_output_tokens": 9,
        "total_tokens": 129,
    }
    assert interaction._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.01
    assert [step["type"] for step in interaction.steps] == ["user_input", "model_output"]


@respx.mock
def test_get_maps_an_unknown_session_to_not_found():
    respx.get(SESSION_URL).mock(
        return_value=Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
    )
    respx.get(f"{SESSION_URL}/events").mock(return_value=Response(404, json={"type": "error"}))

    with pytest.raises(litellm.NotFoundError):
        litellm.interactions.get(SESSION_ID, custom_llm_provider="anthropic", api_key="sk-ant-test")


@respx.mock
def test_cancel_interrupts_the_session():
    events = respx.post(f"{SESSION_URL}/events").mock(return_value=Response(200, json={"data": []}))

    result = litellm.interactions.cancel(SESSION_ID, custom_llm_provider="anthropic", api_key="sk-ant-test")

    assert result.id == SESSION_ID
    assert result.status == "in_progress"
    assert json.loads(events.calls.last.request.content) == {"events": [{"type": "user.interrupt"}]}


@respx.mock
def test_delete_removes_the_session():
    delete = respx.delete(SESSION_URL).mock(
        return_value=Response(200, json={"id": SESSION_ID, "type": "session_deleted"})
    )

    result = litellm.interactions.delete(SESSION_ID, custom_llm_provider="anthropic", api_key="sk-ant-test")

    assert result.success is True
    assert result.id == SESSION_ID
    assert delete.called


def test_create_without_an_environment_is_refused_before_any_request():
    with pytest.raises(litellm.BadRequestError, match="environment="):
        litellm.interactions.create(agent=AGENT_ID, input="hi", custom_llm_provider="anthropic", api_key="sk-ant-test")

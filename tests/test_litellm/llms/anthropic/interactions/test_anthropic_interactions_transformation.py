import json
from collections.abc import Mapping

import httpx
import pytest

import litellm
from litellm.interactions.utils import get_provider_interactions_api_config
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.interactions.transformation import AnthropicSessionsInteractionsConfig
from litellm.llms.base_llm.interactions.session_transformation import BaseSessionInteractionsConfig
from litellm.types.router import GenericLiteLLMParams

SESSION_ID = "sesn_011CZkZAtmR3yMPDzynEDxu7"
AGENT_ID = "agent_011CZkYpogX7uDKUyvBTophP"
ENV_ID = "env_011CZkZ9X2dpNyB7HsEFoRfW"
COST_HEADER = "llm_provider-x-litellm-response-cost"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _config() -> AnthropicSessionsInteractionsConfig:
    return AnthropicSessionsInteractionsConfig()


def _params() -> GenericLiteLLMParams:
    return GenericLiteLLMParams(api_key="sk-ant-test")


def _response(payload: Mapping[str, object], status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode(),
        headers={"request-id": "req_123"},
        request=httpx.Request("GET", "https://api.anthropic.com/v1/sessions"),
    )


def _usage(**overrides: object) -> dict[str, object]:
    return {
        "input_tokens": 5000,
        "output_tokens": 3200,
        "cache_read_input_tokens": 20000,
        "cache_creation": {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 100},
        "list_cost": {"amount": "187", "currency": "USD"},
        "active_seconds": 342.5,
        **overrides,
    }


def _session(status: str = "idle", **overrides: object) -> dict[str, object]:
    return {
        "type": "session",
        "id": SESSION_ID,
        "agent": {"type": "agent", "id": AGENT_ID, "version": 1, "model": {"id": "claude-haiku-4-5"}},
        "environment_id": ENV_ID,
        "status": status,
        "created_at": "2026-03-15T10:00:00Z",
        "updated_at": "2026-03-15T10:05:00Z",
        "usage": _usage(list_cost={"amount": "0", "currency": "USD"}),
        **overrides,
    }


def _event(event_type: str, event_id: str = "sevt_x", **fields: object) -> dict[str, object]:
    return {"id": event_id, "type": event_type, "processed_at": "2026-03-15T10:00:00Z", **fields}


def _idle(stop_reason: str = "end_turn") -> dict[str, object]:
    return _event("session.status_idle", "sevt_idle", stop_reason={"type": stop_reason})


def _events(*newest_first: Mapping[str, object]) -> dict[str, object]:
    return {"data": list(newest_first), "next_page": None}


def _get(session: Mapping[str, object], events: Mapping[str, object]):
    return _config().assemble_get_response(
        session_response=_response(session), transcript_response=_response(events), logging_obj=None
    )


def _create_body(**optional: object) -> dict[str, object]:
    return _config().transform_request(
        model=None,
        agent=AGENT_ID,
        input=optional.pop("input", "Reply with the single word ok."),
        optional_params={"environment": ENV_ID, **optional},
        litellm_params=_params(),
        headers={},
    )


def test_registered_for_agent_interactions_only():
    assert isinstance(get_provider_interactions_api_config("anthropic"), AnthropicSessionsInteractionsConfig)
    assert isinstance(get_provider_interactions_api_config("anthropic"), BaseSessionInteractionsConfig)
    assert get_provider_interactions_api_config("anthropic", model="claude-haiku-4-5") is None


def test_validate_environment_sends_the_managed_agents_beta():
    headers = _config().validate_environment(headers={}, model="", litellm_params=_params())
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-beta"] == "managed-agents-2026-04-01"
    assert headers["anthropic-version"] == "2023-06-01"


def test_create_url_is_the_sessions_collection():
    assert (
        _config().get_complete_url(api_base=None, model=None, agent=AGENT_ID) == "https://api.anthropic.com/v1/sessions"
    )
    assert _config().get_complete_url(api_base="https://gw.example", model=None) == "https://gw.example/v1/sessions"


def test_create_request_starts_the_session_with_the_prompt_as_first_event():
    assert _create_body() == {
        "agent": {"type": "agent", "id": AGENT_ID},
        "environment_id": ENV_ID,
        "initial_events": [
            {"type": "user.message", "content": [{"type": "text", "text": "Reply with the single word ok."}]}
        ],
    }


def test_create_request_accepts_text_blocks_and_user_turns():
    blocks = _create_body(input=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])
    assert blocks["initial_events"][0]["content"] == [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    turns = _create_body(input=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert turns["initial_events"][0]["content"] == [{"type": "text", "text": "hello"}]


@pytest.mark.parametrize(
    ("optional", "expected"),
    [
        ({"stream": True}, "streaming"),
        ({"previous_interaction_id": SESSION_ID}, "previous_interaction_id"),
        ({"environment": {"type": "cloud"}}, "environment="),
        ({"input": [{"role": "model", "content": "no"}]}, "user text"),
    ],
)
def test_create_request_refuses_what_a_session_cannot_do(optional: Mapping[str, object], expected: str):
    with pytest.raises(litellm.BadRequestError, match=expected):
        _create_body(**optional)


def test_create_request_needs_an_agent():
    with pytest.raises(litellm.BadRequestError, match="agent="):
        _config().transform_request(
            model=None,
            agent=None,
            input="hi",
            optional_params={"environment": ENV_ID},
            litellm_params=_params(),
            headers={},
        )


def test_create_response_is_an_in_progress_interaction_without_usage():
    interaction = _config().transform_response(
        model=None, raw_response=_response(_session(status="running")), logging_obj=_Logging()
    )
    assert interaction.id == SESSION_ID
    assert interaction.status == "in_progress"
    assert interaction.agent == AGENT_ID
    assert interaction.model == "claude-haiku-4-5"
    assert interaction.usage is None
    assert interaction.created == "2026-03-15T10:00:00Z"
    assert COST_HEADER not in interaction._hidden_params.get("additional_headers", {})


def test_create_response_maps_upstream_errors():
    with pytest.raises(AnthropicError) as excinfo:
        _config().transform_response(
            model=None, raw_response=_response({"type": "error"}, status_code=403), logging_obj=_Logging()
        )
    assert excinfo.value.status_code == 403


@pytest.mark.parametrize(
    ("session_status", "events", "expected"),
    [
        ("running", _events(), "in_progress"),
        ("rescheduling", _events(), "in_progress"),
        ("terminated", _events(_event("session.status_terminated")), "failed"),
        ("idle", _events(), "completed"),
        ("idle", _events(_idle("end_turn"), _event("agent.message")), "completed"),
        ("idle", _events(_idle("requires_action"), _event("agent.custom_tool_use")), "requires_action"),
        ("idle", _events(_idle("budget_reached")), "budget_exceeded"),
        ("idle", _events(_idle("retries_exhausted")), "failed"),
        ("idle", _events(_event("user.message", "sevt_new"), _idle("end_turn")), "in_progress"),
        (
            "idle",
            _events(_event("session.error", error={"retry_status": {"type": "exhausted"}}), _idle("end_turn")),
            "failed",
        ),
        (
            "idle",
            _events(_idle("end_turn"), _event("session.error", error={"retry_status": {"type": "retrying"}})),
            "completed",
        ),
        (
            "idle",
            _events(_idle("end_turn"), _event("session.error", error={"retry_status": {"type": "terminal"}})),
            "completed",
        ),
    ],
)
def test_get_status_follows_the_session_and_its_newest_events(
    session_status: str, events: Mapping[str, object], expected: str
):
    assert _get(_session(status=session_status), events).status == expected


def test_get_takes_usage_and_price_from_the_newest_session_usage_event():
    older = _event("session.usage", "sevt_u1", usage=_usage(list_cost={"amount": "40", "currency": "USD"}))
    newest = _event("session.usage", "sevt_u2", usage=_usage())
    interaction = _get(_session(), _events(_idle(), newest, older))

    assert interaction.usage == {
        "total_input_tokens": 27100,
        "total_cached_tokens": 20000,
        "total_output_tokens": 3200,
        "total_tokens": 30300,
    }
    assert interaction._hidden_params["additional_headers"][COST_HEADER] == 1.87


def test_get_falls_back_to_the_session_object_usage():
    interaction = _get(_session(usage=_usage(list_cost={"amount": "42", "currency": "USD"})), _events(_idle()))
    assert interaction.usage["total_output_tokens"] == 3200
    assert interaction._hidden_params["additional_headers"][COST_HEADER] == 0.42


def test_get_reports_no_price_without_a_usd_list_cost():
    foreign = _get(_session(usage=_usage(list_cost={"amount": "42", "currency": "EUR"})), _events(_idle()))
    assert COST_HEADER not in foreign._hidden_params.get("additional_headers", {})
    missing = _get(_session(usage=None), _events(_idle()))
    assert missing.usage is None
    assert COST_HEADER not in missing._hidden_params.get("additional_headers", {})


def test_get_keeps_raw_anthropic_token_names_out_of_usage():
    interaction = _get(_session(), _events(_idle(), _event("session.usage", usage=_usage())))
    assert "input_tokens" not in interaction.usage
    assert "list_cost" not in interaction.usage


def test_get_steps_are_chronological_and_speak_the_interactions_vocabulary():
    newest_first = _events(
        _idle("requires_action"),
        _event("agent.custom_tool_use", "sevt_call", name="lookup_ticket", input={"id": "42"}),
        _event("agent.tool_result", "sevt_r", tool_use_id="sevt_bash", content=[{"type": "text", "text": "ok\n"}]),
        _event("agent.tool_use", "sevt_bash", name="bash", input={"command": "ls"}),
        _event("agent.tool_use", "sevt_read", name="read", input={"path": "/x"}),
        _event(
            "agent.mcp_tool_result", "sevt_mr", mcp_tool_use_id="sevt_mcp", content=[{"type": "text", "text": "{}"}]
        ),
        _event("agent.mcp_tool_use", "sevt_mcp", name="search", mcp_server_name="docs", input={"q": "a"}),
        _event(
            "agent.message", "sevt_m", content=[{"type": "text", "text": "Looking"}, {"type": "text", "text": " up"}]
        ),
        _event("session.status_running"),
        _event("user.message", "sevt_user", content=[{"type": "text", "text": "Where is order 42?"}]),
    )
    steps = _get(_session(), newest_first).steps
    assert steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Where is order 42?"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Looking up"}]},
        {
            "type": "model_output",
            "content": [
                {
                    "type": "mcp_server_tool_call",
                    "id": "sevt_mcp",
                    "name": "search",
                    "server_name": "docs",
                    "arguments": {"q": "a"},
                }
            ],
        },
        {
            "type": "model_output",
            "content": [{"type": "mcp_server_tool_result", "call_id": "sevt_mcp", "result": "{}"}],
        },
        {
            "type": "model_output",
            "content": [{"type": "code_execution_call", "id": "sevt_bash", "arguments": {"code": "ls"}}],
        },
        {
            "type": "model_output",
            "content": [{"type": "code_execution_result", "call_id": "sevt_bash", "result": "ok\n", "is_error": False}],
        },
        {
            "type": "model_output",
            "content": [
                {"type": "function_call", "id": "sevt_call", "name": "lookup_ticket", "arguments": {"id": "42"}}
            ],
        },
    ]


def test_get_steps_carry_the_answer_to_a_custom_tool():
    steps = _get(
        _session(),
        _events(
            _idle(),
            _event(
                "user.custom_tool_result",
                "sevt_ans",
                custom_tool_use_id="sevt_call",
                content=[{"type": "text", "text": "42"}],
                is_error=True,
            ),
        ),
    ).steps
    assert steps == [
        {
            "type": "user_input",
            "content": [{"type": "function_result", "call_id": "sevt_call", "result": "42", "is_error": True}],
        }
    ]


def test_get_refuses_a_transcript_that_is_not_a_page():
    with pytest.raises(AnthropicError, match="events schema"):
        _get(_session(), {"data": "nope"})


def test_get_maps_a_missing_session_to_the_upstream_status():
    with pytest.raises(AnthropicError) as excinfo:
        _config().assemble_get_response(
            session_response=_response({"type": "error"}, status_code=404),
            transcript_response=_response(_events()),
            logging_obj=None,
        )
    assert excinfo.value.status_code == 404


def test_get_requests_address_the_session_and_its_newest_events():
    url, params = _config().transform_get_interaction_request(
        interaction_id=SESSION_ID, api_base="", litellm_params=_params(), headers={}
    )
    assert url == f"https://api.anthropic.com/v1/sessions/{SESSION_ID}"
    assert params == {}
    transcript_url, transcript_params = _config().transform_get_transcript_request(
        interaction_id=SESSION_ID, api_base="https://gw.example", litellm_params=_params()
    )
    assert transcript_url == f"https://gw.example/v1/sessions/{SESSION_ID}/events"
    assert dict(transcript_params) == {"order": "desc", "limit": 100}


def test_get_interaction_response_without_a_transcript_reads_the_session_object():
    interaction = _config().transform_get_interaction_response(
        raw_response=_response(_session(status="running")), logging_obj=None
    )
    assert interaction.status == "in_progress"
    assert interaction.usage["total_output_tokens"] == 3200


def test_cancel_sends_a_user_interrupt_and_leaves_the_session_running():
    url, body = _config().transform_cancel_interaction_request(
        interaction_id=SESSION_ID, api_base="", litellm_params=_params(), headers={}
    )
    assert url == f"https://api.anthropic.com/v1/sessions/{SESSION_ID}/events"
    assert body == {"events": [{"type": "user.interrupt"}]}
    result = _config().transform_cancel_interaction_response(
        raw_response=_response({"data": [{"id": "sevt_i", "type": "user.interrupt", "processed_at": None}]}),
        logging_obj=None,
    )
    assert result.status == "in_progress"
    assert result.id is None


def test_delete_addresses_the_session_and_reports_success():
    url, _ = _config().transform_delete_interaction_request(
        interaction_id=SESSION_ID, api_base="", litellm_params=_params(), headers={}
    )
    assert url == f"https://api.anthropic.com/v1/sessions/{SESSION_ID}"
    result = _config().transform_delete_interaction_response(
        raw_response=_response({"id": SESSION_ID, "type": "session_deleted"}),
        logging_obj=None,
        interaction_id=SESSION_ID,
    )
    assert result.success is True
    assert result.id == SESSION_ID
    with pytest.raises(AnthropicError) as excinfo:
        _config().transform_delete_interaction_response(
            raw_response=_response({"type": "error"}, status_code=409), logging_obj=None, interaction_id=SESSION_ID
        )
    assert excinfo.value.status_code == 409


def test_streaming_chunks_are_refused():
    with pytest.raises(litellm.BadRequestError, match="streaming"):
        _config().transform_streaming_response(model=None, parsed_chunk={}, logging_obj=None)


class _Logging:
    def __init__(self):
        self.calls: list[str] = []

    def post_call(self, original_response: str, input=None, api_key=None, additional_args=None):
        self.calls.append(original_response)

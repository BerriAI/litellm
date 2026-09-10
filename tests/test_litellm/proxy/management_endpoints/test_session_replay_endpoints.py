import json

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.session_replay_endpoints import (
    _job_response,
    _load_source_request,
    _require_admin_viewer,
    _require_admin_writer,
    _SourceRequest,
    _SourceUnavailable,
)

RECORDED = {
    "model": "router",
    "max_tokens": 1024,
    "system": [{"type": "text", "text": "sys"}],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
}


class _FakePrisma:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    class _Db:
        def __init__(self, outer):
            self.outer = outer

        async def query_raw(self, query, *args):
            self.outer.queries.append((query, args))
            return self.outer.rows

    @property
    def db(self):
        return self._Db(self)


@pytest.mark.asyncio
async def test_session_with_no_replayable_rows_is_a_404():
    result = await _load_source_request(_FakePrisma([]), "missing-session")

    assert isinstance(result, _SourceUnavailable)
    assert result.status_code == 404


@pytest.mark.asyncio
async def test_session_whose_body_was_not_stored_explains_the_config_gate():
    """store_prompts_in_spend_logs off writes an empty object, so the row exists but holds
    nothing to replay: that must not read as a missing session."""
    result = await _load_source_request(
        _FakePrisma([{"request_id": "r1", "proxy_server_request": {}, "prompt_tokens": 10, "metadata": {}}]),
        "sess",
    )

    assert isinstance(result, _SourceUnavailable)
    assert result.status_code == 409
    assert "store_prompts_in_spend_logs" in result.reason


@pytest.mark.asyncio
async def test_session_rewritten_by_a_mutating_guardrail_is_refused():
    """The stored body predates the guardrail's rewrite, so replaying it would resurrect
    content the guardrail stripped."""
    rows = [
        {
            "request_id": "r1",
            "proxy_server_request": RECORDED,
            "prompt_tokens": 10,
            "metadata": {
                "standard_logging_guardrail_information": [{"guardrail_mode": "pre_call"}],
            },
        }
    ]

    result = await _load_source_request(_FakePrisma(rows), "sess")

    assert isinstance(result, _SourceUnavailable)
    assert result.status_code == 409
    assert "guardrail" in result.reason


@pytest.mark.asyncio
async def test_replayable_session_returns_the_parsed_body_and_recorded_token_count():
    rows = [
        {
            "request_id": "req-42",
            "proxy_server_request": RECORDED,
            "prompt_tokens": 41920,
            "metadata": {},
        }
    ]

    result = await _load_source_request(_FakePrisma(rows), "sess")

    assert isinstance(result, _SourceRequest)
    assert result.request_id == "req-42"
    assert result.recorded_prompt_tokens == 41920
    assert len(result.body.messages) == 1


@pytest.mark.asyncio
async def test_source_lookup_is_scoped_to_replayable_call_types():
    """A chat-completions row cannot be replayed through the Anthropic Messages path, so
    picking one up would fail at the first turn instead of at start."""
    prisma = _FakePrisma([])

    await _load_source_request(prisma, "sess")

    _, args = prisma.queries[0]
    assert args == ("sess", "anthropic_messages")


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.TEAM,
        None,
    ],
)
def test_only_a_proxy_admin_can_start_a_replay(role):
    """A replay reads another key's recorded prompts and spends real money on them."""
    with pytest.raises(HTTPException) as excinfo:
        _require_admin_writer(UserAPIKeyAuth(api_key="k", user_role=role), "start a session replay")

    assert excinfo.value.status_code == 403


def test_proxy_admin_may_start_a_replay():
    admin = UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert _require_admin_writer(admin, "start") is None


@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
def test_admin_viewers_may_read_a_replay(role):
    assert _require_admin_viewer(UserAPIKeyAuth(api_key="k", user_role=role), "read") is None


def test_internal_user_cannot_read_a_replay():
    with pytest.raises(HTTPException) as excinfo:
        _require_admin_viewer(UserAPIKeyAuth(api_key="k", user_role=LitellmUserRoles.INTERNAL_USER), "read")

    assert excinfo.value.status_code == 403


def test_finished_job_row_surfaces_verdict_and_fidelity():
    row = {
        "id": "job-1",
        "session_id": "sess",
        "status": "completed",
        "judge_model": "judge",
        "max_turns": 20,
        "source_request_id": "req-42",
        "turns_completed": 3,
        "created_by": "admin",
        "result": {
            "human_asks": ["plan my week"],
            "arms": [
                {"label": "auto_router", "model": "r", "final_text": "a", "turns": []},
                {"label": "baseline", "model": "b", "final_text": "c", "turns": []},
            ],
            "verdict": {"winner": "auto_router", "confidence": 0.72, "reasoning": "more complete"},
            "fidelity": {
                "truncated_strings": 6,
                "recorded_prompt_tokens": 41920,
                "replayed_first_turn_prompt_tokens": 16016,
            },
        },
    }

    response = _job_response(row)

    assert response.status == "completed"
    assert response.verdict is not None and response.verdict.winner == "auto_router"
    assert response.fidelity is not None
    assert response.fidelity.prompt_retained_ratio == pytest.approx(16016 / 41920)
    assert tuple(arm.label for arm in response.arms) == ("auto_router", "baseline")


def test_running_job_row_has_no_verdict_yet():
    response = _job_response(
        {"id": "job-2", "session_id": "s", "status": "running", "judge_model": "j", "max_turns": 5}
    )

    assert response.status == "running"
    assert response.verdict is None
    assert response.arms == ()


@pytest.mark.asyncio
async def test_jsonb_columns_returned_as_text_are_still_parsed():
    """query_raw hands a jsonb column back as a dict or as raw JSON text depending on the
    driver, and treating the text form as absent would report every session unreplayable."""
    rows = [
        {
            "request_id": "req-1",
            "proxy_server_request": json.dumps(RECORDED),
            "prompt_tokens": 100,
            "metadata": "{}",
        }
    ]

    result = await _load_source_request(_FakePrisma(rows), "sess")

    assert isinstance(result, _SourceRequest)
    assert len(result.body.messages) == 1

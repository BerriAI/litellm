import json
import uuid
from contextlib import ExitStack
from datetime import datetime
from hashlib import sha256
from typing import Final

from integration._support.client import Gateway, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROMPT_PENDING: Final = "pending submission must not run"
_PROMPT_APPROVED: Final = "approved submission runs"
_PROMPT_REJECTED: Final = "rejected submission must not run"
_APPLY_TEXT: Final = "apply guardrail after approval"
_PROVIDER_RESPONSE: Final = {
    "id": "chatcmpl-guardrail-submission",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "provider response"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _RegisterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guardrail_id: str
    guardrail_name: str
    status: str
    submitted_at: datetime


class _GuardrailRow(BaseModel):
    guardrail_id: str
    status: str
    team_id: str


class _ProxyError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None
    code: str


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: _ProxyError


class _AdminActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guardrail_id: str
    status: str
    message: str


class _ApplyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_text: str


class _ChatMessage(BaseModel):
    role: str
    content: str | None


class _ChatChoice(BaseModel):
    index: int
    message: _ChatMessage
    finish_reason: str | None


class _ChatResponse(BaseModel):
    choices: list[_ChatChoice]


class _SubmissionItem(BaseModel):
    guardrail_id: str


class _SubmissionsResponse(BaseModel):
    submissions: list[_SubmissionItem]
    summary: dict[str, int]


def _delete_guardrail(gateway: Gateway, guardrail_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{guardrail_id}")
    assert deleted.status_code == 200, deleted.text
    assert read_rows('SELECT guardrail_id FROM "LiteLLM_GuardrailsTable" WHERE guardrail_id=%s', (guardrail_id,)) == []


def _assert_submission_status(guardrail_id: str, status: str, team_id: str) -> None:
    rows: Final = read_rows(
        'SELECT guardrail_id, status, team_id FROM "LiteLLM_GuardrailsTable" WHERE guardrail_id=%s',
        (guardrail_id,),
    )
    assert len(rows) == 1, rows
    assert _GuardrailRow.model_validate(rows[0]) == _GuardrailRow(
        guardrail_id=guardrail_id, status=status, team_id=team_id
    ), rows


def _register(
    gateway: Gateway,
    scenario_cleanups: ExitStack,
    name: str,
    team_id: str | None,
    key: str,
    api_base: str,
) -> str:
    params: Final[dict[str, JsonValue]] = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": False,
        "api_base": api_base,
        "api_key": "synthetic-submission-guardrail-key",
    }
    request_body: Final[dict[str, JsonValue]] = {"guardrail_name": name, "litellm_params": params}
    complete_body: Final[dict[str, JsonValue]] = {
        **request_body,
        **({"team_id": team_id} if team_id is not None else {}),
    }
    response: Final = gateway.request("POST", "/guardrails/register", complete_body, key=key)
    assert response.status_code == 200, response.text
    created: Final = _RegisterResponse.model_validate_json(response.content)
    assert (created.guardrail_name, created.status) == (name, "pending_review"), response.text
    scenario_cleanups.callback(_delete_guardrail, gateway, created.guardrail_id)
    return created.guardrail_id


def _chat(gateway: Gateway, model: str, key: str, text: str, guardrail: str) -> _ChatResponse:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}], "guardrails": [guardrail]},
        key=key,
    )
    assert response.status_code == 200, response.text
    parsed: Final = _ChatResponse.model_validate_json(response.content)
    assert parsed.choices[0].message.content == "provider response", response.text
    return parsed


def _assert_provider_request(request: Request, text: str) -> None:
    assert (request.method, request.target) == ("POST", "/v1/chat/completions")
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    assert payload == {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]}, payload


def _assert_guardrail_request(
    request: Request,
    text: str,
    model: str | None,
    key: str,
    *,
    key_alias: str | None = None,
) -> None:
    assert (request.method, request.target) == ("POST", "/beta/litellm_basic_guardrail_api")
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    assert set(payload) == {
        "input_type",
        "litellm_call_id",
        "litellm_trace_id",
        "structured_messages",
        "images",
        "tools",
        "texts",
        "request_data",
        "request_headers",
        "litellm_version",
        "additional_provider_specific_params",
        "tool_calls",
        "model",
    }, payload
    assert {
        field: payload[field]
        for field in (
            "input_type",
            "structured_messages",
            "images",
            "tools",
            "texts",
            "additional_provider_specific_params",
            "tool_calls",
            "model",
        )
    } == {
        "input_type": "request",
        "structured_messages": [{"role": "user", "content": text}] if model is not None else None,
        "images": None,
        "tools": None,
        "texts": [text],
        "additional_provider_specific_params": {},
        "tool_calls": None,
        "model": model,
    }, payload
    assert payload["litellm_call_id"] is None or isinstance(payload["litellm_call_id"], str), payload
    assert payload["litellm_trace_id"] is None or isinstance(payload["litellm_trace_id"], str), payload
    assert payload["litellm_version"] is None or isinstance(payload["litellm_version"], str), payload
    request_data: Final = object_value(payload["request_data"])
    if model is None:
        assert request_data == {}, payload
    else:
        assert set(request_data).issubset(
            {
                "user_api_key_hash",
                "user_api_key_alias",
                "user_api_key_user_id",
                "user_api_key_user_email",
                "user_api_key_team_id",
                "user_api_key_team_alias",
                "user_api_key_end_user_id",
                "user_api_key_org_id",
            }
        ), request_data
        assert request_data["user_api_key_hash"] == sha256(key.encode()).hexdigest(), request_data
        assert request_data.get("user_api_key_alias") == key_alias, request_data
        assert all(value is None or isinstance(value, str) for value in request_data.values()), request_data
    request_headers: Final = payload["request_headers"]
    assert request_headers is None or isinstance(request_headers, dict), payload
    if isinstance(request_headers, dict):
        assert "authorization" not in request_headers
    assert request.headers["x-api-key"] == "synthetic-submission-guardrail-key"


def _masked_user_id(user_id: str) -> str:
    return f"{user_id[:6]}{'*' * (len(user_id) - 8)}{user_id[-2:]}"


def test_non_admins_are_refused_from_guardrail_submission_actions(gateway: Gateway) -> None:
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(lambda _request: Reply(body=b'{"action":"NONE"}')) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        team_id: Final = scenario.team(models=[model])
        member: Final = scenario.member(team_id)
        team_admin: Final = scenario.member(team_id, role="admin")
        member_key: Final = scenario.key(user_id=member, team_id=team_id, models=[model])
        team_admin_key: Final = scenario.key(user_id=team_admin, team_id=team_id, models=[model])
        name: Final = f"submitted-auth-{uuid.uuid4().hex}"
        guardrail_id: Final = _register(
            gateway,
            scenario.cleanups,
            name,
            team_id,
            member_key,
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
        )
        responses: Final = (
            gateway.request("POST", f"/guardrails/submissions/{guardrail_id}/approve", key=team_admin_key),
            gateway.request("POST", f"/guardrails/submissions/{guardrail_id}/reject", key=team_admin_key),
            gateway.request("POST", f"/guardrails/submissions/{guardrail_id}/approve", key=member_key),
            gateway.request("POST", f"/guardrails/submissions/{guardrail_id}/reject", key=member_key),
        )
        expected_errors: Final = tuple(
            {
                "error": {
                    "message": (
                        "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
                        f"keys/users/teams. Route=/guardrails/submissions/{guardrail_id}/{action}. "
                        f"Your role=internal_user. Your user_id={_masked_user_id(user_id)}"
                    ),
                    "type": "auth_error",
                    "param": "None",
                    "code": "401",
                }
            }
            for action, user_id in (
                ("approve", team_admin),
                ("reject", team_admin),
                ("approve", member),
                ("reject", member),
            )
        )
        assert tuple(response.status_code for response in responses) == (401, 401, 401, 401), tuple(
            response.text for response in responses
        )
        assert tuple(
            (response.status_code, _ErrorResponse.model_validate_json(response.content).model_dump())
            for response in responses
        ) == tuple((401, expected_error) for expected_error in expected_errors), tuple(
            response.text for response in responses
        )
        _assert_submission_status(guardrail_id, "pending_review", team_id)
        assert guardrail.drain() == ()


def test_team_guardrail_submissions_require_admin_approval(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(lambda _request: Reply(body=b'{"action":"NONE"}')) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        team_id: Final = scenario.team(models=[model])
        member: Final = scenario.member(team_id)
        team_admin: Final = scenario.member(team_id, role="admin")
        member_key_alias: Final = f"submission-member-{uuid.uuid4().hex}"
        member_key: Final = scenario.key(
            user_id=member,
            team_id=team_id,
            models=[model],
            key_alias=member_key_alias,
        )
        team_admin_key: Final = scenario.key(
            user_id=team_admin,
            team_id=team_id,
            models=[model],
            key_alias=f"submission-team-admin-{uuid.uuid4().hex}",
        )
        other_team_id: Final = scenario.team(models=[model])
        api_base: Final = f"{guardrail.url}/beta/litellm_basic_guardrail_api"
        first_name: Final = f"submitted-member-{uuid.uuid4().hex}"
        second_name: Final = f"submitted-admin-{uuid.uuid4().hex}"
        other_name: Final = f"submitted-other-team-{uuid.uuid4().hex}"
        first_id: Final = _register(gateway, scenario.cleanups, first_name, team_id, member_key, api_base)
        second_id: Final = _register(gateway, scenario.cleanups, second_name, None, team_admin_key, api_base)
        other_id: Final = _register(gateway, scenario.cleanups, other_name, other_team_id, gateway.key, api_base)

        _assert_submission_status(first_id, "pending_review", team_id)
        _assert_submission_status(second_id, "pending_review", team_id)
        _assert_submission_status(other_id, "pending_review", other_team_id)

        pending_apply: Final = gateway.request(
            "POST", "/apply_guardrail", {"guardrail_name": first_name, "text": _APPLY_TEXT}
        )
        assert pending_apply.status_code == 404, pending_apply.text
        assert _ErrorResponse.model_validate_json(pending_apply.content).model_dump() == {
            "error": {
                "message": (
                    f"Guardrail '{first_name}' not found. "
                    "Please ensure the guardrail is configured in your LiteLLM proxy."
                ),
                "type": "internal_server_error",
                "param": None,
                "code": "404",
            }
        }, pending_apply.text

        pending_chat: Final = _chat(gateway, model, member_key, _PROMPT_PENDING, first_name)
        assert len(pending_chat.choices) == 1
        assert guardrail.drain() == ()

        _assert_submission_status(first_id, "pending_review", team_id)

        approved: Final = gateway.request("POST", f"/guardrails/submissions/{first_id}/approve")
        assert approved.status_code == 200, approved.text
        assert _AdminActionResponse.model_validate_json(approved.content).model_dump() == {
            "guardrail_id": first_id,
            "status": "active",
            "message": "Guardrail approved",
        }, approved.text
        _assert_submission_status(first_id, "active", team_id)

        approved_chat: Final = _chat(gateway, model, member_key, _PROMPT_APPROVED, first_name)
        assert len(approved_chat.choices) == 1
        active_apply: Final = gateway.request(
            "POST", "/apply_guardrail", {"guardrail_name": first_name, "text": _APPLY_TEXT}
        )
        assert active_apply.status_code == 200, active_apply.text
        assert _ApplyResponse.model_validate_json(active_apply.content).response_text == _APPLY_TEXT, active_apply.text

        rejected: Final = gateway.request("POST", f"/guardrails/submissions/{second_id}/reject")
        assert rejected.status_code == 200, rejected.text
        assert _AdminActionResponse.model_validate_json(rejected.content).model_dump() == {
            "guardrail_id": second_id,
            "status": "rejected",
            "message": "Guardrail rejected",
        }, rejected.text
        _assert_submission_status(second_id, "rejected", team_id)
        rejected_chat: Final = _chat(gateway, model, member_key, _PROMPT_REJECTED, second_name)
        assert len(rejected_chat.choices) == 1
        rejected_apply: Final = gateway.request(
            "POST", "/apply_guardrail", {"guardrail_name": second_name, "text": _APPLY_TEXT}
        )
        assert rejected_apply.status_code == 404, rejected_apply.text
        assert _ErrorResponse.model_validate_json(rejected_apply.content).model_dump() == {
            "error": {
                "message": (
                    f"Guardrail '{second_name}' not found. "
                    "Please ensure the guardrail is configured in your LiteLLM proxy."
                ),
                "type": "internal_server_error",
                "param": None,
                "code": "404",
            }
        }, rejected_apply.text

        team_list: Final = gateway.request("GET", "/guardrails/submissions", key=member_key)
        assert team_list.status_code == 200, team_list.text
        team_submissions: Final = _SubmissionsResponse.model_validate_json(team_list.content)
        assert {item.guardrail_id for item in team_submissions.submissions} == {first_id, second_id}, team_list.text
        admin_list: Final = gateway.request("GET", "/guardrails/submissions")
        assert admin_list.status_code == 200, admin_list.text
        admin_submissions: Final = _SubmissionsResponse.model_validate_json(admin_list.content)
        assert {item.guardrail_id for item in admin_submissions.submissions} == {
            first_id,
            second_id,
            other_id,
        }, admin_list.text

        provider_requests: Final = provider.drain()
        assert [(request.method, request.target) for request in provider_requests] == [
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions"),
        ]
        for request, text in zip(provider_requests, (_PROMPT_PENDING, _PROMPT_APPROVED, _PROMPT_REJECTED), strict=True):
            _assert_provider_request(request, text)
        guardrail_requests: Final = guardrail.drain()
        assert [(request.method, request.target) for request in guardrail_requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api"),
            ("POST", "/beta/litellm_basic_guardrail_api"),
        ]
        _assert_guardrail_request(
            guardrail_requests[0],
            _PROMPT_APPROVED,
            model,
            member_key,
            key_alias=member_key_alias,
        )
        _assert_guardrail_request(guardrail_requests[1], _APPLY_TEXT, None, gateway.key)

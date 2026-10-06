import json
import uuid
from contextlib import ExitStack
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MASKED: Final = "masked integration prompt"
_MASKED_FIRST: Final = "masked original policy prompt"
_MASKED_SECOND: Final = "masked published policy prompt"
_PROMPT: Final = "integration prompt containing a secret"
_PROVIDER_RESPONSE: Final = {
    "id": "chatcmpl-policy-attachment",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "provider response"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _GuardrailCreated(BaseModel):
    guardrail_id: str
    guardrail_name: str


class _PolicyResponse(BaseModel):
    policy_id: str
    policy_name: str
    version_number: int
    version_status: str
    guardrails_add: list[str]
    guardrails_remove: list[str]


class _PolicyVersionsResponse(BaseModel):
    policy_name: str
    versions: list[_PolicyResponse]
    total_count: int


class _AttachmentResponse(BaseModel):
    attachment_id: str
    policy_name: str
    scope: str | None
    teams: list[str]
    keys: list[str]
    models: list[str]
    tags: list[str]
    priority: int | None
    default: bool


class _ChatMessage(BaseModel):
    role: str
    content: str | None


class _ChatChoice(BaseModel):
    index: int
    message: _ChatMessage
    finish_reason: str | None


class _ChatResponse(BaseModel):
    choices: list[_ChatChoice]


def _delete_guardrail(gateway: Gateway, guardrail_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{guardrail_id}")
    assert deleted.status_code == 200, deleted.text
    assert read_rows('SELECT guardrail_id FROM "LiteLLM_GuardrailsTable" WHERE guardrail_id=%s', (guardrail_id,)) == []


def _delete_policy(gateway: Gateway, policy_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/policies/{policy_id}")
    assert deleted.status_code == 200, deleted.text
    assert read_rows('SELECT policy_id FROM "LiteLLM_PolicyTable" WHERE policy_id=%s', (policy_id,)) == []


def _create_guardrail(
    gateway: Gateway,
    cleanups: ExitStack,
    guardrail_name: str,
    api_base: str,
    api_key: str,
) -> str:
    created: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": guardrail_name,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": False,
                    "api_base": api_base,
                    "api_key": api_key,
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    payload: Final = _GuardrailCreated.model_validate_json(created.content)
    cleanups.callback(_delete_guardrail, gateway, payload.guardrail_id)
    return payload.guardrail_name


def _create_policy(gateway: Gateway, cleanups: ExitStack, policy_name: str, guardrail_name: str) -> str:
    created: Final = gateway.request(
        "POST",
        "/policies",
        {"policy_name": policy_name, "guardrails_add": [guardrail_name]},
    )
    assert created.status_code == 200, created.text
    payload: Final = _PolicyResponse.model_validate_json(created.content)
    cleanups.callback(_delete_policy, gateway, payload.policy_id)
    assert (payload.policy_name, payload.version_status, payload.guardrails_add) == (
        policy_name,
        "production",
        [guardrail_name],
    ), created.text
    return payload.policy_id


def _create_published_version(
    gateway: Gateway,
    cleanups: ExitStack,
    policy_name: str,
    source_policy_id: str,
    guardrail_name: str,
) -> str:
    create_version: Final = gateway.request(
        "POST", f"/policies/name/{policy_name}/versions", {"source_policy_id": source_policy_id}
    )
    assert create_version.status_code == 200, create_version.text
    draft: Final = _PolicyResponse.model_validate_json(create_version.content)
    cleanups.callback(_delete_policy, gateway, draft.policy_id)
    update_version: Final = gateway.request(
        "PUT",
        f"/policies/{draft.policy_id}",
        {"guardrails_add": [guardrail_name], "guardrails_remove": []},
    )
    assert update_version.status_code == 200, update_version.text
    assert _PolicyResponse.model_validate_json(update_version.content).guardrails_add == [guardrail_name], (
        update_version.text
    )
    publish_version: Final = gateway.request(
        "PUT", f"/policies/{draft.policy_id}/status", {"version_status": "published"}
    )
    assert publish_version.status_code == 200, publish_version.text
    assert _PolicyResponse.model_validate_json(publish_version.content).version_status == "published", (
        publish_version.text
    )
    versions_response: Final = gateway.request("GET", f"/policies/name/{policy_name}/versions")
    assert versions_response.status_code == 200, versions_response.text
    versions: Final = _PolicyVersionsResponse.model_validate_json(versions_response.content)
    assert versions.policy_name == policy_name, versions_response.text
    second_version: Final = next(version for version in versions.versions if version.policy_id == draft.policy_id)
    assert second_version.guardrails_add == [guardrail_name], versions_response.text
    assert second_version.version_status == "published", versions_response.text
    return second_version.policy_id


def _assert_guardrail_request(
    request: Request,
    text: str,
    model: str,
    key: str,
    *,
    key_alias: str | None = None,
    response: httpx.Response | None = None,
) -> None:
    assert request.method == "POST"
    assert request.target == "/beta/litellm_basic_guardrail_api"
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
        key: payload[key]
        for key in (
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
        "structured_messages": [{"role": "user", "content": text}],
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
    if response is not None and response.headers.get("x-litellm-call-id") is not None:
        assert payload["litellm_call_id"] == response.headers["x-litellm-call-id"], payload
    request_data: Final = object_value(payload["request_data"])
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


def _assert_provider_request(request: Request, text: str) -> None:
    assert request.method == "POST"
    assert request.target == "/v1/chat/completions"
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    assert payload == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": text}],
    }, payload


def _mask_policy_prompt(request: Request, mask: str) -> str:
    payload: Final = _JSON_OBJECT.validate_json(request.body)
    texts: Final = payload["texts"]
    assert isinstance(texts, list) and len(texts) == 1, payload
    return f"{mask}: {string_value(texts[0])}"


def _chat(
    gateway: Gateway,
    model: str,
    key: str,
    text: str,
    *,
    policies: list[str] | None = None,
) -> httpx.Response:
    body: Final = _JSON_OBJECT.validate_python(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            **({} if policies is None else {"policies": policies}),
        }
    )
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        body,
        key=key,
    )
    assert response.status_code == 200, response.text
    _ChatResponse.model_validate_json(response.content)
    return response


def _attachment_scope(
    spelling: str,
    policy_name: str,
    matching_model: str,
    prefix: str,
) -> dict[str, JsonValue]:
    return {
        "policy_name": policy_name,
        "models": [matching_model] if spelling == "models" else [],
        "teams": [f"{prefix}-*"] if spelling == "teams" else [],
        "keys": [f"{prefix}-*"] if spelling == "keys" else [],
        "scope": "*" if spelling == "*" else None,
    }


def _matching_and_nonmatching_keys(
    scenario: Scenario,
    spelling: str,
    prefix: str,
    model: str,
    second_model: str,
) -> tuple[str, str | None]:
    if spelling == "teams":
        matching_team: Final = scenario.team(team_alias=f"{prefix}-eng", models=[model])
        nonmatching_team: Final = scenario.team(team_alias=f"other-{prefix}-team", models=[model])
        return (
            scenario.key(team_id=matching_team, models=[model], key_alias=f"{prefix}-key"),
            scenario.key(team_id=nonmatching_team, models=[model], key_alias=f"other-{prefix}-key"),
        )
    if spelling == "keys":
        team: Final = scenario.team(models=[model])
        return (
            scenario.key(team_id=team, models=[model], key_alias=f"{prefix}-key"),
            scenario.key(team_id=team, models=[model], key_alias=f"other-{prefix}-key"),
        )
    return scenario.key(models=[model, second_model]), None


@pytest.mark.parametrize("spelling", ("models", "teams", "keys", "*"))
def test_policy_attachments_match_live_models_teams_keys_and_global_scope(gateway: Gateway, spelling: str) -> None:
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(
            lambda _request: Reply(body=b'{"action":"GUARDRAIL_INTERVENED","texts":["masked integration prompt"]}')
        ) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        second_model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        prefix: Final = f"attachment-{uuid.uuid4().hex}"
        guardrail_name: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-{prefix}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "synthetic-guardrail-key",
        )
        policy_name: Final = f"policy-{prefix}"
        _create_policy(gateway, scenario.cleanups, policy_name, guardrail_name)
        active_scope: Final = _attachment_scope(spelling, policy_name, model, prefix)
        matching_key, nonmatching_key = _matching_and_nonmatching_keys(scenario, spelling, prefix, model, second_model)
        attachment: Final = gateway.request("POST", "/policies/attachments", active_scope)
        assert attachment.status_code == 200, attachment.text
        created_attachment: Final = _AttachmentResponse.model_validate_json(attachment.content)
        scenario.cleanups.callback(_delete_attachment, gateway, created_attachment.attachment_id)
        expected_scope: Final = {
            "attachment_id": created_attachment.attachment_id,
            "policy_name": policy_name,
            "scope": "*" if spelling == "*" else None,
            "teams": [f"{prefix}-*"] if spelling == "teams" else [],
            "keys": [f"{prefix}-*"] if spelling == "keys" else [],
            "models": [model] if spelling == "models" else [],
            "tags": [],
            "priority": None,
            "default": False,
        }
        assert created_attachment.model_dump() == expected_scope, attachment.text

        readback: Final = gateway.request("GET", f"/policies/attachments/{created_attachment.attachment_id}")
        assert readback.status_code == 200, readback.text
        assert _AttachmentResponse.model_validate_json(readback.content).model_dump() == expected_scope, readback.text
        rows: Final = read_rows(
            'SELECT policy_name, scope, teams, keys, models FROM "LiteLLM_PolicyAttachmentTable" '
            "WHERE attachment_id=%s",
            (created_attachment.attachment_id,),
        )
        assert rows == [
            {
                "policy_name": policy_name,
                "scope": "*" if spelling == "*" else None,
                "teams": [f"{prefix}-*"] if spelling == "teams" else [],
                "keys": [f"{prefix}-*"] if spelling == "keys" else [],
                "models": [model] if spelling == "models" else [],
            }
        ]

        matching_response: Final = _chat(gateway, model, matching_key, _PROMPT)
        assert matching_response.headers.get("x-litellm-applied-policies") == policy_name, matching_response.text
        global_response: Final = _chat(gateway, second_model, matching_key, _PROMPT) if spelling == "*" else None
        if global_response is not None:
            assert global_response.headers.get("x-litellm-applied-policies") == policy_name, global_response.text
        elif spelling == "models":
            nonmatching_model_response: Final = _chat(gateway, second_model, matching_key, _PROMPT)
            assert nonmatching_model_response.headers.get("x-litellm-applied-policies") is None, (
                nonmatching_model_response.text
            )
        elif nonmatching_key is not None:
            nonmatching_key_response: Final = _chat(gateway, model, nonmatching_key, _PROMPT)
            assert nonmatching_key_response.headers.get("x-litellm-applied-policies") is None, (
                nonmatching_key_response.text
            )
        global_responses: Final[tuple[httpx.Response, ...]] = () if global_response is None else (global_response,)

        guardrail_requests: Final = guardrail.drain()
        provider_requests: Final = provider.drain()
        expected_matches: Final = 2 if spelling == "*" else 1
        assert len(guardrail_requests) == expected_matches
        assert len(provider_requests) == 2
        expected_guardrail_models: Final = (model, second_model) if spelling == "*" else (model,)
        matching_responses: Final = (matching_response,) + global_responses
        expected_key_alias: Final = f"{prefix}-key" if spelling in ("teams", "keys") else None
        for request, expected_model, response in zip(
            guardrail_requests,
            expected_guardrail_models,
            matching_responses,
            strict=True,
        ):
            _assert_guardrail_request(
                request,
                _PROMPT,
                expected_model,
                matching_key,
                key_alias=expected_key_alias,
                response=response,
            )
        assert [(request.method, request.target) for request in guardrail_requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api")
        ] * expected_matches
        expected_provider_texts: Final = (
            (_MASKED, _MASKED)
            if spelling == "*"
            else (_MASKED, _PROMPT)
            if spelling in ("models", "teams", "keys")
            else (_MASKED,)
        )
        assert [(request.method, request.target) for request in provider_requests] == [
            ("POST", "/v1/chat/completions")
        ] * len(expected_provider_texts)
        for request, expected_text in zip(provider_requests, expected_provider_texts, strict=True):
            _assert_provider_request(request, expected_text)


def _delete_attachment(gateway: Gateway, attachment_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/policies/attachments/{attachment_id}")
    assert deleted.status_code == 200, deleted.text
    assert (
        read_rows('SELECT attachment_id FROM "LiteLLM_PolicyAttachmentTable" WHERE attachment_id=%s', (attachment_id,))
        == []
    )


def test_request_policy_name_selects_its_production_guardrail_and_is_removed_from_provider_body(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(
            lambda request: Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": [_mask_policy_prompt(request, _MASKED_FIRST)],
                    }
                ).encode()
            )
        ) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model])
        policy_name: Final = f"policy-{uuid.uuid4().hex}"
        first_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-one-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-one",
        )
        second_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-two-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-two",
        )
        policy_id: Final = _create_policy(gateway, scenario.cleanups, policy_name, first_guardrail)
        _create_published_version(gateway, scenario.cleanups, policy_name, policy_id, second_guardrail)

        response: Final = _chat(gateway, model, key, _PROMPT, policies=[policy_name])
        assert response.headers.get("x-litellm-applied-policies") == policy_name, response.text

        sdk_prompt: Final = f"{_PROMPT} via SDK {uuid.uuid4().hex}"
        with httpx.Client(trust_env=False, timeout=15) as client:
            sdk: Final = OpenAI(
                base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
                api_key=key,
                http_client=client,
            )
            sdk_response: Final = sdk.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": sdk_prompt}],
                extra_body={"policies": [policy_name]},
            )
            assert sdk_response.choices[0].message.content == "provider response"
        guardrail_requests: Final = guardrail.drain()
        provider_requests: Final = provider.drain()
        assert len(guardrail_requests) == 2
        assert len(provider_requests) == 2
        assert [request.headers["x-api-key"] for request in guardrail_requests] == ["guardrail-one"] * 2
        _assert_guardrail_request(
            guardrail_requests[0],
            _PROMPT,
            model,
            key,
            response=response,
        )
        _assert_guardrail_request(guardrail_requests[1], sdk_prompt, model, key)
        expected_provider_texts: Final = (f"{_MASKED_FIRST}: {_PROMPT}", f"{_MASKED_FIRST}: {sdk_prompt}")
        for request, expected_text in zip(provider_requests, expected_provider_texts, strict=True):
            _assert_provider_request(request, expected_text)
        assert [(request.method, request.target) for request in guardrail_requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api")
        ] * len(guardrail_requests)
        assert [(request.method, request.target) for request in provider_requests] == [
            ("POST", "/v1/chat/completions")
        ] * len(provider_requests)


def test_request_policy_version_id_selects_its_published_guardrail(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a just-published policy version id in request-body policies applies no guardrail until a DB resync"
    )
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(
            lambda request: Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": [
                            _MASKED_FIRST if request.headers.get("x-api-key") == "guardrail-one" else _MASKED_SECOND
                        ],
                    }
                ).encode()
            )
        ) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model])
        policy_name: Final = f"policy-{uuid.uuid4().hex}"
        first_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-one-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-one",
        )
        second_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-two-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-two",
        )
        policy_id: Final = _create_policy(gateway, scenario.cleanups, policy_name, first_guardrail)
        version_id: Final = _create_published_version(
            gateway, scenario.cleanups, policy_name, policy_id, second_guardrail
        )

        response: Final = _chat(gateway, model, key, _PROMPT, policies=[f"policy_{version_id}"])
        assert response.headers.get("x-litellm-applied-policies") == policy_name, response.text
        guardrail_requests: Final = guardrail.drain()
        assert len(guardrail_requests) == 1
        assert guardrail_requests[0].headers["x-api-key"] == "guardrail-two"
        _assert_guardrail_request(guardrail_requests[0], _PROMPT, model, key, response=response)
        provider_requests: Final = provider.drain()
        assert len(provider_requests) == 1
        _assert_provider_request(provider_requests[0], _MASKED_SECOND)
        assert [(request.method, request.target) for request in guardrail_requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api")
        ]
        assert [(request.method, request.target) for request in provider_requests] == [("POST", "/v1/chat/completions")]


def test_request_published_policy_version_id_resolves_after_force_sync(gateway: Gateway) -> None:
    with (
        wire_server(lambda _request: Reply(body=json.dumps(_PROVIDER_RESPONSE).encode())) as provider,
        wire_server(
            lambda request: Reply(
                body=json.dumps(
                    {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": [
                            _MASKED_FIRST if request.headers.get("x-api-key") == "guardrail-one" else _MASKED_SECOND
                        ],
                    }
                ).encode()
            )
        ) as guardrail,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        key: Final = scenario.key(models=[model])
        policy_name: Final = f"policy-{uuid.uuid4().hex}"
        first_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-one-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-one",
        )
        second_guardrail: Final = _create_guardrail(
            gateway,
            scenario.cleanups,
            f"guardrail-two-{uuid.uuid4().hex}",
            f"{guardrail.url}/beta/litellm_basic_guardrail_api",
            "guardrail-two",
        )
        policy_id: Final = _create_policy(gateway, scenario.cleanups, policy_name, first_guardrail)
        version_id: Final = _create_published_version(
            gateway, scenario.cleanups, policy_name, policy_id, second_guardrail
        )

        synchronized: Final = gateway.request(
            "POST",
            "/policies/resolve?force_sync=true",
            {"model": model},
        )
        assert synchronized.status_code == 200, synchronized.text

        response: Final = _chat(gateway, model, key, _PROMPT, policies=[f"policy_{version_id}"])
        assert response.headers.get("x-litellm-applied-policies") == policy_name, response.text
        guardrail_requests: Final = guardrail.drain()
        assert [(request.method, request.target) for request in guardrail_requests] == [
            ("POST", "/beta/litellm_basic_guardrail_api")
        ]
        assert [request.headers["x-api-key"] for request in guardrail_requests] == ["guardrail-two"]
        _assert_guardrail_request(guardrail_requests[0], _PROMPT, model, key, response=response)
        provider_requests: Final = provider.drain()
        assert [(request.method, request.target) for request in provider_requests] == [("POST", "/v1/chat/completions")]
        _assert_provider_request(provider_requests[0], _MASKED_SECOND)

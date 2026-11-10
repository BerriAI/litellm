import json
import uuid
from collections.abc import Callable
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai.types.chat import ChatCompletion
from pydantic import JsonValue

from litellm.types.prompts.init_prompts import ListPromptsResponse, PromptInfoResponse, PromptSpec

_PROVIDER_KEY: Final = "synthetic-prompt-provider-key"
_DISCOVERY_PROBE: Final = ("GET", "/v1/models")
_PROMPT_LIFECYCLE_STAGING_LATEST_BODY: Final[dict[str, JsonValue]] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "staging three"},
        {"role": "user", "content": "Hi x"},
        {"role": "user", "content": "client turn"},
    ],
    "temperature": 0.2,
}
_PROMPT_LIFECYCLE_STAGING_V2_BODY: Final[dict[str, JsonValue]] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "staging two patched"},
        {"role": "user", "content": "Hi x"},
        {"role": "user", "content": "client turn"},
    ],
    "temperature": 0.2,
}
_PROMPT_LIFECYCLE_PRODUCTION_BODY: Final[dict[str, JsonValue]] = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "production one"},
        {"role": "user", "content": "Hi x"},
        {"role": "user", "content": "client turn"},
    ],
    "temperature": 0.2,
}


def _is_discovery_probe(request: Request) -> bool:
    return (request.method, request.target) == _DISCOVERY_PROBE


def _template(model: str, marker: str) -> str:
    return f"---\nmodel: {model}\ntemperature: 0.2\n---\nSystem: {marker}\nUser: Hi {{{{name}}}}"


def _prompt_request(prompt_id: str, model: str, environment: str, marker: str) -> dict[str, JsonValue]:
    return {
        "prompt_id": prompt_id,
        "litellm_params": {
            "prompt_id": prompt_id,
            "prompt_integration": "dotprompt",
            "dotprompt_content": _template(model, marker),
        },
        "prompt_info": {"prompt_type": "db", "environment": environment},
    }


def _delete_prompt(gateway: Gateway, prompt_id: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/prompts/{prompt_id}")
    assert deleted.status_code == 200, deleted.text


def _create_prompt(
    gateway: Gateway,
    prompt_id: str,
    model: str,
    environment: str,
    marker: str,
) -> PromptSpec:
    response: Final = gateway.request("POST", "/prompts", _prompt_request(prompt_id, model, environment, marker))
    assert response.status_code == 200, response.text
    return PromptSpec.model_validate_json(response.content)


def _chat(
    gateway: Gateway,
    model: str,
    prompt_id: str,
    environment: str | None,
    version: int | None = None,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "client turn"}],
            "prompt_id": prompt_id,
            "prompt_variables": {"name": "x"},
            **({"prompt_environment": environment} if environment is not None else {}),
            **({"prompt_version": version} if version is not None else {}),
        },
    )


def _assert_chat_response(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    completion: Final = ChatCompletion.model_validate_json(response.content)
    assert completion.choices[0].message.content == "management response", response.text



def _chat_reply(content: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-prompt-management",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()
    )


def _provider_reply(request: Request) -> Reply:
    if _is_discovery_probe(request):
        return Reply(body=b'{"object":"list","data":[]}')
    assert request.method == "POST"
    assert request.target == "/v1/chat/completions"
    body: Final[dict[str, JsonValue]] = json.loads(request.body)
    assert body == {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "staging one"},
            {"role": "user", "content": "Hi x"},
            {"role": "user", "content": "client turn"},
        ],
        "temperature": 0.2,
    }, body
    return _chat_reply("management response")


def _prompt_rows(prompt_id: str, environment: str | None = None) -> list[dict[str, JsonValue]]:
    if environment is None:
        return read_rows(
            'SELECT prompt_id, version, environment, created_by, litellm_params FROM "LiteLLM_PromptTable" '
            "WHERE prompt_id = %s ORDER BY environment, version",
            (prompt_id,),
        )
    return read_rows(
        'SELECT prompt_id, version, environment, created_by, litellm_params FROM "LiteLLM_PromptTable" '
        "WHERE prompt_id = %s AND environment = %s ORDER BY version",
        (prompt_id, environment),
    )


def _prompt_labels(prompt_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT environment, version, prompt_info->>'label' AS label FROM \"LiteLLM_PromptTable\" "
        "WHERE prompt_id = %s ORDER BY environment, version",
        (prompt_id,),
    )


def _params(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    raw: Final = row["litellm_params"]
    return json.loads(raw) if isinstance(raw, str) else object_value(raw)


def test_prompt_create_persists_and_resolves_on_primary(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
        return _provider_reply(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        created_response: Final = gateway.request(
            "POST", "/prompts", _prompt_request(prompt_id, model, "staging", "staging one")
        )
        assert created_response.status_code == 200, created_response.text
        scenario.cleanups.callback(_delete_prompt, gateway, prompt_id)
        created: Final = PromptSpec.model_validate_json(created_response.content)
        assert (created.prompt_id, created.version, created.environment) == (f"{prompt_id}.v1", 1, "staging"), (
            created_response.text
        )

        rows: Final = _prompt_rows(prompt_id, "staging")
        assert len(rows) == 1, rows
        assert {
            "prompt_id": rows[0]["prompt_id"],
            "version": rows[0]["version"],
            "environment": rows[0]["environment"],
            "created_by": rows[0]["created_by"],
        } == {
            "prompt_id": prompt_id,
            "version": 1,
            "environment": "staging",
            "created_by": "default_user_id",
        }, rows
        assert _params(rows[0]) == {
            "prompt_id": prompt_id,
            "prompt_integration": "dotprompt",
            "dotprompt_content": _template(model, "staging one"),
            "ignore_prompt_manager_model": False,
            "ignore_prompt_manager_optional_params": False,
            "api_base": None,
            "api_key": None,
            "provider_specific_query_params": None,
        }, rows

        for path in (f"/prompts/{prompt_id}", f"/prompts/{prompt_id}/info"):
            response: Final = gateway.request("GET", path, params={"environment": "staging"})
            assert response.status_code == 200, response.text
            info: Final = PromptInfoResponse.model_validate_json(response.content)
            assert (
                info.prompt_spec.prompt_id,
                info.prompt_spec.version,
                info.prompt_spec.environment,
                info.prompt_spec.litellm_params.dotprompt_content,
            ) == (prompt_id, 1, "staging", _template(model, "staging one")), response.text

        staging_response: Final = gateway.request("GET", "/prompts/list", params={"environment": "staging"})
        production_response: Final = gateway.request("GET", "/prompts/list", params={"environment": "production"})
        assert staging_response.status_code == 200, staging_response.text
        assert production_response.status_code == 200, production_response.text
        staging: Final = ListPromptsResponse.model_validate_json(staging_response.content)
        production: Final = ListPromptsResponse.model_validate_json(production_response.content)
        assert tuple(prompt.prompt_id for prompt in staging.prompts if prompt.prompt_id == prompt_id) == (prompt_id,), (
            staging_response.text
        )
        assert tuple(prompt.prompt_id for prompt in production.prompts if prompt.prompt_id == prompt_id) == (), (
            production_response.text
        )

        _assert_chat_response(_chat(gateway, model, prompt_id, "staging"))
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/chat/completions"),
        ]


def _rendered_body(marker: str) -> dict[str, JsonValue]:
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": marker},
            {"role": "user", "content": "Hi x"},
            {"role": "user", "content": "client turn"},
        ],
        "temperature": 0.2,
    }


def _replies_with(content: str) -> Callable[[httpx.Response], bool]:
    def matches(response: httpx.Response) -> bool:
        if response.status_code != 200:
            return False
        return ChatCompletion.model_validate_json(response.content).choices[0].message.content == content

    return matches


def test_prompt_create_and_patch_resolve_on_peer_after_database_sync(gateway: Gateway, peer: Gateway) -> None:
    replies: Final = {
        json.dumps(_rendered_body("staging one"), sort_keys=True): "created template",
        json.dumps(_rendered_body("staging one patched"), sort_keys=True): "patched template",
    }

    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        raw: Final = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "client turn"}]}
        if body == raw:
            return _chat_reply("prompt has not synchronized")
        canonical: Final = json.dumps(body, sort_keys=True)
        assert canonical in replies, body
        return _chat_reply(replies[canonical])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        created: Final = gateway.request(
            "POST", "/prompts", _prompt_request(prompt_id, model, "staging", "staging one")
        )
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(_delete_prompt, gateway, prompt_id)
        assert PromptSpec.model_validate_json(created.content).prompt_id == f"{prompt_id}.v1", created.text
        synced: Final = eventually(
            lambda: _chat(peer, model, prompt_id, "staging"),
            _replies_with("created template"),
            seconds=40,
        )
        assert ChatCompletion.model_validate_json(synced.content).choices[0].message.content == "created template", (
            synced.text
        )

        patched: Final = gateway.request(
            "PATCH",
            f"/prompts/{prompt_id}.v1",
            {
                "litellm_params": {
                    "prompt_id": prompt_id,
                    "prompt_integration": "dotprompt",
                    "dotprompt_content": _template(model, "staging one patched"),
                },
            },
            params={"environment": "staging"},
        )
        assert patched.status_code == 200, patched.text
        resynced: Final = eventually(
            lambda: _chat(peer, model, prompt_id, "staging"),
            _replies_with("patched template"),
            seconds=40,
        )
        assert (
            ChatCompletion.model_validate_json(resynced.content).choices[0].message.content == "patched template"
        ), resynced.text
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert all((request.method, request.target) == ("POST", "/v1/chat/completions") for request in requests), (
            requests
        )
        assert json.loads(requests[-1].body) == _rendered_body("staging one patched"), requests[-1].body


def test_prompt_update_patch_and_environment_delete_are_isolated(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body in (
            _PROMPT_LIFECYCLE_STAGING_LATEST_BODY,
            _PROMPT_LIFECYCLE_STAGING_V2_BODY,
            _PROMPT_LIFECYCLE_PRODUCTION_BODY,
        ), body
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-prompt-lifecycle",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "management response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _create_prompt(gateway, prompt_id, model, "production", "production one")
        scenario.cleanups.callback(_delete_prompt, gateway, prompt_id)
        _create_prompt(gateway, prompt_id, model, "staging", "staging one")

        updated_response: Final = gateway.request(
            "PUT",
            f"/prompts/{prompt_id}",
            _prompt_request(prompt_id, model, "staging", "staging two"),
        )
        assert updated_response.status_code == 200, updated_response.text
        updated: Final = PromptSpec.model_validate_json(updated_response.content)
        assert (updated.prompt_id, updated.version, updated.environment) == (f"{prompt_id}.v2", 2, "staging"), (
            updated_response.text
        )
        latest_response: Final = gateway.request(
            "PUT",
            f"/prompts/{prompt_id}",
            _prompt_request(prompt_id, model, "staging", "staging three"),
        )
        assert latest_response.status_code == 200, latest_response.text
        latest: Final = PromptSpec.model_validate_json(latest_response.content)
        assert (latest.prompt_id, latest.version, latest.environment) == (f"{prompt_id}.v3", 3, "staging"), (
            latest_response.text
        )
        versions_response: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}/versions",
            params={"environment": "staging"},
        )
        assert versions_response.status_code == 200, versions_response.text
        staging_versions: Final = ListPromptsResponse.model_validate_json(versions_response.content)
        assert tuple((prompt.version, prompt.environment) for prompt in staging_versions.prompts) == (
            (3, "staging"),
            (2, "staging"),
            (1, "staging"),
        ), versions_response.text

        production_patch: Final = gateway.request(
            "PATCH",
            f"/prompts/{prompt_id}",
            {"prompt_info": {"prompt_type": "db", "environment": "production", "label": "production-patched"}},
            params={"environment": "production"},
        )
        assert production_patch.status_code == 200, production_patch.text
        production_info: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}/info",
            params={"environment": "production"},
        )
        assert production_info.status_code == 200, production_info.text
        production_spec: Final = PromptInfoResponse.model_validate_json(production_info.content).prompt_spec
        assert object_value(production_spec.prompt_info.model_dump(mode="json")).get("label") == "production-patched", (
            production_info.text
        )
        prompt_labels: Final = _prompt_labels(prompt_id)
        assert tuple((row["environment"], row["version"], row["label"]) for row in prompt_labels) == (
            ("production", 1, "production-patched"),
            ("staging", 1, None),
            ("staging", 2, None),
            ("staging", 3, None),
        ), prompt_labels

        base_patch: Final = gateway.request(
            "PATCH",
            f"/prompts/{prompt_id}",
            {"prompt_info": {"prompt_type": "db", "environment": "staging", "label": "base-patched"}},
            params={"environment": "staging"},
        )
        assert base_patch.status_code == 200, base_patch.text
        base_info: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}.v3/info",
            params={"environment": "staging"},
        )
        assert base_info.status_code == 200, base_info.text
        base_spec: Final = PromptInfoResponse.model_validate_json(base_info.content).prompt_spec
        assert base_spec.litellm_params.dotprompt_content == _template(model, "staging three"), base_info.text
        assert object_value(base_spec.prompt_info.model_dump(mode="json"))["label"] == "base-patched", base_info.text

        version_patch: Final = gateway.request(
            "PATCH",
            f"/prompts/{prompt_id}.v2",
            {
                "litellm_params": {
                    "prompt_id": prompt_id,
                    "prompt_integration": "dotprompt",
                    "dotprompt_content": _template(model, "staging two patched"),
                },
                "prompt_info": {"prompt_type": "db", "environment": "staging", "label": "version-patched"},
            },
            params={"environment": "staging"},
        )
        assert version_patch.status_code == 200, version_patch.text
        version_info: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}.v2/info",
            params={"environment": "staging"},
        )
        assert version_info.status_code == 200, version_info.text
        version_spec: Final = PromptInfoResponse.model_validate_json(version_info.content).prompt_spec
        assert version_spec.litellm_params.dotprompt_content == _template(model, "staging two patched"), (
            version_info.text
        )
        assert object_value(version_spec.prompt_info.model_dump(mode="json"))["label"] == "version-patched", (
            version_info.text
        )
        staging_rows: Final = _prompt_rows(prompt_id, "staging")
        assert tuple(_params(row)["dotprompt_content"] for row in staging_rows) == (
            _template(model, "staging one"),
            _template(model, "staging two patched"),
            _template(model, "staging three"),
        ), staging_rows
        latest_info: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}/info",
            params={"environment": "staging"},
        )
        assert latest_info.status_code == 200, latest_info.text
        latest_spec: Final = PromptInfoResponse.model_validate_json(latest_info.content).prompt_spec
        assert latest_spec.litellm_params.dotprompt_content == _template(model, "staging three"), latest_info.text
        assert object_value(latest_spec.prompt_info.model_dump(mode="json"))["label"] == "base-patched", (
            latest_info.text
        )
        patched_versions_response: Final = gateway.request(
            "GET",
            f"/prompts/{prompt_id}/versions",
            params={"environment": "staging"},
        )
        assert patched_versions_response.status_code == 200, patched_versions_response.text
        patched_versions: Final = ListPromptsResponse.model_validate_json(patched_versions_response.content)
        assert tuple((prompt.version, prompt.environment) for prompt in patched_versions.prompts) == (
            (3, "staging"),
            (2, "staging"),
            (1, "staging"),
        ), patched_versions_response.text

        _assert_chat_response(_chat(gateway, model, prompt_id, "staging"))
        _assert_chat_response(_chat(gateway, model, prompt_id, "staging", 2))
        deleted: Final = gateway.request(
            "DELETE",
            f"/prompts/{prompt_id}",
            params={"environment": "staging"},
        )
        assert deleted.status_code == 200, deleted.text
        assert _prompt_rows(prompt_id, "staging") == []
        production_rows: Final = _prompt_rows(prompt_id, "production")
        assert len(production_rows) == 1, production_rows
        assert _params(production_rows[0])["dotprompt_content"] == _template(model, "production one"), production_rows
        remaining_response: Final = gateway.request("GET", f"/prompts/{prompt_id}/versions")
        assert remaining_response.status_code == 200, remaining_response.text
        remaining: Final = ListPromptsResponse.model_validate_json(remaining_response.content)
        assert tuple((prompt.version, prompt.environment) for prompt in remaining.prompts) == ((1, "production"),), (
            remaining_response.text
        )

        _assert_chat_response(_chat(gateway, model, prompt_id, None))
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        expected_requests: Final = [
            ("POST", "/v1/chat/completions", _PROMPT_LIFECYCLE_STAGING_LATEST_BODY),
            ("POST", "/v1/chat/completions", _PROMPT_LIFECYCLE_STAGING_V2_BODY),
            ("POST", "/v1/chat/completions", _PROMPT_LIFECYCLE_PRODUCTION_BODY),
        ]
        assert [
            (request.method, request.target, json.loads(request.body)) for request in requests
        ] == expected_requests, requests


def test_prompt_environment_delete_stops_applying_the_deleted_template(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: after DELETE /prompts/{id}?environment=staging, a prompt_environment=staging request still applies "
        "the deleted template's model and fails with 400 LLM Provider NOT provided"
    )

    def respond(request: Request) -> Reply:
        if _is_discovery_probe(request):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}"
        body: Final[dict[str, JsonValue]] = json.loads(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "client turn"}],
        }, body
        return _chat_reply("management response")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1", api_key=_PROVIDER_KEY)
        prompt_id: Final = f"prompt-{uuid.uuid4().hex}"
        _create_prompt(gateway, prompt_id, model, "production", "production one")
        scenario.cleanups.callback(_delete_prompt, gateway, prompt_id)
        _create_prompt(gateway, prompt_id, model, "staging", "staging one")
        deleted: Final = gateway.request(
            "DELETE",
            f"/prompts/{prompt_id}",
            params={"environment": "staging"},
        )
        assert deleted.status_code == 200, deleted.text

        response: Final = _chat(gateway, model, prompt_id, "staging")
        _assert_chat_response(response)
        requests: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
        assert [(request.method, request.target) for request in requests] == [
            ("POST", "/v1/chat/completions"),
        ]

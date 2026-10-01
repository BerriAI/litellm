from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Final, TypeAlias

import httpx
from integration._support.client import Gateway, Scenario, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

CONFIG_STORE_ID: Final = "vs_integration_config_store"
SEARCH_PATH: Final = f"/vector_stores/{CONFIG_STORE_ID}/search"
OTHER_STORE_ID: Final = "vs_some_other_store"
JsonObject: TypeAlias = dict[str, JsonValue]


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: request payloads require JSON arrays


def _permission_for_stores(*store_ids: str) -> JsonObject:
    permission: Final[JsonObject] = {"vector_stores": _json_array(*store_ids)}
    return permission


def _team_restricted_key(scenario: Scenario, model: str) -> str:
    team: Final = scenario.team(models=_json_array(model), object_permission=_permission_for_stores(OTHER_STORE_ID))
    return scenario.key(team_id=team, models=_json_array(model))


def _store_searches(
    observations: tuple[Mapping[str, JsonValue], ...], marker: str
) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        observation
        for observation in observations
        if observation["path"] == SEARCH_PATH and marker in str(observation["body"])
    )


def _anthropic_provider(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/messages", request.target
    body: Final = object_value(json.loads(request.body))
    reply_body: Final[JsonObject] = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": _json_array({"type": "text", "text": "synthetic"}),
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    return Reply(body=json.dumps(reply_body).encode())


def _messages(gateway: Gateway, model: str, key: str, marker: str) -> httpx.Response:
    body: Final[JsonObject] = {
        "model": model,
        "messages": _json_array({"role": "user", "content": marker}),
        "max_tokens": 16,
    }
    return gateway.request(
        "POST",
        "/v1/messages",
        body,
        key=key,
    )


def _responses(gateway: Gateway, model: str, key: str, marker: str) -> httpx.Response:
    body: Final[JsonObject] = {"model": model, "input": marker}
    return gateway.request("POST", "/v1/responses", body, key=key)


def test_chat_completions_vector_store_ids_are_checked_against_key_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model), object_permission=_permission_for_stores(OTHER_STORE_ID))
        marker: Final = f"lit5610 chat vector_store_ids denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "vector_store_ids": _json_array(CONFIG_STORE_ID),
        }

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=key,
        )

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "key_vector_store_access_denied", response.text
        assert _store_searches(upstream_observations(gateway), marker) == ()


def test_chat_completions_file_search_uses_team_vector_store_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _team_restricted_key(scenario, model)
        marker: Final = f"lit5610 chat file search denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "tools": _json_array({"type": "file_search", "vector_store_ids": _json_array(CONFIG_STORE_ID)}),
        }

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=key,
        )

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text
        assert _store_searches(upstream_observations(gateway), marker) == ()


def test_responses_file_search_uses_team_vector_store_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=f"{gateway.upstream_url}/v1")
        key: Final = _team_restricted_key(scenario, model)
        marker: Final = f"lit5610 responses file search denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "input": marker,
            "tools": _json_array({"type": "file_search", "vector_store_ids": _json_array(CONFIG_STORE_ID)}),
        }

        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            body,
            key=key,
        )

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text
        assert _store_searches(upstream_observations(gateway), marker) == ()


def test_vector_store_search_route_uses_team_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _team_restricted_key(scenario, model)
        marker: Final = f"lit5610 vector store search route denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {"query": marker}

        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{CONFIG_STORE_ID}/search",
            body,
            key=key,
        )

        assert response.status_code == 401, response.text
        assert _store_searches(upstream_observations(gateway), marker) == ()


def test_restricted_key_can_use_plain_chat_messages_and_responses(gateway: Gateway) -> None:
    with wire_server(_anthropic_provider) as anthropic_wire:
        with gateway.scenario() as scenario:
            chat_model: Final = scenario.model()
            messages_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929",
                api_base=anthropic_wire.url,
                api_key="synthetic-anthropic-key",
            )
            responses_model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=f"{gateway.upstream_url}/v1")
            key: Final = scenario.key(
                models=_json_array(chat_model, messages_model, responses_model),
                object_permission=_permission_for_stores(OTHER_STORE_ID),
            )
            chat_marker: Final = f"lit5610 plain chat restricted key {uuid.uuid4().hex}"
            messages_marker: Final = f"lit5610 plain messages restricted key {uuid.uuid4().hex}"
            responses_marker: Final = f"lit5610 plain responses restricted key {uuid.uuid4().hex}"

            chat_body: Final[JsonObject] = {
                "model": chat_model,
                "messages": _json_array({"role": "user", "content": chat_marker}),
            }
            chat: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                chat_body,
                key=key,
            )
            messages: Final = _messages(gateway, messages_model, key, messages_marker)
            responses: Final = _responses(gateway, responses_model, key, responses_marker)

        assert chat.status_code == 200, chat.text
        assert messages.status_code == 200, messages.text
        assert responses.status_code == 200, responses.text


def test_chat_completions_top_level_retrieval_config_uses_team_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _team_restricted_key(scenario, model)
        marker: Final = f"lit5610 chat top-level retrieval config denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "retrieval_config": {
                "vector_store_id": CONFIG_STORE_ID,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        }

        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=key,
        )
        observations: Final = upstream_observations(gateway)

        assert response.status_code == 401, f"{response.text}; scripted_upstream_observations={observations!r}"
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text

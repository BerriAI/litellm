from __future__ import annotations

import operator
import time
import uuid
from collections.abc import Callable, Iterator
from itertools import accumulate
from pathlib import Path
from typing import Final

import httpx
import pytest
from _openinference_support import (
    _anthropic_response,
    _chat_request_marker,
    _chat_response,
    _chat_stream_response,
    _chat_tool_call,
    _matching_marker_spans,
    _responses_response,
    _rig,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

CREDENTIAL_NAME: Final = "arize-space-a"
SECRET_API_KEY: Final = "space-a-arize-key-" + uuid.uuid4().hex
SECRET_SPACE_ID: Final = "space-a-id-" + uuid.uuid4().hex
AMBIENT_ARIZE_ENVIRONMENT: Final = ("ARIZE_API_KEY", "ARIZE_SPACE_ID")
CREDENTIAL_CONFIG: Final[dict[str, JsonValue]] = {
    "credential_list": [
        {
            "credential_name": CREDENTIAL_NAME,
            "credential_values": {
                "arize_api_key": "os.environ/SPACE_A_ARIZE_API_KEY",
                "arize_space_id": "os.environ/SPACE_A_ARIZE_SPACE_ID",
            },
            "credential_info": {},
        },
        {
            "credential_name": "model-only",
            "credential_values": {"api_key": "sk-not-a-logging-value"},
            "credential_info": {},
        },
    ]
}
SECRET_ENVIRONMENT: Final = {"SPACE_A_ARIZE_API_KEY": SECRET_API_KEY, "SPACE_A_ARIZE_SPACE_ID": SECRET_SPACE_ID}


def _reference_entry(credential_name: str = CREDENTIAL_NAME) -> dict[str, JsonValue]:
    return {
        "callback_name": "arize",
        "callback_type": "success",
        "litellm_credential_name": credential_name,
        "callback_vars": {},
    }


def _echo_marker(request: Request) -> Reply:
    return _chat_response(_chat_request_marker(request))


def _send_chat(proxy: Gateway, model: str, key: str, marker: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
        key=key,
    )


def _chat(proxy: Gateway, model: str, key: str, marker: str) -> None:
    response: Final = _send_chat(proxy, model, key, marker)
    assert response.status_code == 200, response.text


def _drains(destination: Wire) -> Iterator[tuple[Request, ...]]:
    while True:
        yield destination.drain()
        time.sleep(0.1)


def _carries(request: Request, marker: str) -> bool:
    return any(True for _ in _matching_marker_spans((request,), marker))


def _received_until(destination: Wire, marker: str, seconds: float = 30) -> tuple[Request, ...]:
    deadline: Final = time.monotonic() + seconds
    received: Final = next(
        batch
        for batch in accumulate(_drains(destination), operator.add)
        if any(_carries(request, marker) for request in batch) or time.monotonic() >= deadline
    )
    assert any(_carries(request, marker) for request in received), f"no span for {marker} reached the destination"
    return received


def test_key_referencing_a_credential_exports_with_the_secret_manager_values(gateway: Gateway, tmp_path: Path) -> None:
    unreferenced_marker: Final = "plain-" + uuid.uuid4().hex
    referenced_marker: Final = "ref-" + uuid.uuid4().hex
    with (
        _rig(
            gateway,
            tmp_path,
            _echo_marker,
            extra_config=CREDENTIAL_CONFIG,
            environment=SECRET_ENVIRONMENT,
            remove_environment=AMBIENT_ARIZE_ENVIRONMENT,
            disabled_environment=AMBIENT_ARIZE_ENVIRONMENT,
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        team: Final = scenario.team()
        referencing_key: Final = scenario.key(team_id=team, metadata={"logging": [_reference_entry()]})
        unreferenced_key: Final = scenario.key(team_id=team)

        _chat(rig.proxy, rig.model, unreferenced_key, unreferenced_marker)
        _chat(rig.proxy, rig.model, referencing_key, referenced_marker)
        received: Final = _received_until(rig.destination, referenced_marker)

        exports: Final = tuple(request for request in received if _carries(request, referenced_marker))
        for export in exports:
            headers = {name.lower(): value for name, value in export.headers.items()}
            assert headers.get("api_key") == SECRET_API_KEY, headers
            assert headers.get("arize-space-id") == SECRET_SPACE_ID, headers
        assert not any(_carries(request, unreferenced_marker) for request in received)

        info: Final = rig.proxy.request("GET", "/key/info", params={"key": referencing_key})
        assert info.status_code == 200, info.text
        assert CREDENTIAL_NAME in info.text
        assert SECRET_API_KEY not in info.text
        assert SECRET_SPACE_ID not in info.text


def test_only_proxy_admins_change_logging_on_a_key_that_references_a_credential(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        _rig(gateway, tmp_path, _echo_marker, extra_config=CREDENTIAL_CONFIG) as rig,
        rig.proxy.scenario() as scenario,
    ):
        team: Final = scenario.team()
        team_admin: Final = scenario.member(team, role="admin")
        team_admin_key: Final = scenario.key(user_id=team_admin, team_id=team)
        langfuse_entry: Final[dict[str, JsonValue]] = {
            "callback_name": "langfuse",
            "callback_type": "success",
            "callback_vars": {"langfuse_public_key": "pk-lf-integration", "langfuse_secret_key": "sk-lf-integration"},
        }
        referencing_key: Final = scenario.key(team_id=team, metadata={"logging": [_reference_entry(), langfuse_entry]})

        def as_team_admin(path: str, body: dict[str, JsonValue]) -> httpx.Response:
            return rig.proxy.request("POST", path, body, key=team_admin_key)

        def assert_gated(response: httpx.Response) -> None:
            assert response.status_code == 403, response.text
            assert "litellm_credential_name" in response.text, response.text

        assert_gated(as_team_admin("/key/generate", {"team_id": team, "metadata": {"logging": [_reference_entry()]}}))

        sibling: Final = as_team_admin(
            "/key/update",
            {
                "key": referencing_key,
                "metadata": {
                    "logging": [
                        _reference_entry(),
                        {
                            "callback_name": "arize",
                            "callback_type": "success",
                            "callback_vars": {"arize_endpoint": "http://127.0.0.1:9/v1/traces"},
                        },
                    ]
                },
            },
        )
        assert_gated(sibling)
        assert_gated(as_team_admin("/key/update", {"key": referencing_key, "metadata": {"team": "unrelated"}}))
        assert_gated(as_team_admin("/key/regenerate", {"key": referencing_key, "metadata": {"team": "unrelated"}}))

        bulk: Final = as_team_admin(
            "/team/key/bulk_update",
            {"team_id": team, "key_ids": [referencing_key], "update_fields": {"metadata": {"team": "unrelated"}}},
        )
        assert bulk.status_code == 200, bulk.text
        assert bulk.json()["successful_updates"] == [], bulk.text
        assert "litellm_credential_name" in bulk.json()["failed_updates"][0]["failed_reason"], bulk.text

        renamed: Final = as_team_admin(
            "/key/update", {"key": referencing_key, "key_alias": "alias-" + uuid.uuid4().hex}
        )
        assert renamed.status_code == 200, renamed.text

        read_back: Final = rig.proxy.request("GET", "/key/info", params={"key": referencing_key}, key=team_admin_key)
        assert read_back.status_code == 200, read_back.text
        resaved: Final = as_team_admin(
            "/key/update",
            {"key": referencing_key, "metadata": {**read_back.json()["info"]["metadata"], "note": "unrelated edit"}},
        )
        assert resaved.status_code == 200, resaved.text

        stored: Final = rig.proxy.get("/key/info", params={"key": referencing_key})
        logging_entries: Final = stored["info"]["metadata"]["logging"]
        assert [entry["callback_name"] for entry in logging_entries] == ["arize", "langfuse"], stored
        assert logging_entries[0] == _reference_entry(), stored

        undefined: Final = rig.proxy.request(
            "POST", "/key/generate", {"metadata": {"logging": [_reference_entry("no-such-credential")]}}
        )
        assert undefined.status_code == 400, undefined.text
        assert "no-such-credential" in undefined.text, undefined.text


def test_only_proxy_admins_create_a_user_whose_key_references_a_credential(gateway: Gateway, tmp_path: Path) -> None:
    with (
        _rig(gateway, tmp_path, _echo_marker, extra_config=CREDENTIAL_CONFIG) as rig,
        rig.proxy.scenario() as scenario,
    ):
        organization: Final = scenario.organization()
        org_admin_key: Final = scenario.key(user_id=scenario.org_member(organization, "org_admin"))
        referencing_metadata: Final = {"logging": [_reference_entry()]}

        single: Final = rig.proxy.request(
            "POST",
            "/user/new",
            {"user_id": "user-" + uuid.uuid4().hex, "organizations": [organization], "metadata": referencing_metadata},
            key=org_admin_key,
        )
        assert single.status_code == 403, single.text
        assert "litellm_credential_name" in single.text, single.text

        delegate_key: Final = scenario.key(
            user_id=scenario.user(user_role="internal_user"), allowed_routes=["/management/v1/users/bulk"]
        )
        bulk: Final = rig.proxy.request(
            "POST",
            "/management/v1/users/bulk",
            {
                "users": [
                    {"user_id": "user-" + uuid.uuid4().hex, "auto_create_key": True, "metadata": referencing_metadata}
                ]
            },
            key=delegate_key,
        )
        assert bulk.status_code == 200, bulk.text
        assert [row["success"] for row in bulk.json()["data"]] == [False], bulk.text
        assert "litellm_credential_name" in bulk.json()["data"][0]["error"], bulk.text


def _carries_credential(request: Request) -> bool:
    headers: Final = {name.lower(): value for name, value in request.headers.items()}
    return headers.get("api_key") == SECRET_API_KEY and headers.get("arize-space-id") == SECRET_SPACE_ID


def _credential_exports(destination: Wire, seconds: float = 30) -> tuple[Request, ...]:
    return eventually(
        lambda: tuple(request for request in destination.drain() if _carries_credential(request)),
        bool,
        seconds=seconds,
    )


def _surface_reply(marker: str) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        if request.target.endswith("/responses"):
            return _responses_response(marker)
        if request.target.endswith("/messages"):
            return _anthropic_response(marker)
        return _chat_stream_response(marker, (_chat_tool_call(marker),))

    return reply


_SURFACES: Final[dict[str, tuple[str, str, dict[str, JsonValue]]]] = {
    "responses": ("/v1/responses", "openai/gpt-4o-mini", {"input": "weather in Paris?"}),
    "messages": (
        "/v1/messages",
        "anthropic/claude-opus-5-5",
        {"max_tokens": 64, "messages": [{"role": "user", "content": "weather in Paris?"}]},
    ),
    "chat-stream": (
        "/v1/chat/completions",
        "openai/gpt-4o-mini",
        {"stream": True, "messages": [{"role": "user", "content": "weather in Paris?"}]},
    ),
}


@pytest.mark.parametrize("surface", sorted(_SURFACES))
def test_referencing_key_exports_on_each_surface(surface: str, gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"{surface}-" + uuid.uuid4().hex
    path, model_name, body = _SURFACES[surface]
    with (
        _rig(
            gateway,
            tmp_path,
            _surface_reply(marker),
            extra_config=CREDENTIAL_CONFIG,
            environment=SECRET_ENVIRONMENT,
            remove_environment=AMBIENT_ARIZE_ENVIRONMENT,
            disabled_environment=AMBIENT_ARIZE_ENVIRONMENT,
            model_name=model_name,
            api_base_suffix="" if surface == "messages" else "/v1",
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"logging": [_reference_entry()]})
        response: Final = rig.proxy.request(
            "POST",
            path,
            {"model": rig.model, **body, "metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
            key=key,
        )
        assert response.status_code == 200, response.text
        received: Final = _received_until(rig.destination, marker)
        exports: Final = tuple(request for request in received if _carries(request, marker))
        assert exports and all(_carries_credential(request) for request in exports), [
            dict(request.headers) for request in exports
        ]


def test_referencing_key_exports_through_anthropic_pass_through(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "passthrough-" + uuid.uuid4().hex
    with wire_server(lambda _request: _anthropic_response(marker)) as anthropic:
        with (
            _rig(
                gateway,
                tmp_path,
                _echo_marker,
                extra_config=CREDENTIAL_CONFIG,
                environment={
                    **SECRET_ENVIRONMENT,
                    "ANTHROPIC_API_BASE": anthropic.url,
                    "ANTHROPIC_API_KEY": "sk-ant-synthetic",
                },
                remove_environment=AMBIENT_ARIZE_ENVIRONMENT,
                disabled_environment=AMBIENT_ARIZE_ENVIRONMENT,
            ) as rig,
            rig.proxy.scenario() as scenario,
        ):
            key: Final = scenario.key(metadata={"logging": [_reference_entry()]})
            response: Final = rig.proxy.request(
                "POST",
                "/anthropic/v1/messages",
                {"model": "claude-opus-5-5", "max_tokens": 16, "messages": [{"role": "user", "content": marker}]},
                key=key,
                headers={"anthropic-version": "2023-06-01"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["id"] == marker, response.text
            assert anthropic.drain(), "the pass-through never reached the scripted Anthropic upstream"
            assert _credential_exports(rig.destination)


def test_referencing_key_exports_on_otel_v1(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "otel-v1-" + uuid.uuid4().hex
    with (
        _rig(
            gateway,
            tmp_path,
            _echo_marker,
            extra_config=CREDENTIAL_CONFIG,
            environment={**SECRET_ENVIRONMENT, "LITELLM_OTEL_V2": "false"},
            remove_environment=AMBIENT_ARIZE_ENVIRONMENT,
            disabled_environment=AMBIENT_ARIZE_ENVIRONMENT,
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"logging": [_reference_entry()]})
        _chat(rig.proxy, rig.model, key, marker)
        assert _credential_exports(rig.destination)


def test_reference_to_a_credential_without_logging_values_exports_nothing_and_is_unhealthy(
    gateway: Gateway, tmp_path: Path
) -> None:
    dropped_marker: Final = "model-only-" + uuid.uuid4().hex
    sentinel_marker: Final = "sentinel-" + uuid.uuid4().hex
    with (
        _rig(
            gateway,
            tmp_path,
            _echo_marker,
            extra_config=CREDENTIAL_CONFIG,
            environment=SECRET_ENVIRONMENT,
            remove_environment=AMBIENT_ARIZE_ENVIRONMENT,
            disabled_environment=AMBIENT_ARIZE_ENVIRONMENT,
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        dropped_key: Final = scenario.key(metadata={"logging": [_reference_entry("model-only")]})
        sentinel_key: Final = scenario.key(metadata={"logging": [_reference_entry()]})

        _chat(rig.proxy, rig.model, dropped_key, dropped_marker)
        _chat(rig.proxy, rig.model, sentinel_key, sentinel_marker)
        received: Final = _received_until(rig.destination, sentinel_marker)
        assert not any(_carries(request, dropped_marker) for request in received)

        health: Final = rig.proxy.request("POST", "/key/health", {}, key=dropped_key)
        assert health.status_code == 200, health.text
        assert health.json()["key"] == "unhealthy", health.text


def test_malformed_key_logging_fails_requests_like_main(gateway: Gateway, tmp_path: Path) -> None:
    stored_logging: Final[dict[str, JsonValue]] = {
        "string": "abc",
        "list-with-string": ["x"],
        "object": {"callback_name": "arize"},
        "empty-list": [],
    }
    with (
        _rig(gateway, tmp_path, _echo_marker, extra_config=CREDENTIAL_CONFIG) as rig,
        rig.proxy.scenario() as scenario,
    ):
        statuses: Final = {
            label: _send_chat(
                rig.proxy, rig.model, scenario.key(metadata={"logging": value}), "malformed-" + uuid.uuid4().hex
            ).status_code
            for label, value in stored_logging.items()
        }

    assert statuses == {"string": 500, "list-with-string": 500, "object": 500, "empty-list": 200}, statuses

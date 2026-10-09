import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def model_identity(gateway: Gateway, alias: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    entry: Final = next(object_value(value) for value in entries if object_value(value)["model_name"] == alias)
    return string_value(object_value(entry["model_info"])["id"])


@dataclass(frozen=True, slots=True)
class _ConfigCallbackRig:
    first: Gateway
    peer: Gateway
    database_url: str
    generic: Wire
    langfuse: Wire
    provider: Wire


@contextmanager
def _config_callback_rig(
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Callable[[Request], Reply],
    *,
    generic_flush_interval: str,
) -> Iterator[_ConfigCallbackRig]:
    def langfuse(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target.startswith("/api/public/projects"), request.target
            return Reply(body=b'{"data":[{"id":"integration-project","name":"integration"}]}')
        assert request.method == "POST"
        assert request.target.startswith("/api/public/otel/v1/traces"), request.target
        return Reply(content_type="application/x-protobuf")

    def generic(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer integration-generic-sink-secret"
        return Reply()

    with (
        scratch_database() as database_url,
        wire_server(provider) as provider_wire,
        wire_server(langfuse) as langfuse_wire,
        wire_server(generic) as generic_wire,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        environment: Final = {
            "DATABASE_URL": database_url,
            "DEFAULT_FLUSH_INTERVAL_SECONDS": generic_flush_interval,
            "LANGFUSE_FLUSH_INTERVAL": "1",
            "LITELLM_STORE_AUDIT_LOGS": "True",
        }
        removed: Final = (
            "DATABASE_URL_READ_REPLICA",
            "GENERIC_LOGGER_ENDPOINT",
            "GENERIC_LOGGER_HEADERS",
            "LANGFUSE_HOST",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        )
        with (
            owned_proxy(gateway, tmp_path, environment, remove_environment=removed) as first,
            owned_proxy(gateway, tmp_path, environment, remove_environment=removed) as peer,
        ):
            yield _ConfigCallbackRig(first, peer, database_url, generic_wire, langfuse_wire, provider_wire)


def _config_section(database_url: str, section: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
        (section,),
        database_url=database_url,
    )
    assert len(rows) == 1, rows
    return object_value(rows[0]["param_value"])


def _callback_entries(response: httpx.Response) -> dict[str, dict[str, JsonValue]]:
    body: Final = object_value(response.json())
    callbacks: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_python(body["callbacks"])
    return {string_value(callback["name"]): callback for callback in callbacks}


def _callback_entries_match(response: httpx.Response, expected: dict[str, dict[str, JsonValue]]) -> bool:
    entries: Final = _callback_entries(response)
    return all(entries.get(name) == callback for name, callback in expected.items())


def _completion_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + text,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def test_multi_section_config_update_merges_sent_keys_normalizes_callbacks_and_applies_on_the_peer(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = "config-update-" + uuid.uuid4().hex
    provider_secret: Final = "provider-secret-" + marker
    langfuse_public_key: Final = "public-key-" + marker
    langfuse_secret_key: Final = "secret-key-" + marker
    provider_requests: Final[list[Request]] = []  # mutable-ok: request count distinguishes the retry

    def provider(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        content: Final = object_value(body["messages"][0])["content"]
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
        }, request.body
        assert isinstance(content, str)
        retry_attempts: Final = sum(marker + "-retry" in prior.body.decode() for prior in provider_requests)
        provider_requests.append(request)
        if marker + "-retry" in content and retry_attempts == 0:
            return Reply(
                status=500,
                body=b'{"error":{"message":"retry control","type":"server_error","code":"retry_control"}}',
            )
        return _completion_reply(content)

    with (
        _config_callback_rig(gateway, tmp_path, monkeypatch, provider, generic_flush_interval="1") as rig,
        rig.first.scenario() as scenario,
    ):
        seed: Final = rig.first.request(
            "POST",
            "/config/update",
            {
                "general_settings": {"alerting_threshold": 321},
                "litellm_settings": {"success_callback": ["generic_api"], "drop_params": True},
                "environment_variables": {
                    "GENERIC_LOGGER_ENDPOINT": rig.generic.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer integration-generic-sink-secret",
                },
                "router_settings": {"allowed_fails": 7},
            },
        )
        assert seed.status_code == 200, seed.text
        assert seed.json() == {"message": "Config updated successfully"}, seed.text
        update: Final = rig.first.request(
            "POST",
            "/config/update",
            {
                "general_settings": {"store_prompts_in_spend_logs": True},
                "litellm_settings": {"success_callback": ["Langfuse"]},
                "environment_variables": {
                    "LANGFUSE_HOST": rig.langfuse.url,
                    "LANGFUSE_PUBLIC_KEY": langfuse_public_key,
                    "LANGFUSE_SECRET_KEY": langfuse_secret_key,
                },
                "router_settings": {"num_retries": 1},
            },
        )
        assert update.status_code == 200, update.text
        assert update.json() == {"message": "Config updated successfully"}, update.text
        assert _config_section(rig.database_url, "general_settings") == {
            "alerting_threshold": 321,
            "store_prompts_in_spend_logs": True,
        }
        litellm_settings: Final = _config_section(rig.database_url, "litellm_settings")
        assert litellm_settings["drop_params"] is True
        success_callbacks: Final = TypeAdapter(list[str]).validate_python(litellm_settings["success_callback"])
        assert sorted(success_callbacks) == ["generic_api", "langfuse"]
        assert _config_section(rig.database_url, "router_settings") == {"allowed_fails": 7, "num_retries": 1}
        environment_values: Final = _config_section(rig.database_url, "environment_variables")
        plain_environment: Final = {
            "GENERIC_LOGGER_ENDPOINT": rig.generic.url,
            "GENERIC_LOGGER_HEADERS": "Authorization=Bearer integration-generic-sink-secret",
            "LANGFUSE_HOST": rig.langfuse.url,
            "LANGFUSE_PUBLIC_KEY": langfuse_public_key,
            "LANGFUSE_SECRET_KEY": langfuse_secret_key,
        }
        assert set(environment_values) == set(plain_environment), environment_values
        assert all(
            isinstance(environment_values[name], str) and environment_values[name] != value
            for name, value in plain_environment.items()
        ), environment_values
        assert all(value not in json.dumps(environment_values) for value in plain_environment.values())

        expected_callbacks: Final[dict[str, dict[str, JsonValue]]] = {
            "generic_api": {
                "name": "generic_api",
                "variables": {
                    "GENERIC_LOGGER_ENDPOINT": rig.generic.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer integration-generic-sink-secret",
                },
                "type": "success",
            },
            "langfuse": {
                "name": "langfuse",
                "variables": {
                    "LANGFUSE_HOST": rig.langfuse.url,
                    "LANGFUSE_PUBLIC_KEY": langfuse_public_key,
                    "LANGFUSE_SECRET_KEY": langfuse_secret_key,
                },
                "type": "success",
            },
        }
        callbacks_first: Final = rig.first.request("GET", "/get/config/callbacks")
        assert callbacks_first.status_code == 200, callbacks_first.text
        assert _callback_entries_match(callbacks_first, expected_callbacks), callbacks_first.text
        callbacks_peer: Final = eventually(
            lambda: rig.peer.request("GET", "/get/config/callbacks"),
            lambda response: response.status_code == 200 and _callback_entries_match(response, expected_callbacks),
            seconds=20,
        )
        assert _callback_entries_match(callbacks_peer, expected_callbacks), callbacks_peer.text

        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key=provider_secret)
        api_key: Final = scenario.key(models=[model])
        chat: Final = rig.peer.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "stream": False},
            key=api_key,
        )
        assert chat.status_code == 200, chat.text
        assert chat.json()["choices"][0]["message"]["content"] == marker
        generic_batches: Final[list[Request]] = []  # mutable-ok: the sink queue is drained while waiting
        langfuse_batches: Final[list[Request]] = []  # mutable-ok: the sink queue is drained while waiting

        def collect_generic() -> tuple[Request, ...]:
            generic_batches.extend(rig.generic.drain())
            return tuple(generic_batches)

        def collect_langfuse() -> tuple[Request, ...]:
            langfuse_batches.extend(rig.langfuse.drain())
            return tuple(langfuse_batches)

        generic_deliveries: Final = eventually(
            collect_generic,
            lambda requests: marker.encode() in b"".join(request.body for request in requests),
            seconds=15,
        )
        langfuse_deliveries: Final = eventually(
            collect_langfuse,
            lambda requests: marker.encode() in b"".join(request.body for request in requests),
            seconds=15,
        )
        assert any(marker.encode() in request.body for request in generic_deliveries)
        assert any(marker.encode() in request.body for request in langfuse_deliveries)

        retry: Final = rig.peer.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker + "-retry"}], "stream": False},
            key=api_key,
        )
        assert retry.status_code == 200, retry.text
        assert sum(marker + "-retry" in request.body.decode() for request in provider_requests) == 2


def test_deleted_callback_case_variant_stops_delivery_on_both_workers(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    marker: Final = "callback-delete-" + uuid.uuid4().hex
    provider_secret: Final = "provider-secret-" + marker

    def provider(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {provider_secret}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": object_value(body["messages"][0])["content"]}],
        }, request.body
        return _completion_reply(string_value(object_value(body["messages"][0])["content"]))

    with (
        _config_callback_rig(gateway, tmp_path, monkeypatch, provider, generic_flush_interval="3") as rig,
        rig.first.scenario() as scenario,
    ):
        seed: Final = rig.first.request(
            "POST",
            "/config/update",
            {
                "litellm_settings": {
                    "success_callback": ["langfuse", "generic_api"],
                    "DEFAULT_FLUSH_INTERVAL_SECONDS": 3,
                },
                "environment_variables": {
                    "GENERIC_LOGGER_ENDPOINT": rig.generic.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer integration-generic-sink-secret",
                    "LANGFUSE_HOST": rig.langfuse.url,
                    "LANGFUSE_PUBLIC_KEY": "public-key-" + marker,
                    "LANGFUSE_SECRET_KEY": "secret-key-" + marker,
                },
            },
        )
        assert seed.status_code == 200, seed.text
        assert seed.json() == {"message": "Config updated successfully"}, seed.text
        expected_callbacks: Final[dict[str, dict[str, JsonValue]]] = {
            "generic_api": {
                "name": "generic_api",
                "variables": {
                    "GENERIC_LOGGER_ENDPOINT": rig.generic.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer integration-generic-sink-secret",
                },
                "type": "success",
            },
            "langfuse": {
                "name": "langfuse",
                "variables": {
                    "LANGFUSE_HOST": rig.langfuse.url,
                    "LANGFUSE_PUBLIC_KEY": "public-key-" + marker,
                    "LANGFUSE_SECRET_KEY": "secret-key-" + marker,
                },
                "type": "success",
            },
        }
        callbacks_first_before: Final = rig.first.request("GET", "/get/config/callbacks")
        assert callbacks_first_before.status_code == 200, callbacks_first_before.text
        assert _callback_entries_match(callbacks_first_before, expected_callbacks), callbacks_first_before.text
        callbacks_peer_before: Final = eventually(
            lambda: rig.peer.request("GET", "/get/config/callbacks"),
            lambda response: response.status_code == 200 and _callback_entries_match(response, expected_callbacks),
            seconds=20,
        )
        assert _callback_entries_match(callbacks_peer_before, expected_callbacks), callbacks_peer_before.text
        model: Final = scenario.model(api_base=rig.provider.url + "/v1", api_key=provider_secret)
        api_key: Final = scenario.key(models=[model])

        for worker, proxy in (("first", rig.first), ("peer", rig.peer)):
            positive: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": marker + "-" + worker}], "stream": False},
                key=api_key,
            )
            assert positive.status_code == 200, positive.text

        positive_generic: Final[list[Request]] = []  # mutable-ok: preserve batches while the async sinks flush
        positive_langfuse: Final[list[Request]] = []  # mutable-ok: preserve batches while the async sinks flush

        def collect_positive_generic() -> tuple[Request, ...]:
            positive_generic.extend(rig.generic.drain())
            return tuple(positive_generic)

        def collect_positive_langfuse() -> tuple[Request, ...]:
            positive_langfuse.extend(rig.langfuse.drain())
            return tuple(positive_langfuse)

        positive_markers: Final = (marker + "-first", marker + "-peer")
        generic_before: Final = eventually(
            collect_positive_generic,
            lambda requests: all(
                value.encode() in b"".join(request.body for request in requests) for value in positive_markers
            ),
            seconds=20,
        )
        langfuse_before: Final = eventually(
            collect_positive_langfuse,
            lambda requests: all(
                value.encode() in b"".join(request.body for request in requests) for value in positive_markers
            ),
            seconds=20,
        )
        assert all(
            marker_value.encode() in b"".join(request.body for request in generic_before)
            for marker_value in positive_markers
        )
        assert all(
            marker_value.encode() in b"".join(request.body for request in langfuse_before)
            for marker_value in positive_markers
        )

        deleted: Final = rig.first.request("POST", "/config/callback/delete", {"callback_name": "Langfuse"})
        assert deleted.status_code == 200, deleted.text
        deleted_body: Final = object_value(deleted.json())
        assert set(deleted_body) == {"message", "removed_callback", "remaining_callbacks", "deleted_at"}, deleted.text
        assert deleted_body["message"] == "Successfully deleted callback: langfuse", deleted.text
        assert deleted_body["removed_callback"] == "langfuse", deleted.text
        assert deleted_body["remaining_callbacks"] == ["generic_api"], deleted.text
        datetime.fromisoformat(string_value(deleted_body["deleted_at"]))
        assert _config_section(rig.database_url, "litellm_settings") == {
            "DEFAULT_FLUSH_INTERVAL_SECONDS": 3,
            "success_callback": ["generic_api"],
        }
        expected_remaining_callbacks: Final = {"generic_api": expected_callbacks["generic_api"]}
        callbacks_first: Final = rig.first.request("GET", "/get/config/callbacks")
        assert callbacks_first.status_code == 200, callbacks_first.text
        assert _callback_entries_match(callbacks_first, expected_remaining_callbacks), callbacks_first.text
        assert "langfuse" not in _callback_entries(callbacks_first), callbacks_first.text
        callbacks_peer: Final = eventually(
            lambda: rig.peer.request("GET", "/get/config/callbacks"),
            lambda response: (
                response.status_code == 200
                and _callback_entries_match(response, expected_remaining_callbacks)
                and "langfuse" not in _callback_entries(response)
            ),
            seconds=20,
        )
        assert _callback_entries_match(callbacks_peer, expected_remaining_callbacks), callbacks_peer.text
        assert "langfuse" not in _callback_entries(callbacks_peer), callbacks_peer.text

        audit: Final = eventually(
            lambda: read_rows(
                'SELECT action, table_name, object_id, before_value, updated_values FROM "LiteLLM_AuditLog" '
                "WHERE table_name = %s AND object_id = %s ORDER BY updated_at DESC LIMIT 1",
                ("LiteLLM_Config", "litellm_settings"),
                database_url=rig.database_url,
            ),
            lambda rows: bool(rows),
            seconds=10,
        )
        assert len(audit) == 1, audit
        audit_row: Final = object_value(audit[0])
        assert set(audit_row) == {"action", "table_name", "object_id", "before_value", "updated_values"}, audit_row
        assert audit_row["action"] == "deleted", audit_row
        assert audit_row["table_name"] == "LiteLLM_Config", audit_row
        assert audit_row["object_id"] == "litellm_settings", audit_row
        before_value: Final = object_value(audit_row["before_value"])
        assert set(before_value) == {"success_callback"}, audit_row
        before_callbacks: Final = TypeAdapter(list[str]).validate_python(before_value["success_callback"])
        assert sorted(before_callbacks) == ["generic_api", "langfuse"], audit_row
        assert audit_row["updated_values"] == {"success_callback": ["generic_api"]}, audit_row

        absence_markers: Final = (marker + "-after-first", marker + "-after-peer")
        for worker, proxy, absence_marker in zip(
            ("first", "peer"), (rig.first, rig.peer), absence_markers, strict=True
        ):
            absent: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": absence_marker}], "stream": False},
                key=api_key,
            )
            assert absent.status_code == 200, absent.text

        generic_after: Final[list[Request]] = []  # mutable-ok: preserve batches while the async sink flushes

        def collect_after_generic() -> tuple[Request, ...]:
            generic_after.extend(rig.generic.drain())
            return tuple(generic_after)

        generic_deliveries: Final = eventually(
            collect_after_generic,
            lambda requests: all(
                value.encode() in b"".join(request.body for request in requests) for value in absence_markers
            ),
            seconds=25,
        )
        assert all(
            value.encode() in b"".join(request.body for request in generic_deliveries) for value in absence_markers
        )
        langfuse_after: Final[list[Request]] = []  # mutable-ok: preserve batches while the async sink flushes

        def collect_after_langfuse() -> tuple[Request, ...]:
            langfuse_after.extend(rig.langfuse.drain())
            return tuple(langfuse_after)

        collect_after_langfuse()
        readded: Final = rig.first.request(
            "POST",
            "/config/update",
            {
                "litellm_settings": {"success_callback": ["langfuse"]},
                "environment_variables": {
                    "LANGFUSE_HOST": rig.langfuse.url,
                    "LANGFUSE_PUBLIC_KEY": "public-key-" + marker,
                    "LANGFUSE_SECRET_KEY": "secret-key-" + marker,
                },
            },
        )
        assert readded.status_code == 200, readded.text
        expected_readded_callback: Final = {"langfuse": expected_callbacks["langfuse"]}
        callbacks_peer_readded: Final = eventually(
            lambda: rig.peer.request("GET", "/get/config/callbacks"),
            lambda response: (
                response.status_code == 200 and _callback_entries_match(response, expected_readded_callback)
            ),
            seconds=20,
        )
        assert _callback_entries_match(callbacks_peer_readded, expected_readded_callback), callbacks_peer_readded.text
        barrier_markers: Final = (marker + "-barrier-first", marker + "-barrier-peer")
        for worker, proxy, barrier_marker in zip(
            ("first", "peer"), (rig.first, rig.peer), barrier_markers, strict=True
        ):
            barrier: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": barrier_marker}], "stream": False},
                key=api_key,
            )
            assert barrier.status_code == 200, barrier.text

        langfuse_deliveries: Final = eventually(
            collect_after_langfuse,
            lambda requests: all(
                value.encode() in b"".join(request.body for request in requests) for value in barrier_markers
            ),
            seconds=25,
        )
        langfuse_bodies: Final = b"".join(request.body for request in langfuse_deliveries)
        assert all(value.encode() in langfuse_bodies for value in barrier_markers)
        leaked_absence_markers: Final = tuple(value for value in absence_markers if value.encode() in langfuse_bodies)
        assert leaked_absence_markers == (), leaked_absence_markers

        missing: Final = rig.first.request("POST", "/config/callback/delete", {"callback_name": "not-a-callback"})
        assert missing.status_code == 404, missing.text
        assert missing.json() == {"detail": {"error": "Callback 'not-a-callback' not found in active configuration"}}, (
            missing.text
        )


def test_dashboard_field_updates_preserve_plugin_key_and_apply_at_runtime(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert os.environ.get("LITELLM_LICENSE"), (
        "LITELLM_LICENSE must be set: max_request_size_mb enforcement is an enterprise feature"
    )
    plugin_key: Final = "plugin-key-" + uuid.uuid4().hex
    plugin_body: Final = {"message": "plugin-" + uuid.uuid4().hex}
    passthrough_body: Final = {"message": "passthrough-" + uuid.uuid4().hex}
    passthrough_path: Final = "/integration-pt-" + uuid.uuid4().hex

    def plugin(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/rpc"
        assert request.headers["authorization"] == f"Bearer {plugin_key}"
        assert _JSON_OBJECT.validate_json(request.body) == plugin_body, request.body
        return Reply(body=json.dumps({"received": plugin_body}).encode())

    def passthrough(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/echo"
        assert request.headers["x-integration"] == "yes"
        assert _JSON_OBJECT.validate_json(request.body) == passthrough_body, request.body
        return Reply(body=json.dumps({"received": passthrough_body}).encode())

    with (
        scratch_database() as database_url,
        wire_server(plugin) as plugin_wire,
        wire_server(passthrough) as passthrough_wire,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        environment: Final = {
            "DATABASE_URL": database_url,
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "600",
        }
        with (
            owned_proxy(gateway, tmp_path, environment) as candidate,
            owned_proxy(gateway, tmp_path, environment) as peer,
        ):
            plugin_value: Final = [
                {
                    "name": "integration-plugin",
                    "display_name": "Integration",
                    "url": plugin_wire.url,
                    "plugin_key": plugin_key,
                }
            ]
            plugin_update: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": "plugins", "field_value": plugin_value, "config_type": "general_settings"},
            )
            assert plugin_update.status_code == 200, plugin_update.text
            assert object_value(plugin_update.json())["param_name"] == "general_settings", plugin_update.text
            plugin_info: Final = object_value(candidate.get("/config/field/info?field_name=plugins"))
            assert plugin_info["field_name"] == "plugins"
            assert plugin_info["field_value"] == [
                {
                    "name": "integration-plugin",
                    "display_name": "Integration",
                    "url": plugin_wire.url,
                    "plugin_key": "***",
                }
            ]
            reposted_plugins: Final = [
                {
                    "name": "integration-plugin",
                    "display_name": "Integration updated",
                    "url": plugin_wire.url,
                    "plugin_key": "***",
                }
            ]
            repost: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": "plugins", "field_value": reposted_plugins, "config_type": "general_settings"},
            )
            assert repost.status_code == 200, repost.text
            assert _config_section(database_url, "general_settings") == {
                "plugins": [
                    {
                        "name": "integration-plugin",
                        "display_name": "Integration updated",
                        "url": plugin_wire.url,
                        "plugin_key": plugin_key,
                    }
                ]
            }
            blank_key_plugins: Final = [
                {
                    "name": "integration-plugin",
                    "display_name": "Integration updated",
                    "url": plugin_wire.url,
                    "plugin_key": "",
                }
            ]
            blank_repost: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": "plugins", "field_value": blank_key_plugins, "config_type": "general_settings"},
            )
            assert blank_repost.status_code == 200, blank_repost.text
            assert _config_section(database_url, "general_settings") == {
                "plugins": [
                    {
                        "name": "integration-plugin",
                        "display_name": "Integration updated",
                        "url": plugin_wire.url,
                        "plugin_key": plugin_key,
                    }
                ]
            }
            plugin_response: Final = candidate.request("POST", "/plugin-proxy/integration-plugin/rpc", plugin_body)
            assert plugin_response.status_code == 200, plugin_response.text
            assert plugin_response.json() == {"received": plugin_body}, plugin_response.text
            peer_plugin_response: Final = eventually(
                lambda: peer.request("POST", "/plugin-proxy/integration-plugin/rpc", plugin_body),
                lambda response: response.status_code == 200 and response.json() == {"received": plugin_body},
                seconds=20,
            )
            assert peer_plugin_response.json() == {"received": plugin_body}, peer_plugin_response.text

            passthrough_value: Final = [
                {
                    "path": passthrough_path,
                    "target": passthrough_wire.url + "/echo",
                    "headers": {"x-integration": "yes"},
                    "auth": True,
                }
            ]
            passthrough_update: Final = candidate.request(
                "POST",
                "/config/field/update",
                {
                    "field_name": "pass_through_endpoints",
                    "field_value": passthrough_value,
                    "config_type": "general_settings",
                },
            )
            assert passthrough_update.status_code == 200, passthrough_update.text
            passthrough_response: Final = candidate.request("POST", passthrough_path, passthrough_body)
            assert passthrough_response.status_code == 200, passthrough_response.text
            assert passthrough_response.json() == {"received": passthrough_body}, passthrough_response.text
            peer_passthrough_response: Final = eventually(
                lambda: peer.request("POST", passthrough_path, passthrough_body),
                lambda response: response.status_code == 200 and response.json() == {"received": passthrough_body},
                seconds=20,
            )
            assert peer_passthrough_response.json() == {"received": passthrough_body}, peer_passthrough_response.text

            alerting_args: Final = {
                "daily_report_frequency": 43200,
                "report_check_interval": 300,
                "budget_alert_ttl": 86400,
                "outage_alert_ttl": 60,
                "region_outage_alert_ttl": 60,
                "minor_outage_alert_threshold": 2,
                "major_outage_alert_threshold": 5,
                "max_outage_alert_list_size": 20,
                "daily_spend_per_user_threshold": 25.0,
                "monthly_spend_per_user_threshold": 500.0,
                "spend_anomaly_multiplier": 3.5,
                "spend_anomaly_baseline_days": 14,
                "spend_anomaly_min_spend": 12.0,
                "user_spend_check_interval": 3600,
            }
            alerting_update: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": "alerting_args", "field_value": alerting_args, "config_type": "general_settings"},
            )
            assert alerting_update.status_code == 200, alerting_update.text
            assert _config_section(database_url, "general_settings") == {
                "plugins": [
                    {
                        "name": "integration-plugin",
                        "display_name": "Integration updated",
                        "url": plugin_wire.url,
                        "plugin_key": plugin_key,
                    }
                ],
                "pass_through_endpoints": passthrough_value,
                "alerting_args": alerting_args,
            }
            alerting_info: Final = object_value(candidate.get("/config/field/info?field_name=alerting_args"))
            assert alerting_info["field_value"] == alerting_args

            invalid: Final = candidate.request(
                "POST",
                "/config/field/update",
                {
                    "field_name": "alerting_args",
                    "field_value": {"daily_spend_per_user_threshold": -1},
                    "config_type": "general_settings",
                },
            )
            assert invalid.status_code == 400, invalid.text
            assert invalid.json() == {
                "detail": {
                    "error": "Invalid alerting_args: daily_spend_per_user_threshold: Input should be greater than 0"
                }
            }, invalid.text

            max_size_update: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": "max_request_size_mb", "field_value": 1, "config_type": "general_settings"},
            )
            assert max_size_update.status_code == 200, max_size_update.text
            expected_settings: Final = {
                "plugins": [
                    {
                        "name": "integration-plugin",
                        "display_name": "Integration updated",
                        "url": plugin_wire.url,
                        "plugin_key": plugin_key,
                    }
                ],
                "pass_through_endpoints": passthrough_value,
                "alerting_args": alerting_args,
                "max_request_size_mb": 1,
            }
            assert _config_section(database_url, "general_settings") == expected_settings
            max_size_info: Final = object_value(candidate.get("/config/field/info?field_name=max_request_size_mb"))
            assert max_size_info["field_value"] == 1
            list_after_size: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_json(
                candidate.request("GET", "/config/list", params={"config_type": "general_settings"}).content
            )
            max_size_listed: Final = next(
                entry for entry in list_after_size if entry["field_name"] == "max_request_size_mb"
            )
            assert max_size_listed["field_value"] == 1 and max_size_listed["stored_in_db"] is True
            oversized: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "x" * (1024 * 1024 + 1)}],
                },
            )
            assert oversized.status_code == 413, oversized.text
            assert oversized.text == '{"error":"Request size is too large. Max size is 1 MB"}'
            peer_oversized: Final = eventually(
                lambda: peer.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": "x" * (1024 * 1024 + 1)}],
                    },
                ),
                lambda response: (
                    response.status_code == 413
                    and response.text == '{"error":"Request size is too large. Max size is 1 MB"}'
                ),
                seconds=20,
            )
            assert peer_oversized.text == '{"error":"Request size is too large. Max size is 1 MB"}', peer_oversized.text

            plugin_requests: Final = plugin_wire.drain()
            assert tuple((request.method, request.target) for request in plugin_requests) == (
                ("POST", "/rpc"),
                ("POST", "/rpc"),
            ), plugin_requests
            passthrough_requests: Final = passthrough_wire.drain()
            assert tuple((request.method, request.target) for request in passthrough_requests) == (
                ("POST", "/echo"),
                ("POST", "/echo"),
            ), passthrough_requests


@pytest.mark.covers("mgmt.model.block.changes_serving_and_preserves_control")
def test_model_block_changes_actual_route_and_leaves_other_route_working(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        identity: Final = model_identity(gateway, model)
        gateway.chat(model)
        gateway.chat(other)
        gateway.post("/model/block", {"model_id": identity})
        assert read_rows('SELECT blocked FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)) == [
            {"blocked": True}
        ]
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "blocked deployment"}]},
        )
        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "permission_error"
        assert response.json()["error"]["message"] == "litellm.PermissionDeniedError: Model is blocked"
        assert object_value(gateway.chat(other)["usage"])["total_tokens"] == 40
        gateway.post("/model/unblock", {"model_id": identity})
        assert read_rows('SELECT blocked FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)) == [
            {"blocked": False}
        ]
        assert object_value(gateway.chat(model)["usage"])["total_tokens"] == 40


@pytest.mark.covers("mgmt.router_settings.update.changes_observed_attempt_count")
def test_saved_retry_setting_controls_real_attempts_and_restores(gateway: Gateway) -> None:
    with (
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        gateway.scenario() as scenario,
    ):
        original: Final = object_value(gateway.get("/router/settings")["current_values"])["num_retries"]
        provider_model: Final = f"retry-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}", input_cost_per_token=0, output_cost_per_token=0)

        def remove_script() -> None:
            response: Final = upstream.delete(f"/__scripts/{provider_model}")
            assert response.status_code in (200, 404), response.text
            assert upstream.get(f"/__scripts/{provider_model}").status_code == 404

        scenario.cleanups.callback(remove_script)
        try:
            for generation, retries in enumerate((0, 1, original)):
                gateway.post("/config/update", {"router_settings": {"num_retries": retries}})
                assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == retries
                configured: Final = upstream.post(f"/__scripts/{provider_model}", json={"statuses": [500, 200]})
                assert configured.status_code == 200, configured.text
                upstream.get("/__observations").raise_for_status()
                response: Final = gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": f"{provider_model} attempt {generation}"}],
                    },
                )
                observed: Final = upstream.get("/__observations")
                observed.raise_for_status()
                requests: Final = observed.json()["requests"]
                assert len(requests) == (1 if retries == 0 else 2), (response.status_code, response.text, requests)
                assert all(value["body"]["model"] == provider_model for value in requests)
                assert response.status_code == (500 if retries == 0 else 200), response.text
                if retries != 0:
                    assert response.json()["usage"]["total_tokens"] == 40
                remaining: Final = upstream.delete(f"/__scripts/{provider_model}")
                assert remaining.status_code == 200, remaining.text
                assert remaining.json()["remaining"] == ([200] if retries == 0 else [])
        finally:
            gateway.post("/config/update", {"router_settings": {"num_retries": original}})
            assert object_value(gateway.get("/router/settings")["current_values"])["num_retries"] == original


@pytest.mark.covers("mgmt.credential.update.saved_value_reaches_wire")
def test_credential_value_update_and_model_reload_reach_provider(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        name: Final = f"credential-{uuid.uuid4().hex}"
        gateway.post(
            "/credentials",
            {
                "credential_name": name,
                "credential_values": {"api_key": "synthetic-credential-first"},
                "credential_info": {},
            },
        )

        def remove_credential() -> None:
            response: Final = gateway.request("DELETE", f"/credentials/{name}")
            assert response.status_code == 200, response.text
            assert (
                read_rows('SELECT credential_name FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,))
                == []
            )

        scenario.cleanups.callback(remove_credential)
        model: Final = scenario.model(api_key=None, litellm_credential_name=name)
        identity: Final = model_identity(gateway, model)
        for value in ("synthetic-credential-first", "synthetic-credential-second"):
            patched: Final = gateway.request(
                "PATCH",
                f"/credentials/{name}",
                {"credential_name": name, "credential_values": {"api_key": value}, "credential_info": {}},
            )
            assert patched.status_code == 200, patched.text
            rows: Final = read_rows(
                'SELECT credential_values FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,)
            )
            assert len(rows) == 1
            stored: Final = object_value(rows[0]["credential_values"])
            assert isinstance(stored["api_key"], str) and stored["api_key"] != value
            for reload in (False, True):
                if reload:
                    response: Final = gateway.request(
                        "PATCH", f"/model/{identity}/update", {"model_info": {"description": value}}
                    )
                    assert response.status_code == 200, response.text
                upstream.get("/__observations").raise_for_status()
                assert (
                    object_value(gateway.chat(model, text=f"{name} {value} reload={reload}")["usage"])["total_tokens"]
                    == 40
                )
                observed: Final = upstream.get("/__observations")
                observed.raise_for_status()
                assert len(observed.json()["requests"]) == 1, (value, reload, observed.text)
                assert observed.json()["requests"][0]["authorization"] == f"Bearer {value}"

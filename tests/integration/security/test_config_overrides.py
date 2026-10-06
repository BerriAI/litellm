from __future__ import annotations

import base64
import json
import ssl
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import quote

import httpx
import pytest
from integration._support.client import Gateway, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.tls import write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_VAULT_ENVIRONMENT: Final = (
    "HCP_VAULT_ADDR",
    "HCP_VAULT_TOKEN",
    "HCP_VAULT_APPROLE_ROLE_ID",
    "HCP_VAULT_APPROLE_SECRET_ID",
    "HCP_VAULT_APPROLE_MOUNT_PATH",
    "HCP_VAULT_CLIENT_CERT",
    "HCP_VAULT_CLIENT_KEY",
    "HCP_VAULT_CERT_ROLE",
    "HCP_VAULT_NAMESPACE",
    "HCP_VAULT_LOGIN_NAMESPACE",
    "HCP_VAULT_SECRET_NAMESPACE",
    "HCP_VAULT_MOUNT_NAME",
    "HCP_VAULT_PATH_PREFIX",
)
_CYBERARK_ENVIRONMENT: Final = (
    "CYBERARK_API_BASE",
    "CYBERARK_ACCOUNT",
    "CYBERARK_USERNAME",
    "CYBERARK_API_KEY",
    "CYBERARK_CLIENT_CERT",
    "CYBERARK_CLIENT_KEY",
    "CYBERARK_SSL_VERIFY",
    "CYBERARK_REFRESH_INTERVAL",
)
_SECRET_MANAGER_ENVIRONMENT: Final = (*_VAULT_ENVIRONMENT, *_CYBERARK_ENVIRONMENT)


@contextmanager
def _config_proxy(
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: Mapping[str, str] | None = None,
) -> Iterator[tuple[Gateway, str]]:
    with scratch_database() as database_url:
        with monkeypatch.context() as scoped_environment:
            scoped_environment.setenv("DATABASE_URL", database_url)
            with owned_proxy(
                gateway,
                tmp_path,
                {"DATABASE_URL": database_url, **(overrides or {})},
                remove_environment=_SECRET_MANAGER_ENVIRONMENT,
            ) as candidate:
                yield candidate, database_url


def _override_values(candidate: Gateway, path: str, config_type: str) -> dict[str, JsonValue]:
    response: Final = candidate.request("GET", path)
    assert response.status_code == 200, response.text
    body: Final = object_value(_JSON_OBJECT.validate_json(response.content))
    assert body["config_type"] == config_type, response.text
    return object_value(body["values"])


def _stored_override(database_url: str, config_type: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT config_value FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
        (config_type,),
        database_url=database_url,
    )
    assert len(rows) == 1, rows
    config_value: Final = rows[0]["config_value"]
    return _JSON_OBJECT.validate_json(config_value) if isinstance(config_value, str) else object_value(config_value)


def _masked(value: str) -> str:
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def _assert_non_admin_refusal(response: httpx.Response, route: str, user_id: str) -> None:
    assert response.status_code == 401, response.text
    masked_user_id: Final = user_id[:6] + "*" * (len(user_id) - 8) + user_id[-2:]
    assert response.json() == {
        "error": {
            "message": (
                "Authentication Error, Only proxy admin can be used to generate, delete, update info for "
                f"new keys/users/teams. Route={route}. Your role=internal_user. Your user_id={masked_user_id}"
            ),
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }, response.text


def _completion(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
    )


def _new_model(candidate: Gateway, name: str, provider: str, secret_name: str) -> str:
    response: Final = candidate.request(
        "POST",
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": f"os.environ/{secret_name}",
                "api_base": provider + "/v1",
            },
            "model_info": {},
        },
    )
    assert response.status_code == 200, response.text
    return string_value(object_value(response.json()["model_info"])["id"])


def _chat(candidate: Gateway, model: str, marker: str) -> None:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == marker, response.text


def test_saved_vault_config_reads_back_masked_and_refuses_non_admins(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def vault_response(request: Request) -> Reply:
        return Reply()

    with wire_server(vault_response) as vault:
        config: Final = {
            "vault_addr": vault.url,
            "vault_token": "vault-token-" + uuid.uuid4().hex,
            "approle_role_id": "approle-role-" + uuid.uuid4().hex,
            "approle_secret_id": "approle-secret-" + uuid.uuid4().hex,
            "approle_mount_path": "auth/approle-integration",
            "client_cert": "/integration/client.crt",
            "client_key": "client-key-" + uuid.uuid4().hex,
            "vault_cert_role": "integration-role",
            "vault_namespace": "integration/root",
            "vault_login_namespace": "integration/login",
            "vault_secret_namespace": "integration/secrets",
            "vault_mount_name": "integration-kv",
            "vault_path_prefix": "integration/prefix",
        }
        with _config_proxy(gateway, tmp_path, monkeypatch) as (candidate, database_url):
            saved: Final = candidate.request("POST", "/config_overrides/hashicorp_vault", config)
            assert saved.status_code == 200, saved.text
            assert saved.json() == {
                "message": "Hashicorp Vault configuration updated successfully",
                "status": "success",
            }, saved.text
            readback: Final = candidate.request("GET", "/config_overrides/hashicorp_vault")
            assert readback.status_code == 200, readback.text
            readback_body: Final = object_value(_JSON_OBJECT.validate_json(readback.content))
            assert readback_body["config_type"] == "hashicorp_vault", readback.text
            values: Final = object_value(readback_body["values"])
            expected_values: Final = {
                **config,
                "vault_token": _masked(string_value(config["vault_token"])),
                "approle_secret_id": _masked(string_value(config["approle_secret_id"])),
                "client_key": _masked(string_value(config["client_key"])),
            }
            assert values == expected_values
            assert all(
                string_value(config[field]).encode() not in readback.content
                for field in ("vault_token", "approle_secret_id", "client_key")
            )

            with candidate.scenario() as scenario:
                non_admin_user_id: Final = "integration-vault-non-admin"
                non_admin: Final = scenario.key(
                    user_id=scenario.user(user_id=non_admin_user_id, user_role="internal_user")
                )
                denied_get: Final = candidate.request("GET", "/config_overrides/hashicorp_vault", key=non_admin)
                _assert_non_admin_refusal(denied_get, "/config_overrides/hashicorp_vault", non_admin_user_id)
                denied_post: Final = candidate.request(
                    "POST", "/config_overrides/hashicorp_vault", config, key=non_admin
                )
                _assert_non_admin_refusal(denied_post, "/config_overrides/hashicorp_vault", non_admin_user_id)

                stored_before_scoped: Final = _stored_override(database_url, "hashicorp_vault")
                scoped_route_key: Final = scenario.key(
                    user_id=scenario.user(user_role="internal_user"),
                    allowed_routes=["/config_overrides/hashicorp_vault"],
                )
                scoped_get: Final = candidate.request("GET", "/config_overrides/hashicorp_vault", key=scoped_route_key)
                assert scoped_get.status_code == 403, scoped_get.text
                assert scoped_get.json() == {"detail": "Only admin users can view config overrides"}, scoped_get.text
                scoped_post: Final = candidate.request(
                    "POST", "/config_overrides/hashicorp_vault", config, key=scoped_route_key
                )
                assert scoped_post.status_code == 403, scoped_post.text
                assert scoped_post.json() == {"detail": "Only admin users can update config overrides"}, (
                    scoped_post.text
                )
                assert _stored_override(database_url, "hashicorp_vault") == stored_before_scoped

                viewer_key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin_viewer"))
                viewer_get: Final = candidate.request("GET", "/config_overrides/hashicorp_vault", key=viewer_key)
                assert viewer_get.status_code == 200, viewer_get.text
                viewer_values: Final = object_value(
                    object_value(_JSON_OBJECT.validate_json(viewer_get.content))["values"]
                )
                assert viewer_values == expected_values, viewer_get.text
                assert all(
                    string_value(config[field]).encode() not in viewer_get.content
                    for field in ("vault_token", "approle_secret_id", "client_key")
                )
                viewer_post: Final = candidate.request(
                    "POST", "/config_overrides/hashicorp_vault", config, key=viewer_key
                )
                assert viewer_post.status_code == 403, viewer_post.text
                assert viewer_post.json() == {
                    "error": {
                        "message": "user not allowed to access this route, role= proxy_admin_viewer. "
                        "Trying to access: /config_overrides/hashicorp_vault",
                        "type": "auth_error",
                        "param": "None",
                        "code": "403",
                    }
                }, viewer_post.text
                assert _stored_override(database_url, "hashicorp_vault") == stored_before_scoped

        env_values: Final = {
            "HCP_VAULT_ADDR": vault.url,
            "HCP_VAULT_TOKEN": "env-vault-token-" + uuid.uuid4().hex,
            "HCP_VAULT_APPROLE_ROLE_ID": "env-role-" + uuid.uuid4().hex,
            "HCP_VAULT_APPROLE_SECRET_ID": "env-secret-" + uuid.uuid4().hex,
            "HCP_VAULT_APPROLE_MOUNT_PATH": "auth/env-approle",
            "HCP_VAULT_CLIENT_CERT": "/integration/env-client.crt",
            "HCP_VAULT_CLIENT_KEY": "env-client-key-" + uuid.uuid4().hex,
            "HCP_VAULT_CERT_ROLE": "env-integration-role",
            "HCP_VAULT_NAMESPACE": "env/root",
            "HCP_VAULT_LOGIN_NAMESPACE": "env/login",
            "HCP_VAULT_SECRET_NAMESPACE": "env/secrets",
            "HCP_VAULT_MOUNT_NAME": "env-kv",
            "HCP_VAULT_PATH_PREFIX": "env/prefix",
        }
        env_config: Final = {
            "vault_addr": env_values["HCP_VAULT_ADDR"],
            "vault_token": env_values["HCP_VAULT_TOKEN"],
            "approle_role_id": env_values["HCP_VAULT_APPROLE_ROLE_ID"],
            "approle_secret_id": env_values["HCP_VAULT_APPROLE_SECRET_ID"],
            "approle_mount_path": env_values["HCP_VAULT_APPROLE_MOUNT_PATH"],
            "client_cert": env_values["HCP_VAULT_CLIENT_CERT"],
            "client_key": env_values["HCP_VAULT_CLIENT_KEY"],
            "vault_cert_role": env_values["HCP_VAULT_CERT_ROLE"],
            "vault_namespace": env_values["HCP_VAULT_NAMESPACE"],
            "vault_login_namespace": env_values["HCP_VAULT_LOGIN_NAMESPACE"],
            "vault_secret_namespace": env_values["HCP_VAULT_SECRET_NAMESPACE"],
            "vault_mount_name": env_values["HCP_VAULT_MOUNT_NAME"],
            "vault_path_prefix": env_values["HCP_VAULT_PATH_PREFIX"],
        }
        with _config_proxy(gateway, tmp_path, monkeypatch, env_values) as (candidate, database_url):
            assert (
                read_rows(
                    'SELECT config_type FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
                    ("hashicorp_vault",),
                    database_url=database_url,
                )
                == []
            )
            env_response: Final = candidate.request("GET", "/config_overrides/hashicorp_vault")
            assert env_response.status_code == 200, env_response.text
            env_body: Final = object_value(_JSON_OBJECT.validate_json(env_response.content))
            assert env_body["config_type"] == "hashicorp_vault", env_response.text
            env_readback: Final = object_value(env_body["values"])
            assert env_readback == {
                **env_config,
                "vault_token": _masked(string_value(env_config["vault_token"])),
                "approle_secret_id": _masked(string_value(env_config["approle_secret_id"])),
                "client_key": _masked(string_value(env_config["client_key"])),
            }
            assert all(
                string_value(env_config[field]).encode() not in env_response.content
                for field in ("vault_token", "approle_secret_id", "client_key")
            )


def test_saved_cyberark_config_reads_back_masked_and_refuses_non_admins(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with wire_server(lambda request: Reply()) as cyberark:
        config: Final = {
            "cyberark_api_base": cyberark.url,
            "cyberark_account": "integration-account",
            "cyberark_username": "integration-user",
            "cyberark_api_key": "cyberark-api-key-" + uuid.uuid4().hex,
            "client_cert": "/integration/client.crt",
            "client_key": "cyberark-client-key-" + uuid.uuid4().hex,
            "ssl_verify": "true",
            "refresh_interval": "300",
        }
        with _config_proxy(gateway, tmp_path, monkeypatch) as (candidate, _database_url):
            saved: Final = candidate.request("POST", "/config_overrides/cyberark", config)
            assert saved.status_code == 200, saved.text
            assert saved.json() == {
                "message": "CyberArk configuration updated successfully",
                "status": "success",
            }, saved.text
            readback: Final = candidate.request("GET", "/config_overrides/cyberark")
            assert readback.status_code == 200, readback.text
            readback_body: Final = object_value(_JSON_OBJECT.validate_json(readback.content))
            assert readback_body["config_type"] == "cyberark", readback.text
            values: Final = object_value(readback_body["values"])
            assert values == {
                **config,
                "cyberark_api_key": _masked(string_value(config["cyberark_api_key"])),
                "client_key": _masked(string_value(config["client_key"])),
            }
            assert all(
                string_value(config[field]).encode() not in readback.content
                for field in ("cyberark_api_key", "client_key")
            )
            with candidate.scenario() as scenario:
                non_admin_user_id: Final = "integration-cyberark-non-admin"
                non_admin: Final = scenario.key(
                    user_id=scenario.user(user_id=non_admin_user_id, user_role="internal_user")
                )
                denied_get: Final = candidate.request("GET", "/config_overrides/cyberark", key=non_admin)
                _assert_non_admin_refusal(denied_get, "/config_overrides/cyberark", non_admin_user_id)
                denied_post: Final = candidate.request("POST", "/config_overrides/cyberark", config, key=non_admin)
                _assert_non_admin_refusal(denied_post, "/config_overrides/cyberark", non_admin_user_id)

                stored_before_scoped: Final = _stored_override(_database_url, "cyberark")
                scoped_route_key: Final = scenario.key(
                    user_id=scenario.user(user_role="internal_user"),
                    allowed_routes=["/config_overrides/cyberark"],
                )
                scoped_get: Final = candidate.request("GET", "/config_overrides/cyberark", key=scoped_route_key)
                assert scoped_get.status_code == 403, scoped_get.text
                assert scoped_get.json() == {"detail": "Only admin users can view config overrides"}, scoped_get.text
                scoped_post: Final = candidate.request(
                    "POST", "/config_overrides/cyberark", config, key=scoped_route_key
                )
                assert scoped_post.status_code == 403, scoped_post.text
                assert scoped_post.json() == {"detail": "Only admin users can update config overrides"}, (
                    scoped_post.text
                )
                assert _stored_override(_database_url, "cyberark") == stored_before_scoped

                viewer_key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin_viewer"))
                viewer_get: Final = candidate.request("GET", "/config_overrides/cyberark", key=viewer_key)
                assert viewer_get.status_code == 200, viewer_get.text
                viewer_values: Final = object_value(
                    object_value(_JSON_OBJECT.validate_json(viewer_get.content))["values"]
                )
                assert viewer_values == values, viewer_get.text
                assert all(
                    string_value(config[field]).encode() not in viewer_get.content
                    for field in ("cyberark_api_key", "client_key")
                )
                viewer_post: Final = candidate.request("POST", "/config_overrides/cyberark", config, key=viewer_key)
                assert viewer_post.status_code == 403, viewer_post.text
                assert viewer_post.json() == {
                    "error": {
                        "message": "user not allowed to access this route, role= proxy_admin_viewer. "
                        "Trying to access: /config_overrides/cyberark",
                        "type": "auth_error",
                        "param": "None",
                        "code": "403",
                    }
                }, viewer_post.text
                assert _stored_override(_database_url, "cyberark") == stored_before_scoped

        env_values: Final = {
            "CYBERARK_API_BASE": cyberark.url,
            "CYBERARK_ACCOUNT": "env-account",
            "CYBERARK_USERNAME": "env-user",
            "CYBERARK_API_KEY": "env-cyberark-api-key-" + uuid.uuid4().hex,
            "CYBERARK_CLIENT_CERT": "/integration/env-client.crt",
            "CYBERARK_CLIENT_KEY": "env-cyberark-client-key-" + uuid.uuid4().hex,
            "CYBERARK_SSL_VERIFY": "false",
            "CYBERARK_REFRESH_INTERVAL": "600",
        }
        env_config: Final = {
            "cyberark_api_base": env_values["CYBERARK_API_BASE"],
            "cyberark_account": env_values["CYBERARK_ACCOUNT"],
            "cyberark_username": env_values["CYBERARK_USERNAME"],
            "cyberark_api_key": env_values["CYBERARK_API_KEY"],
            "client_cert": env_values["CYBERARK_CLIENT_CERT"],
            "client_key": env_values["CYBERARK_CLIENT_KEY"],
            "ssl_verify": env_values["CYBERARK_SSL_VERIFY"],
            "refresh_interval": env_values["CYBERARK_REFRESH_INTERVAL"],
        }
        with _config_proxy(gateway, tmp_path, monkeypatch, env_values) as (env_candidate, env_database_url):
            assert (
                read_rows(
                    'SELECT config_type FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
                    ("cyberark",),
                    database_url=env_database_url,
                )
                == []
            )
            env_response: Final = env_candidate.request("GET", "/config_overrides/cyberark")
            assert env_response.status_code == 200, env_response.text
            env_body: Final = object_value(_JSON_OBJECT.validate_json(env_response.content))
            assert env_body["config_type"] == "cyberark", env_response.text
            assert object_value(env_body["values"]) == {
                **env_config,
                "cyberark_api_key": _masked(env_config["cyberark_api_key"]),
                "client_key": _masked(env_config["client_key"]),
            }, env_response.text
            assert all(
                env_config[field].encode() not in env_response.content for field in ("cyberark_api_key", "client_key")
            )


def test_vault_partial_update_keeps_omitted_fields_encrypts_and_drives_secret_reads(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_secret: Final = "vault-provider-secret-" + uuid.uuid4().hex
    vault_token: Final = "vault-token-" + uuid.uuid4().hex
    marker_one: Final = "vault-runtime-one-" + uuid.uuid4().hex
    marker_two: Final = "vault-runtime-two-" + uuid.uuid4().hex
    path_one: Final = "/v1/integration/team/kv-original/data/initial-prefix/secret-one"
    path_two: Final = "/v1/integration/team/kv-updated/data/secret-two"

    def vault(request: Request) -> Reply:
        assert request.method == "GET"
        assert request.headers["x-vault-token"] == vault_token
        secrets: Final = {
            path_one: provider_secret,
            path_two: provider_secret,
        }
        if request.target not in secrets:
            return Reply(status=404, body=b'{"errors":["secret not found"]}')
        return Reply(body=json.dumps({"data": {"data": {"key": secrets[request.target]}}}).encode())

    def provider(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        messages: Final = body["messages"]
        assert isinstance(messages, list)
        message: Final = object_value(messages[0])
        marker: Final = string_value(message["content"])
        if marker == "after-delete":
            return _completion(marker)
        assert request.headers["authorization"] == f"Bearer {provider_secret}", (
            "Provider request did not use the Vault-resolved secret"
        )
        assert marker in {marker_one, marker_two}, request.body
        return _completion(marker)

    with wire_server(vault) as vault_wire, wire_server(provider) as provider_wire:
        with _config_proxy(gateway, tmp_path, monkeypatch) as (candidate, database_url):
            full_config: Final = {
                "vault_addr": vault_wire.url,
                "vault_token": vault_token,
                "vault_secret_namespace": "integration/team",
                "vault_mount_name": "kv-original",
                "vault_path_prefix": "initial-prefix",
            }
            full_update: Final = candidate.request("POST", "/config_overrides/hashicorp_vault", full_config)
            assert full_update.status_code == 200, full_update.text
            assert full_update.json() == {
                "message": "Hashicorp Vault configuration updated successfully",
                "status": "success",
            }, full_update.text
            stored_full: Final = _stored_override(database_url, "hashicorp_vault")
            assert set(stored_full) == set(full_config), stored_full
            serialized_full: Final = json.dumps(stored_full)
            assert all(value not in serialized_full for value in full_config.values()), serialized_full

            with candidate.scenario() as scenario:
                model_one: Final = scenario.model(
                    api_key="os.environ/secret-one",
                    api_base=provider_wire.url + "/v1",
                )
                response_one: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model_one, "messages": [{"role": "user", "content": marker_one}]},
                )
                assert response_one.status_code == 200, response_one.text
                assert response_one.json()["choices"][0]["message"]["content"] == marker_one, response_one.text
                first_secret_reads: Final = tuple(
                    (request.method, request.target)
                    for request in vault_wire.drain()
                    if request.target in {path_one, path_two}
                )
                assert first_secret_reads == (("GET", path_one),)

                partial_config: Final = {
                    "vault_addr": vault_wire.url,
                    "vault_secret_namespace": "integration/team",
                    "vault_mount_name": "kv-updated",
                    "vault_path_prefix": "",
                }
                partial_update: Final = candidate.request("POST", "/config_overrides/hashicorp_vault", partial_config)
                assert partial_update.status_code == 200, partial_update.text
                assert partial_update.json() == {
                    "message": "Hashicorp Vault configuration updated successfully",
                    "status": "success",
                }, partial_update.text
                stored_partial: Final = _stored_override(database_url, "hashicorp_vault")
                assert set(stored_partial) == {
                    "vault_addr",
                    "vault_token",
                    "vault_secret_namespace",
                    "vault_mount_name",
                }, stored_partial
                serialized_partial: Final = json.dumps(stored_partial)
                assert all(
                    value not in serialized_partial
                    for value in (
                        vault_wire.url,
                        vault_token,
                        "integration/team",
                        "kv-updated",
                    )
                ), serialized_partial
                readback: Final = _override_values(candidate, "/config_overrides/hashicorp_vault", "hashicorp_vault")
                assert readback == {
                    "vault_addr": vault_wire.url,
                    "vault_token": _masked(vault_token),
                    "vault_secret_namespace": "integration/team",
                    "vault_mount_name": "kv-updated",
                }
                readback_response: Final = candidate.request("GET", "/config_overrides/hashicorp_vault")
                assert readback_response.status_code == 200, readback_response.text
                assert vault_token not in readback_response.text, readback_response.text

                model_two: Final = scenario.model(
                    api_key="os.environ/secret-two",
                    api_base=provider_wire.url + "/v1",
                )
                response_two: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model_two, "messages": [{"role": "user", "content": marker_two}]},
                )
                assert response_two.status_code == 200, response_two.text
                assert response_two.json()["choices"][0]["message"]["content"] == marker_two, response_two.text
                second_secret_reads: Final = tuple(
                    (request.method, request.target)
                    for request in vault_wire.drain()
                    if request.target in {path_one, path_two}
                )
                assert second_secret_reads == (("GET", path_two),)
                provider_requests: Final = provider_wire.drain()
                assert tuple((request.method, request.target) for request in provider_requests) == (
                    ("POST", "/v1/chat/completions"),
                    ("POST", "/v1/chat/completions"),
                )

                deleted: Final = candidate.request("DELETE", "/config_overrides/hashicorp_vault")
                assert deleted.status_code == 200, deleted.text
                assert deleted.json() == {
                    "message": "Hashicorp Vault configuration deleted successfully",
                    "status": "success",
                }, deleted.text
                vault_wire.drain()
                assert (
                    read_rows(
                        'SELECT config_type FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
                        ("hashicorp_vault",),
                        database_url=database_url,
                    )
                    == []
                )
                empty_readback: Final = candidate.request("GET", "/config_overrides/hashicorp_vault")
                assert empty_readback.status_code == 200, empty_readback.text
                empty_values: Final = object_value(
                    object_value(_JSON_OBJECT.validate_json(empty_readback.content))["values"]
                )
                assert empty_values == {
                    field: None
                    for field in (
                        "vault_addr",
                        "vault_token",
                        "approle_role_id",
                        "approle_secret_id",
                        "approle_mount_path",
                        "client_cert",
                        "client_key",
                        "vault_cert_role",
                        "vault_namespace",
                        "vault_login_namespace",
                        "vault_secret_namespace",
                        "vault_mount_name",
                        "vault_path_prefix",
                    )
                }, empty_readback.text
                model_after_delete: Final = scenario.model(
                    api_key="os.environ/LITELLM_MASTER_KEY",
                    api_base=provider_wire.url + "/v1",
                )
                response_after_delete: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model_after_delete,
                        "messages": [{"role": "user", "content": "after-delete"}],
                    },
                )
                assert response_after_delete.status_code == 200, response_after_delete.text
                assert response_after_delete.json()["choices"][0]["message"]["content"] == "after-delete", (
                    response_after_delete.text
                )
                after_delete_requests: Final = vault_wire.drain()
                assert after_delete_requests == (), after_delete_requests
                assert first_secret_reads + second_secret_reads == (
                    ("GET", path_one),
                    ("GET", path_two),
                )


def test_cyberark_incremental_update_encrypts_reloads_on_restart_and_delete_clears(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_secret: Final = "cyberark-provider-secret-" + uuid.uuid4().hex
    api_key: Final = "cyberark-api-key-" + uuid.uuid4().hex
    session_token: Final = "integration-session-" + uuid.uuid4().hex
    account: Final = "integration-account"
    username: Final = "integration-user"
    auth_path: Final = f"/authn/{account}/{username}/authenticate"
    secret_names: Final = (
        "integration-cyberark-secret-one",
        "integration-cyberark-secret-two",
        "integration-cyberark-secret-three",
    )
    encoded_secret_names: Final = frozenset(quote(name, safe="") for name in secret_names)

    def cyberark(request: Request) -> Reply:
        if request.method == "POST" and request.target == auth_path:
            assert request.body == api_key.encode()
            return Reply(body=session_token.encode())
        expected_target: Final = "/secrets/integration-account/variable/"
        assert request.method == "GET"
        assert request.target.startswith(expected_target), request.target
        encoded_name: Final = request.target.removeprefix(expected_target)
        expected_header: Final = f'Token token="{base64.b64encode(session_token.encode()).decode()}"'
        assert request.headers["authorization"] == expected_header
        if encoded_name not in encoded_secret_names:
            return Reply(status=404, body=b"secret not found")
        return Reply(body=provider_secret.encode())

    def provider(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/v1/models", request.target
            return Reply(status=404)
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), (
            f"Unexpected provider request {request.method} {request.target}"
        )
        body: Final = _JSON_OBJECT.validate_json(request.body)
        messages: Final = body["messages"]
        assert isinstance(messages, list)
        marker: Final = string_value(object_value(messages[0])["content"])
        if marker == "after-delete":
            return _completion(marker)
        assert request.headers["authorization"] == f"Bearer {provider_secret}", (
            "Provider request did not use the CyberArk-resolved secret"
        )
        return _completion(marker)

    with (
        scratch_database() as database_url,
        wire_server(cyberark) as cyberark_wire,
        wire_server(provider) as provider_wire,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        environment: Final = {"DATABASE_URL": database_url}
        with owned_proxy(
            gateway,
            tmp_path,
            environment,
            remove_environment=_SECRET_MANAGER_ENVIRONMENT,
        ) as first:
            full_config: Final = {
                "cyberark_api_base": cyberark_wire.url,
                "cyberark_account": account,
                "cyberark_username": username,
                "cyberark_api_key": api_key,
                "ssl_verify": "true",
                "refresh_interval": "300",
            }
            saved: Final = first.request("POST", "/config_overrides/cyberark", full_config)
            assert saved.status_code == 200, saved.text
            assert saved.json() == {
                "message": "CyberArk configuration updated successfully",
                "status": "success",
            }, saved.text
            stored_full: Final = _stored_override(database_url, "cyberark")
            assert set(stored_full) == set(full_config), stored_full
            serialized_full: Final = json.dumps(stored_full)
            assert all(value not in serialized_full for value in full_config.values()), serialized_full

            model_one: Final = "integration-cyberark-" + uuid.uuid4().hex
            _new_model(first, model_one, provider_wire.url, secret_names[0])
            _chat(first, model_one, "cyberark-runtime-one-" + uuid.uuid4().hex)
            first_proxy_requests: Final = cyberark_wire.drain()
            assert any(request.target == auth_path for request in first_proxy_requests)
            assert any(request.target.endswith(quote(secret_names[0], safe="")) for request in first_proxy_requests)

            partial: Final = first.request(
                "POST",
                "/config_overrides/cyberark",
                {"cyberark_api_base": cyberark_wire.url, "refresh_interval": "301"},
            )
            assert partial.status_code == 200, partial.text
            assert partial.json() == {
                "message": "CyberArk configuration updated successfully",
                "status": "success",
            }, partial.text
            stored_partial: Final = _stored_override(database_url, "cyberark")
            assert set(stored_partial) == set(full_config), stored_partial
            serialized_partial: Final = json.dumps(stored_partial)
            assert all(value not in serialized_partial for value in (cyberark_wire.url, account, username, api_key)), (
                serialized_partial
            )
            assert string_value(
                _override_values(first, "/config_overrides/cyberark", "cyberark")["cyberark_api_key"]
            ) == _masked(api_key)
            model_two: Final = "integration-cyberark-" + uuid.uuid4().hex
            _new_model(first, model_two, provider_wire.url, secret_names[1])
            _chat(first, model_two, "cyberark-runtime-two-" + uuid.uuid4().hex)
            second_proxy_requests: Final = cyberark_wire.drain()

        with owned_proxy(
            gateway,
            tmp_path,
            environment,
            remove_environment=_SECRET_MANAGER_ENVIRONMENT,
        ) as restarted:
            model_three: Final = "integration-cyberark-" + uuid.uuid4().hex
            _new_model(restarted, model_three, provider_wire.url, secret_names[2])
            _chat(restarted, model_three, "cyberark-runtime-three-" + uuid.uuid4().hex)
            restarted_proxy_requests: Final = cyberark_wire.drain()
            known_secret_names: Final = frozenset(quote(name, safe="") for name in secret_names[:3])
            observed_secret_names: Final = frozenset(
                request.target.rsplit("/", 1)[-1]
                for request in (
                    *first_proxy_requests,
                    *second_proxy_requests,
                    *restarted_proxy_requests,
                )
                if request.method == "GET"
                and request.target.startswith("/secrets/")
                and request.target.rsplit("/", 1)[-1] in known_secret_names
            )
            assert observed_secret_names == known_secret_names

            deleted: Final = restarted.request("DELETE", "/config_overrides/cyberark")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json() == {
                "message": "CyberArk configuration deleted successfully",
                "status": "success",
            }, deleted.text
            assert (
                read_rows(
                    'SELECT config_type FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
                    ("cyberark",),
                    database_url=database_url,
                )
                == []
            )
            deleted_secret_reads: Final = tuple(
                request.target
                for request in cyberark_wire.drain()
                if request.method == "GET"
                and request.target.startswith("/secrets/")
                and request.target.rsplit("/", 1)[-1] in encoded_secret_names
            )
            assert deleted_secret_reads == ()
            model_after_delete: Final = "integration-cyberark-" + uuid.uuid4().hex
            _new_model(restarted, model_after_delete, provider_wire.url, "LITELLM_MASTER_KEY")
            _chat(restarted, model_after_delete, "after-delete")
            requests_after_delete: Final = cyberark_wire.drain()
            assert requests_after_delete == (), requests_after_delete
            empty_readback: Final = restarted.request("GET", "/config_overrides/cyberark")
            assert empty_readback.status_code == 200, empty_readback.text
            empty_values: Final = object_value(
                object_value(_JSON_OBJECT.validate_json(empty_readback.content))["values"]
            )
            assert empty_values == {
                field: None
                for field in (
                    "cyberark_api_base",
                    "cyberark_account",
                    "cyberark_username",
                    "cyberark_api_key",
                    "client_cert",
                    "client_key",
                    "ssl_verify",
                    "refresh_interval",
                )
            }, empty_readback.text


def test_vault_approle_and_cert_logins_use_login_namespace_and_drive_secret_reads(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_secret: Final = "vault-auth-provider-secret-" + uuid.uuid4().hex
    approle_token: Final = "approle-token-" + uuid.uuid4().hex
    cert_token: Final = "cert-token-" + uuid.uuid4().hex
    role_id: Final = "approle-role-" + uuid.uuid4().hex
    secret_id: Final = "approle-secret-" + uuid.uuid4().hex
    mount_path: Final = "approle-" + uuid.uuid4().hex
    cert_role: Final = "cert-role-" + uuid.uuid4().hex
    base_namespace: Final = "ns/base-" + uuid.uuid4().hex[:8]
    login_namespace: Final = "ns/login-" + uuid.uuid4().hex[:8]
    secret_namespace: Final = "ns/secrets-" + uuid.uuid4().hex[:8]
    mount_name: Final = "kv-" + uuid.uuid4().hex[:8]
    secret_a: Final = "secret-a-" + uuid.uuid4().hex
    secret_b: Final = "secret-b-" + uuid.uuid4().hex
    marker_a: Final = "vault-approle-runtime-" + uuid.uuid4().hex
    marker_b: Final = "vault-cert-runtime-" + uuid.uuid4().hex
    login_approle_path: Final = f"/v1/auth/{mount_path}/login"
    login_cert_path: Final = "/v1/auth/cert/login"
    secret_a_path: Final = f"/v1/{secret_namespace}/{mount_name}/data/{secret_a}"
    secret_b_path: Final = f"/v1/{secret_namespace}/{mount_name}/data/{secret_b}"

    def vault(request: Request) -> Reply:
        if request.method == "POST" and request.target == login_approle_path:
            assert _JSON_OBJECT.validate_json(request.body) == {
                "role_id": role_id,
                "secret_id": secret_id,
            }, request.body
            return Reply(body=json.dumps({"auth": {"client_token": approle_token, "lease_duration": 3600}}).encode())
        if request.method == "POST" and request.target == login_cert_path:
            assert _JSON_OBJECT.validate_json(request.body) == {"name": cert_role}, request.body
            return Reply(body=json.dumps({"auth": {"client_token": cert_token, "lease_duration": 3600}}).encode())
        if request.method == "GET" and request.target in {secret_a_path, secret_b_path}:
            return Reply(body=json.dumps({"data": {"data": {"key": provider_secret}}}).encode())
        return Reply(status=404, body=b'{"errors":["secret not found"]}')

    def provider(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        messages: Final = body["messages"]
        assert isinstance(messages, list)
        marker: Final = string_value(object_value(messages[0])["content"])
        if marker == "after-delete":
            return _completion(marker)
        assert request.headers["authorization"] == f"Bearer {provider_secret}", (
            "Provider request did not use the Vault-resolved secret"
        )
        assert marker in {marker_a, marker_b}, request.body
        return _completion(marker)

    cert_dir: Final = tmp_path / "vault-client-cert"
    cert_dir.mkdir()
    cert_file, key_file = write_self_signed_cert(cert_dir)
    client_cert_der: Final = ssl.PEM_cert_to_DER_cert(cert_file.read_text())
    server_cert_dir: Final = tmp_path / "vault-server-cert"
    server_cert_dir.mkdir()
    server_cert, server_key = write_self_signed_cert(server_cert_dir)
    presented_certs: Final[list[bytes | None]] = []

    class RecordingSSLSocket(ssl.SSLSocket):
        def do_handshake(self) -> None:
            super().do_handshake()
            presented_certs.append(self.getpeercert(binary_form=True))

    vault_tls_context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    vault_tls_context.load_cert_chain(server_cert, server_key)
    vault_tls_context.verify_mode = ssl.CERT_OPTIONAL
    vault_tls_context.load_verify_locations(cafile=str(cert_file))
    vault_tls_context.sslsocket_class = RecordingSSLSocket
    with (
        wire_server(vault) as vault_wire,
        wire_server(vault, tls=vault_tls_context) as vault_tls_wire,
        wire_server(provider) as provider_wire,
    ):
        with _config_proxy(gateway, tmp_path, monkeypatch, {"SSL_CERT_FILE": str(server_cert)}) as (
            candidate,
            database_url,
        ):
            full_config: Final = {
                "vault_addr": vault_wire.url,
                "approle_role_id": role_id,
                "approle_secret_id": secret_id,
                "approle_mount_path": mount_path,
                "vault_namespace": base_namespace,
                "vault_login_namespace": login_namespace,
                "vault_secret_namespace": secret_namespace,
                "vault_mount_name": mount_name,
            }
            saved: Final = candidate.request("POST", "/config_overrides/hashicorp_vault", full_config)
            assert saved.status_code == 200, saved.text
            assert saved.json() == {
                "message": "Hashicorp Vault configuration updated successfully",
                "status": "success",
            }, saved.text
            stored_full: Final = _stored_override(database_url, "hashicorp_vault")
            assert set(stored_full) == set(full_config), stored_full
            serialized_full: Final = json.dumps(stored_full)
            assert all(value not in serialized_full for value in full_config.values()), serialized_full

            with candidate.scenario() as scenario:
                model_a: Final = scenario.model(
                    api_key=f"os.environ/{secret_a}",
                    api_base=provider_wire.url + "/v1",
                )
                response_a: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model_a, "messages": [{"role": "user", "content": marker_a}]},
                )
                assert response_a.status_code == 200, response_a.text
                assert response_a.json()["choices"][0]["message"]["content"] == marker_a, response_a.text
                approle_targets: Final = frozenset({login_approle_path, login_cert_path, secret_a_path, secret_b_path})
                approle_phase: Final = tuple(
                    request for request in vault_wire.drain() if request.target in approle_targets
                )
                assert tuple((request.method, request.target) for request in approle_phase) == (
                    ("POST", login_approle_path),
                    ("GET", secret_a_path),
                ), approle_phase
                assert approle_phase[0].headers["x-vault-namespace"] == login_namespace
                assert _JSON_OBJECT.validate_json(approle_phase[0].body) == {
                    "role_id": role_id,
                    "secret_id": secret_id,
                }
                assert approle_phase[1].headers["x-vault-token"] == approle_token

                partial_config: Final = {
                    "vault_addr": vault_tls_wire.url,
                    "approle_role_id": "",
                    "approle_secret_id": "",
                    "client_cert": str(cert_file),
                    "client_key": str(key_file),
                    "vault_cert_role": cert_role,
                }
                partial: Final = candidate.request("POST", "/config_overrides/hashicorp_vault", partial_config)
                assert partial.status_code == 200, partial.text
                assert partial.json() == {
                    "message": "Hashicorp Vault configuration updated successfully",
                    "status": "success",
                }, partial.text
                vault_wire.drain()
                readback: Final = _override_values(candidate, "/config_overrides/hashicorp_vault", "hashicorp_vault")
                assert readback == {
                    "vault_addr": vault_tls_wire.url,
                    "approle_mount_path": mount_path,
                    "client_cert": str(cert_file),
                    "client_key": _masked(str(key_file)),
                    "vault_cert_role": cert_role,
                    "vault_namespace": base_namespace,
                    "vault_login_namespace": login_namespace,
                    "vault_secret_namespace": secret_namespace,
                    "vault_mount_name": mount_name,
                }, readback

                model_b: Final = scenario.model(
                    api_key=f"os.environ/{secret_b}",
                    api_base=provider_wire.url + "/v1",
                )
                response_b: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model_b, "messages": [{"role": "user", "content": marker_b}]},
                )
                assert response_b.status_code == 200, response_b.text
                assert response_b.json()["choices"][0]["message"]["content"] == marker_b, response_b.text
                plain_phase_b: Final = vault_wire.drain()
                assert plain_phase_b == (), plain_phase_b
                cert_phase: Final = tuple(
                    request for request in vault_tls_wire.drain() if request.target in approle_targets
                )
                assert all(request.target != login_approle_path for request in cert_phase), cert_phase
                assert cert_phase[0].target == login_cert_path, cert_phase
                cert_logins: Final = tuple(request for request in cert_phase if request.target == login_cert_path)
                assert all(
                    request.method == "POST"
                    and request.headers["x-vault-namespace"] == login_namespace
                    and _JSON_OBJECT.validate_json(request.body) == {"name": cert_role}
                    for request in cert_logins
                ), cert_logins
                cert_secret_reads: Final = tuple(
                    request for request in cert_phase if request.target in {secret_a_path, secret_b_path}
                )
                assert secret_b_path in tuple(request.target for request in cert_secret_reads), cert_secret_reads
                assert all(request.headers["x-vault-token"] == cert_token for request in cert_secret_reads), (
                    cert_secret_reads
                )
                assert client_cert_der in presented_certs, presented_certs

                deleted: Final = candidate.request("DELETE", "/config_overrides/hashicorp_vault")
                assert deleted.status_code == 200, deleted.text
                assert deleted.json() == {
                    "message": "Hashicorp Vault configuration deleted successfully",
                    "status": "success",
                }, deleted.text
                vault_wire.drain()
                vault_tls_wire.drain()
                assert (
                    read_rows(
                        'SELECT config_type FROM "LiteLLM_ConfigOverrides" WHERE config_type = %s',
                        ("hashicorp_vault",),
                        database_url=database_url,
                    )
                    == []
                )
                model_after_delete: Final = scenario.model(
                    api_key="os.environ/LITELLM_MASTER_KEY",
                    api_base=provider_wire.url + "/v1",
                )
                response_after_delete: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model_after_delete,
                        "messages": [{"role": "user", "content": "after-delete"}],
                    },
                )
                assert response_after_delete.status_code == 200, response_after_delete.text
                assert response_after_delete.json()["choices"][0]["message"]["content"] == "after-delete", (
                    response_after_delete.text
                )
                after_delete_requests: Final = vault_wire.drain()
                assert after_delete_requests == (), after_delete_requests
                after_delete_tls_requests: Final = vault_tls_wire.drain()
                assert after_delete_tls_requests == (), after_delete_tls_requests

            provider_requests: Final = provider_wire.drain()
            assert tuple((request.method, request.target) for request in provider_requests) == (
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/chat/completions"),
            ), provider_requests

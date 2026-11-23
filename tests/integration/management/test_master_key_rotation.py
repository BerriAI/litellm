import base64
import json
import uuid
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_if_encrypted_with,
    encrypt_secret_map,
    encrypt_value_helper,
)
from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration._support.database import read_rows, scratch_database, write_rows
from tests.integration._support.process import owned_proxy

SALT_KEY: Final = "sk-integration-rotation-salt"
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _register(proxy: Gateway, upstream: str, model_key: str, credential_key: str) -> tuple[str, str]:
    credential: Final = f"rotation-credential-{uuid.uuid4().hex}"
    inline_model: Final = f"rotation-inline-{uuid.uuid4().hex}"
    credential_model: Final = f"rotation-credential-model-{uuid.uuid4().hex}"
    proxy.post(
        "/credentials",
        {
            "credential_name": credential,
            "credential_values": {"api_key": credential_key},
            "credential_info": {"custom_llm_provider": "openai"},
        },
    )
    proxy.post(
        "/model/new",
        {
            "model_name": inline_model,
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": upstream, "api_key": model_key},
        },
    )
    proxy.post(
        "/model/new",
        {
            "model_name": credential_model,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": upstream,
                "litellm_credential_name": credential,
            },
        },
    )
    return inline_model, credential_model


def _statuses(proxy: Gateway, models: tuple[str, ...], key: str, text: str) -> dict[str, int]:
    return {
        model: proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}]},
            key=key,
        ).status_code
        for model in models
    }


def _upstream_authorizations(gateway: Gateway) -> list[str]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    response.raise_for_status()
    return [
        string_value(object_value(entry)["authorization"])
        for entry in list_value(object_value(_JSON.validate_json(response.content))["requests"])
        if string_value(object_value(entry)["path"]).endswith("/chat/completions")
    ]


def _clear_upstream(gateway: Gateway) -> None:
    httpx.delete(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False).raise_for_status()


@pytest.mark.timeout(300)
def test_master_key_rotation_with_a_salt_key_keeps_db_models_and_credentials_working(
    gateway: Gateway, tmp_path: Path
) -> None:
    new_master_key: Final = f"sk-integration-rotated-{uuid.uuid4().hex}"
    model_key: Final = f"sk-model-{uuid.uuid4().hex}"
    credential_key: Final = f"sk-credential-{uuid.uuid4().hex}"
    expected: Final = [f"Bearer {model_key}", f"Bearer {credential_key}"]
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    with scratch_database() as database_url:
        environment: Final = {"DATABASE_URL": database_url, "LITELLM_SALT_KEY": SALT_KEY}
        with owned_proxy(gateway, tmp_path / "before", environment) as before:
            models: Final = _register(before, f"{gateway.upstream_url}/v1", model_key, credential_key)
            _clear_upstream(gateway)
            assert _statuses(before, models, gateway.key, "before rotation") == dict.fromkeys(models, 200)
            assert _upstream_authorizations(gateway) == expected
            rotated: Final = before.post("/key/regenerate", {"key": gateway.key, "new_master_key": new_master_key})
            assert rotated["key"] == new_master_key

        with owned_proxy(gateway, tmp_path / "after", {**environment, "LITELLM_MASTER_KEY": new_master_key}) as after:
            _clear_upstream(gateway)
            served: Final = eventually(
                lambda: _statuses(after, models, new_master_key, "after rotation"),
                lambda statuses: statuses == dict.fromkeys(models, 200),
                seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
                return_last_on_timeout=True,
            )
            assert served == dict.fromkeys(models, 200)
            assert _upstream_authorizations(gateway)[-2:] == expected


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("salt_key", "reject_write"),
    [(None, None), (SALT_KEY, None), (None, "constraint"), (None, "skip")],
    ids=("master-key-encryption", "salt-key-encryption", "write-error-rollback", "write-conflict-rollback"),
)
def test_master_key_rotation_keeps_stored_secrets_readable(
    gateway: Gateway, tmp_path: Path, salt_key: str | None, reject_write: Literal["constraint", "skip"] | None
) -> None:
    new_master_key: Final = f"sk-integration-rotated-{uuid.uuid4().hex}"
    old_encryption_key: Final = salt_key or gateway.key
    expected_key: Final = old_encryption_key if reject_write else salt_key or new_master_key
    secret: Final = uuid.uuid4().hex
    encrypted: Final = encrypt_value_helper(secret, old_encryption_key)
    sso_settings: Final = {"generic_client_secret": encrypted}
    cache_settings: Final = {"password": encrypted, "port": 6379}
    header_map: Final = {"Authorization": f"Bearer {secret}"}
    env_map: Final = {"MCP_TOKEN": secret}
    encrypted_headers: Final = encrypt_secret_map(header_map, new_encryption_key=old_encryption_key)
    encrypted_env: Final = encrypt_secret_map(env_map, new_encryption_key=old_encryption_key)
    legacy_credential: Final = base64.urlsafe_b64encode(secret.encode()).decode()
    with scratch_database() as database_url:
        environment: Final = {"DATABASE_URL": database_url, "LITELLM_MASTER_KEY": gateway.key}
        with owned_proxy(
            gateway,
            tmp_path,
            environment if salt_key is None else {**environment, "LITELLM_SALT_KEY": salt_key},
            remove_environment=("LITELLM_SALT_KEY",) if salt_key is None else (),
        ) as proxy:
            write_rows(
                'INSERT INTO "LiteLLM_SSOConfig" (id, sso_settings, updated_at) VALUES (%s, %s::jsonb, now())',
                ("sso_config", json.dumps(sso_settings)),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_CacheConfig" (id, cache_settings, updated_at) VALUES (%s, %s::jsonb, now())',
                ("cache_config", json.dumps(json.dumps(cache_settings))),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES (%s, %s::jsonb)',
                ("vantage_settings", json.dumps({"integration_token": encrypted})),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_UserProviderCredentials" '
                "(id, user_id, credential_name, provider, credential_b64, updated_at) "
                "VALUES ('connection', 'user', 'copilot', 'github_copilot', %s, now())",
                (encrypted,),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_MCPServerTable" (server_id, static_headers, env) VALUES (%s, %s::jsonb, %s::jsonb)',
                ("mcp", json.dumps(encrypted_headers), json.dumps(encrypted_env)),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_MCPServerTable" (server_id, static_headers, env) VALUES (%s, %s::jsonb, %s::jsonb)',
                ("legacy", json.dumps(header_map), json.dumps(json.dumps(env_map))),
                database_url=database_url,
            )
            write_rows(
                'INSERT INTO "LiteLLM_MCPUserCredentials" (id, user_id, server_id, credential_b64, updated_at) '
                "VALUES ('legacy-credential', 'user', 'legacy', %s, now())",
                (legacy_credential,),
                database_url=database_url,
            )
            if reject_write == "constraint":
                write_rows(
                    'ALTER TABLE "LiteLLM_SSOConfig" ADD CONSTRAINT reject_encrypted_sso '
                    "CHECK (sso_settings ->> 'generic_client_secret' IS NULL) NOT VALID",
                    (),
                    database_url=database_url,
                )
            if reject_write == "skip":
                write_rows(
                    "CREATE FUNCTION skip_mcp_credential_update() RETURNS trigger LANGUAGE plpgsql "
                    "AS $$ BEGIN RETURN NULL; END; $$",
                    (),
                    database_url=database_url,
                )
                write_rows(
                    'CREATE TRIGGER skip_mcp_credential_update BEFORE UPDATE ON "LiteLLM_MCPUserCredentials" '
                    "FOR EACH ROW EXECUTE PROCEDURE skip_mcp_credential_update()",
                    (),
                    database_url=database_url,
                )
            response: Final = proxy.request(
                "POST", "/key/regenerate", {"key": gateway.key, "new_master_key": new_master_key}
            )
            assert response.status_code == (500 if reject_write else 200), response.text
            if not reject_write:
                assert object_value(_JSON.validate_json(response.content))["key"] == new_master_key

        sso_row: Final = read_rows(
            'SELECT sso_settings FROM "LiteLLM_SSOConfig" WHERE id = %s', ("sso_config",), database_url=database_url
        )
        cache_row: Final = read_rows(
            'SELECT cache_settings FROM "LiteLLM_CacheConfig" WHERE id = %s',
            ("cache_config",),
            database_url=database_url,
        )
        billing_row: Final = read_rows(
            'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
            ("vantage_settings",),
            database_url=database_url,
        )
        provider_row: Final = read_rows(
            'SELECT credential_b64 FROM "LiteLLM_UserProviderCredentials" WHERE id = %s',
            ("connection",),
            database_url=database_url,
        )
        mcp_row: Final = read_rows(
            'SELECT static_headers, env FROM "LiteLLM_MCPServerTable" WHERE server_id = %s',
            ("mcp",),
            database_url=database_url,
        )
        legacy_mcp_row: Final = read_rows(
            'SELECT static_headers, env FROM "LiteLLM_MCPServerTable" WHERE server_id = %s',
            ("legacy",),
            database_url=database_url,
        )
        legacy_credential_row: Final = read_rows(
            'SELECT credential_b64 FROM "LiteLLM_MCPUserCredentials" WHERE id = %s',
            ("legacy-credential",),
            database_url=database_url,
        )
        stored_sso: Final = object_value(sso_row[0]["sso_settings"])
        stored_cache: Final = object_value(_JSON.validate_json(string_value(cache_row[0]["cache_settings"])))
        ciphertexts: Final = (
            string_value(stored_sso["generic_client_secret"]),
            string_value(stored_cache["password"]),
            string_value(object_value(billing_row[0]["param_value"])["integration_token"]),
            string_value(provider_row[0]["credential_b64"]),
        )
        assert tuple(decrypt_if_encrypted_with(value, expected_key) for value in ciphertexts) == (secret,) * 4
        stored_mcp: Final = (mcp_row[0]["static_headers"], mcp_row[0]["env"])
        stored_legacy_mcp: Final = (legacy_mcp_row[0]["static_headers"], legacy_mcp_row[0]["env"])
        stored_legacy_credential: Final = string_value(legacy_credential_row[0]["credential_b64"])
        mcp_ciphertexts: Final = tuple(string_value(_JSON.validate_json(string_value(value))) for value in stored_mcp)
        assert tuple(decrypt_if_encrypted_with(value, expected_key) for value in mcp_ciphertexts) == (
            json.dumps(header_map, separators=(",", ":")),
            json.dumps(env_map, separators=(",", ":")),
        )
        assert stored_cache["port"] == cache_settings["port"]
        if expected_key == old_encryption_key:
            assert ciphertexts == (encrypted,) * 4
            assert stored_mcp == (encrypted_headers, encrypted_env)
            assert stored_legacy_mcp == (header_map, json.dumps(env_map))
            assert stored_legacy_credential == legacy_credential
        else:
            assert tuple(decrypt_if_encrypted_with(value, old_encryption_key) for value in ciphertexts) == (None,) * 4
            assert tuple(decrypt_if_encrypted_with(value, old_encryption_key) for value in mcp_ciphertexts) == (
                None,
                None,
            )
            upgraded_maps: Final = tuple(
                string_value(_JSON.validate_json(string_value(value))) for value in stored_legacy_mcp
            )
            assert tuple(decrypt_if_encrypted_with(value, expected_key) for value in upgraded_maps) == (
                json.dumps(header_map, separators=(",", ":")),
                json.dumps(env_map, separators=(",", ":")),
            )
            assert decrypt_if_encrypted_with(stored_legacy_credential, expected_key) == secret

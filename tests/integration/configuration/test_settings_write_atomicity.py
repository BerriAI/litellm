import base64
import json
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import jwt
import psycopg
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from psycopg import sql
from pydantic import JsonValue

from integration._support.client import Gateway, eventually, object_value
from integration._support.process import owned_proxy
from integration._support.redis_process import owned_redis
from integration._support.upstream import register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse


@contextmanager
def _owned_database(tmp_path: Path) -> Iterator[str]:
    original: Final = os.environ["DATABASE_URL"]
    identity: Final = "integration_atomicity_" + uuid.uuid4().hex
    parsed: Final = urlsplit(original)
    database_url: Final = urlunsplit((parsed.scheme, parsed.netloc, "/" + identity, "", ""))
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(identity)))
        try:
            yield database_url
        finally:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(identity)))


def _b64url_uint(value: int) -> str:
    raw: Final = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _write_owned_config(
    tmp_path: Path, general_settings: Mapping[str, JsonValue], litellm_settings: Mapping[str, JsonValue]
) -> Path:
    path: Final = tmp_path / "atomicity-config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    **general_settings,
                },
                "litellm_settings": {
                    "cache": True,
                    "cache_params": {
                        "type": "redis",
                        "host": "os.environ/REDIS_HOST",
                        "port": "os.environ/REDIS_PORT",
                    },
                    **litellm_settings,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return path


def test_refused_patch_leaves_serving_value(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _write_owned_config(tmp_path, {}, {"default_internal_user_params": {"max_budget": 999.0}})
    with _owned_database(tmp_path) as database_url, owned_redis(tmp_path) as cache:
        with owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url, "REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)},
            config=config,
        ) as candidate:
            first_user: Final = candidate.post("/user/new", {"user_id": f"integration-{uuid.uuid4().hex}"})
            assert first_user["max_budget"] == 999.0, json.dumps(first_user)
            refused: Final = candidate.request("PATCH", "/update/internal_user_settings", {"max_budget": 777.0})
            assert refused.status_code == 400, refused.text
            detail: Final = refused.json()["detail"]
            assert (
                detail["error"]
                == "litellm_settings key 'default_internal_user_params' is set in the config file and cannot be changed here."
            ), refused.text
            assert detail["keys"] == ["default_internal_user_params"], refused.text
            assert detail["section"] == "litellm_settings", refused.text
            second_user: Final = candidate.post("/user/new", {"user_id": f"integration-{uuid.uuid4().hex}"})
            assert second_user["max_budget"] == 999.0, json.dumps(second_user)
            settings: Final = candidate.get("/get/internal_user_settings")
            assert object_value(settings["values"])["max_budget"] == 999.0, json.dumps(settings)


def test_db_general_settings_row_cannot_rebind_role_permissions(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers: Final = private_key.public_key().public_numbers()
    kid: Final = f"integration-jwk-{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        f"integration-jwks-{uuid.uuid4().hex}",
        JsonResponse(
            content_type="application/json",
            body={
                "keys": [
                    {
                        "kty": "RSA",
                        "kid": kid,
                        "use": "sig",
                        "alg": "RS256",
                        "n": _b64url_uint(numbers.n),
                        "e": _b64url_uint(numbers.e),
                    }
                ]
            },
        ),
    )
    passthrough_path: Final = f"/atomicity-passthrough-{uuid.uuid4().hex}"
    passthrough: Final = register_scenario(
        f"integration-passthrough-{uuid.uuid4().hex}",
        JsonResponse(content_type="application/json", body={"atomicity": "reconciled"}),
    )

    def mint_admin_token() -> str:
        return jwt.encode(
            {
                "sub": f"integration-admin-{uuid.uuid4().hex}",
                "scope": "litellm_proxy_admin",
                "jti": uuid.uuid4().hex,
                "exp": int(time.time()) + 3600,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": kid},
        )

    config: Final = _write_owned_config(
        tmp_path,
        {
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "admin_jwt_scope": "litellm_proxy_admin",
                "admin_allowed_routes": [
                    "management_routes",
                    "spend_tracking_routes",
                    "global_spend_tracking_routes",
                    "info_routes",
                    "openai_routes",
                ],
                "enforce_rbac": True,
                "public_key_ttl": 1,
            },
        },
        {},
    )
    with _owned_database(tmp_path) as database_url, owned_redis(tmp_path) as cache:
        overrides: Final = {
            "DATABASE_URL": database_url,
            "REDIS_HOST": cache.host,
            "REDIS_PORT": str(cache.port),
            "JWT_PUBLIC_KEY_URL": handle.api_base(),
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "1",
            "LITELLM_CONFIG_PARAM_CACHE_TTL_SECONDS": "1",
        }
        with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            body: Final = {"model": model, "messages": [{"role": "user", "content": "atomicity control"}]}
            first: Final = candidate.request("POST", "/v1/chat/completions", body, key=mint_admin_token())
            assert first.status_code == 200, first.text
            unregistered: Final = candidate.request("GET", passthrough_path)
            assert unregistered.status_code == 404, unregistered.text
            with psycopg.connect(database_url, autocommit=True) as connection:
                connection.execute(
                    'INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES (%s, %s::jsonb) '
                    "ON CONFLICT (param_name) DO UPDATE SET param_value = EXCLUDED.param_value",
                    (
                        "general_settings",
                        json.dumps(
                            {
                                "pass_through_endpoints": [
                                    {"path": passthrough_path, "target": passthrough.api_base(), "headers": {}}
                                ],
                                "role_permissions": [
                                    {"role": "proxy_admin", "models": ["not-the-model"], "routes": ["/nowhere"]}
                                ],
                            }
                        ),
                    ),
                )

            def passthrough_body() -> dict[str, JsonValue] | None:
                response: Final = candidate.request("GET", passthrough_path)
                return object_value(response.json()) if response.status_code == 200 else None

            eventually(passthrough_body, lambda value: value == {"atomicity": "reconciled"}, seconds=30)
            second: Final = candidate.request("POST", "/v1/chat/completions", body, key=mint_admin_token())
            assert second.status_code == 200, second.text
            assert second.json()["usage"]["total_tokens"] == 40, second.text

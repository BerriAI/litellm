import json
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from jwt.algorithms import RSAAlgorithm
from pydantic import JsonValue

AUDIENCE: Final = "litellm-integration"
KEY_ID: Final = "integration-signing-key"
CLIENT_CLAIM: Final = "client_id"


def _tool_call_request(model: str, tool_name: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "tool policy user control"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "integration tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }


def _forget_tool(tool_name: str) -> None:
    write_rows('DELETE FROM "LiteLLM_ToolTable" WHERE tool_name = %s', (tool_name,))


def _discovered_tool(gateway: Gateway, tool_name: str) -> dict[str, JsonValue]:
    def rows() -> list[dict[str, JsonValue]]:
        tools: Final = gateway.get("/v1/tool/list")["tools"]
        assert isinstance(tools, list)
        return [object_value(tool) for tool in tools if object_value(tool)["tool_name"] == tool_name]

    return eventually(rows, lambda found: len(found) == 1, seconds=70)[0]


def test_tool_list_reports_the_user_that_owns_the_discovering_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = "integration-alias-" + uuid.uuid4().hex
        user: Final = scenario.user(user_alias=alias, user_email=f"{alias}@integration.example")
        key: Final = scenario.key(user_id=user, models=[model])
        tool_name: Final = "integration_tool_" + uuid.uuid4().hex
        scenario.cleanups.callback(_forget_tool, tool_name)
        response: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
        assert response.status_code == 200, response.text
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] == sha256(key.encode()).hexdigest(), tool
        assert tool["user"] == {"user_id": user, "user_email": f"{alias}@integration.example", "user_alias": alias}, (
            tool
        )


def test_tool_list_reports_no_user_for_a_key_without_an_owner(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        tool_name: Final = "integration_tool_" + uuid.uuid4().hex
        scenario.cleanups.callback(_forget_tool, tool_name)
        response: Final = gateway.request("POST", "/v1/chat/completions", _tool_call_request(model, tool_name), key=key)
        assert response.status_code == 200, response.text
        tool: Final = _discovered_tool(gateway, tool_name)
        assert tool["key_hash"] == sha256(key.encode()).hexdigest(), tool
        assert tool["user"] is None, tool


def _proxy_config(directory: Path, model: str, upstream_url: str) -> Path:
    config: Final = directory / "jwt_auto_register_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/" + model,
                            "api_base": upstream_url + "/v1",
                            "api_key": "sk-upstream",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {
                        "user_id_jwt_field": "sub",
                        "user_email_jwt_field": "email",
                        "user_id_upsert": True,
                        "virtual_key_claim_field": CLIENT_CLAIM,
                        "unregistered_jwt_client_behavior": "auto_register",
                    },
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _signed_token(private_key: rsa.RSAPrivateKey, user_id: str, email: str, client_id: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": user_id, "email": email, CLIENT_CLAIM: client_id, "aud": AUDIENCE, "iat": now, "exp": now + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _forget_auto_registered_client(client_id: str, user_id: str) -> None:
    write_rows(
        'DELETE FROM "LiteLLM_VerificationToken" WHERE token IN '
        '(SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value = %s)',
        (client_id,),
    )
    write_rows('DELETE FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_value = %s', (client_id,))
    write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))


def test_tool_list_reports_the_jwt_user_behind_an_auto_registered_key(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.target == "/jwks", request
        return Reply(body=jwks)

    model: Final = "integration-jwt-" + uuid.uuid4().hex
    with wire_server(respond) as issuer:
        config: Final = _proxy_config(tmp_path, model, gateway.upstream_url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": issuer.url + "/jwks", "JWT_AUDIENCE": AUDIENCE}
        with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
            user: Final = "integration-jwt-user-" + uuid.uuid4().hex
            email: Final = f"{user}@integration.example"
            client_id: Final = "integration-client-" + uuid.uuid4().hex
            tool_name: Final = "integration_tool_" + uuid.uuid4().hex
            scenario.cleanups.callback(_forget_tool, tool_name)
            scenario.cleanups.callback(_forget_auto_registered_client, client_id, user)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                _tool_call_request(model, tool_name),
                key=_signed_token(private_key, user, email, client_id),
            )
            assert response.status_code == 200, response.text
            mapped: Final = read_rows(
                'SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_name = %s AND jwt_claim_value = %s',
                (CLIENT_CLAIM, client_id),
            )
            assert len(mapped) == 1, mapped
            assert read_rows(
                'SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token = %s', (mapped[0]["token"],)
            ) == [{"user_id": user}]
            tool: Final = _discovered_tool(candidate, tool_name)
            assert tool["key_hash"] == mapped[0]["token"], tool
            assert tool["user"] == {"user_id": user, "user_email": email, "user_alias": None}, tool

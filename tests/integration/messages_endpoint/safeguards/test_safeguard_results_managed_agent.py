"""A managed agent signed in with its Entra application token reaches the classifier only through its own model
grants. Grants that omit the classifier model answer ``unsupported`` without a classifier call, and grants that name it
classify the turn, so the agent's policy and not the gateway's own access decides."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Scenario, string_value
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    CLASSIFIER_MODEL,
    UNSUPPORTED,
    Rig,
    auto_mode_body,
    chat_backend,
    flagged,
    json_body,
    nonce,
    safeguards_rig,
    tool_id,
    tool_use_ids,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

KEY_ID: Final = "safeguards-managed-agent-key"
TENANT: Final = str(uuid.uuid4())
ISSUER: Final = f"https://login.microsoftonline.com/{TENANT}/v2.0"
AUDIENCE: Final = "api://safeguards-gateway"
AGENT_CARD: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "protocolVersion": "0.3.0",
        "name": "integration",
        "description": "integration agent",
        "url": "http://127.0.0.1:1/agent",
        "version": "1",
        "capabilities": {},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }
)


@dataclass(frozen=True, slots=True)
class Signer:
    private_key: rsa.RSAPrivateKey
    jwks_url: str


@pytest.fixture(scope="module")
def signer() -> Iterator[Signer]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    body: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID}]}).encode()

    def respond(request: Request) -> Reply:
        return Reply(body=body)

    with wire_server(respond) as jwks:
        yield Signer(private_key, jwks.url)


@pytest.fixture(scope="module")
def rig(signer: Signer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    jwt_settings: Final[dict[str, JsonValue]] = {
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"issuers": [{"issuer": ISSUER, "jwks_url": signer.jwks_url, "audience": AUDIENCE}]},
    }
    with safeguards_rig(tmp_path_factory.mktemp("safeguards-managed"), general_settings=jwt_settings) as value:
        yield value


def managed_agent_token(scenario: Scenario, signer: Signer, models: Sequence[str]) -> str:
    client_id: Final = str(uuid.uuid4())
    service_principal: Final = str(uuid.uuid4())
    agent: Final = scenario.gateway.post(
        "/v1/agents",
        {
            "agent_name": f"integration-{uuid.uuid4().hex[:10]}",
            "agent_card_params": dict(AGENT_CARD),
            "identity": {
                "provider": "microsoft_entra",
                "tenant_id": TENANT,
                "client_id": client_id,
                "service_principal_id": service_principal,
            },
            "execution_mode": "autonomous",
            "object_permission": {"models": list(models)},
        },
    )
    agent_id: Final = string_value(agent["agent_id"])
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"/v1/agents/{agent_id}")
    issued_at: Final = int(time.time())
    return jwt.encode(
        {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "tid": TENANT,
            "azp": client_id,
            "oid": service_principal,
            "sub": service_principal,
            "idtyp": "app",
            "iat": issued_at,
            "exp": issued_at + 600,
        },
        signer.private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def classify(rig: Rig, token: str) -> tuple[dict[str, JsonValue], str]:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    return json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce), key=token)), request_nonce


def test_managed_agent_without_a_classifier_grant_gets_unsupported(rig: Rig, signer: Signer) -> None:
    with rig.gateway.scenario() as scenario:
        token: Final = managed_agent_token(scenario, signer, [CHAT_MODEL])
        direct: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": CLASSIFIER_MODEL, "messages": [{"role": "user", "content": f"direct {nonce()}"}]},
            key=token,
        )
        assert direct.status_code == 403, direct.text
        body, request_nonce = classify(rig, token)
        assert body["safeguard_results"] == list(UNSUPPORTED), body
        received: Final = rig.received(request_nonce)
        assert rig.classifier_calls(received) == (), [request.target for request in received]
        assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def test_managed_agent_granted_the_classifier_is_classified(rig: Rig, signer: Signer) -> None:
    with rig.gateway.scenario() as scenario:
        token: Final = managed_agent_token(scenario, signer, [CHAT_MODEL, CLASSIFIER_MODEL])
        body, request_nonce = classify(rig, token)
        assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
        received: Final = rig.received(request_nonce)
        assert len(rig.classifier_calls(received)) == 1, [request.target for request in received]

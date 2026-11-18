import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue, TypeAdapter

from tests.integration._support import responses_stream
from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, Wire, wire_server

KEY_ID: Final = "integration-jwt-signing-key"
JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])

Signer = Callable[[Mapping[str, JsonValue]], str]


@dataclass(frozen=True, slots=True)
class UpsertRig:
    owned: OwnedProxy
    sign: Signer
    forge: Signer
    jwks: Wire

    @property
    def gateway(self) -> Gateway:
        return self.owned.gateway


def _signer(private_key: rsa.RSAPrivateKey) -> Signer:
    def sign(claims: Mapping[str, JsonValue]) -> str:
        return jwt.encode(
            {**claims, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )

    return sign


@contextmanager
def upserting_jwt_rig(
    gateway: Gateway,
    directory: Path,
    team_claim: Mapping[str, str],
    default_team_params: Mapping[str, JsonValue] | None,
    *,
    workers: int = 1,
) -> Iterator[UpsertRig]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"user_id_jwt_field": "sub", **team_claim, "team_id_upsert": True},
    }
    config["litellm_settings"] = {
        **config["litellm_settings"],
        **({} if default_team_params is None else {"default_team_params": dict(default_team_params)}),
    }
    path: Final = directory / "jwt_team_upsert.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(respond) as jwks,
        owned_proxy_process(
            gateway, directory, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=workers
        ) as owned,
    ):
        yield UpsertRig(
            owned,
            _signer(private_key),
            _signer(rsa.generate_private_key(public_exponent=65537, key_size=2048)),
            jwks,
        )


def team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT models, team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def team_models(team_id: str) -> list[JsonValue]:
    return [row["models"] for row in team_rows(team_id)]


def delete_team_if_present(scenario: Scenario, team_id: str) -> None:
    if team_rows(team_id):
        scenario.delete_team(team_id)


def prompt_of(request: Request) -> str:
    return json.dumps(JSON_BODY.validate_json(request.body))


def marker_of(request: Request) -> str:
    text: Final = prompt_of(request)
    start: Final = text.index("marker-")
    return text[start : start + len("marker-") + 32]


def provider_reply(request: Request) -> Reply:
    """An OpenAI-compatible provider whose response id and text carry the marker the caller put in the prompt."""
    marker: Final = marker_of(request)
    streaming: Final = JSON_BODY.validate_json(request.body).get("stream") is True
    if request.target.endswith("/responses"):
        identity: Final = f"resp_{marker}"
        if not streaming:
            return Reply(body=json.dumps(responses_stream.completed(identity, marker)["response"]).encode())
        return Reply(
            content_type="text/event-stream",
            chunks=(
                responses_stream.frame(responses_stream.created(identity)),
                responses_stream.frame(responses_stream.delta(identity, marker)),
                responses_stream.frame(responses_stream.completed(identity, marker)),
            ),
        )
    chat_id: Final = f"chatcmpl-{marker}"
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )
    chunk: Final = {"id": chat_id, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": marker}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
    )

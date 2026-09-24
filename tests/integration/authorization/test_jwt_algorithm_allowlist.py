import asyncio
import base64
import json
import os
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import jwt
import psutil
import yaml
from anthropic import Anthropic
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAAlgorithm
from openai import AsyncOpenAI, OpenAI

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-jwt-allowlist-key"
APPROVED_WARNING_PARTS: Final = ("EdDSA", "deprecated", "LITELLM_FIPS_MODE")


def _jwks_reply(public_jwk: str) -> Reply:
    return _jwks_reply_keys([{**json.loads(public_jwk), "kid": KEY_ID}])


def _jwks_reply_keys(keys: list[dict[str, object]]) -> Reply:
    return Reply(body=json.dumps({"keys": keys}).encode())


def _eddsa_keypair() -> tuple[ed25519.Ed25519PrivateKey, dict[str, object]]:
    private_key: Final = ed25519.Ed25519PrivateKey.generate()
    return private_key, {**json.loads(OKPAlgorithm.to_jwk(private_key.public_key())), "kid": "ed"}


def _rsa_keypair() -> tuple[rsa.RSAPrivateKey, dict[str, object]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, {**json.loads(RSAAlgorithm.to_jwk(private_key.public_key())), "kid": "rsa"}


def _ec_keypair() -> tuple[ec.EllipticCurvePrivateKey, dict[str, object]]:
    private_key: Final = ec.generate_private_key(ec.SECP256R1())
    return private_key, {**json.loads(ECAlgorithm.to_jwk(private_key.public_key())), "kid": "ec"}


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def _observed_bodies() -> tuple[str, ...]:
    response: Final = httpx.get(f"{os.environ['INTEGRATION_UPSTREAM_URL']}/__observations", timeout=15)
    assert response.status_code == 200, response.text
    return tuple(json.dumps(entry["body"]) for entry in response.json()["requests"])


def _jwt_auth_config(tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
    }
    path: Final = tmp_path / "jwt_allowlist.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _token(key: object, algorithm: str, subject: str, kid: str = KEY_ID) -> str:
    return jwt.encode(
        {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
        key,  # pyright: ignore[reportArgumentType]  # jwt.encode takes Any key material
        algorithm=algorithm,
        headers={"kid": kid},
    )


def test_eddsa_signed_token_is_accepted_and_logged_as_deprecated_outside_fips_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = ed25519.Ed25519PrivateKey.generate()
    public_jwk: Final = OKPAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "EdDSA", subject)
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path)
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            deprecation: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: "EdDSA" in text and "deprecated" in text and "LITELLM_FIPS_MODE" in text,
                seconds=30,
            )
            assert "EdDSA" in deprecation and "deprecated" in deprecation and "LITELLM_FIPS_MODE" in deprecation


def test_rs256_signed_token_is_accepted_without_deprecation_log(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "RS256", subject)
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path)
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            assert "EdDSA" not in owned.log.read_text(), owned.log.read_text()


def _deprecation_lines(log_text: str) -> list[str]:
    return [line for line in log_text.splitlines() if all(part in line for part in APPROVED_WARNING_PARTS)]


def _alg_lied_token(private_key: ed25519.Ed25519PrivateKey, kid: str, subject: str) -> str:
    def segment(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    signing_input: Final = (
        f"{segment(json.dumps({'alg': 'RS256', 'typ': 'JWT', 'kid': kid}).encode())}."
        f"{segment(json.dumps({'sub': subject, 'iat': int(time.time()), 'exp': int(time.time()) + 300}).encode())}"
    )
    signature: Final = segment(private_key.sign(signing_input.encode()))
    return f"{signing_input}.{signature}"


def test_eddsa_non_stream_request_reaches_upstream_and_logs_deprecation(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _eddsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "EdDSA", subject, kid="ed")
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            marker: Final = f"jwt-allowlist-a1-{uuid.uuid4().hex}"
            _observed_bodies()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": marker}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            deprecation: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: _deprecation_lines(text) != [],
                seconds=30,
            )
            assert _deprecation_lines(deprecation) != []
            observed: Final = _observed_bodies()
            assert any(marker in body for body in observed), observed


def test_eddsa_streamed_request_through_openai_sdk_reaches_upstream_and_logs_deprecation(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key, jwks_key = _eddsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "EdDSA", subject, kid="ed")
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            marker: Final = f"jwt-allowlist-a3-{uuid.uuid4().hex}"
            _observed_bodies()
            sdk: Final = OpenAI(base_url=f"{owned.gateway.client.base_url}/v1", api_key=token, timeout=30)
            chunks: Final = [
                chunk
                for chunk in sdk.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": marker}], stream=True
                )
            ]
            assert chunks, "stream produced no chunks"
            deprecation: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: _deprecation_lines(text) != [],
                seconds=30,
            )
            assert _deprecation_lines(deprecation) != []
            observed: Final = _observed_bodies()
            assert any(marker in body for body in observed), observed


def test_rs256_messages_request_through_anthropic_sdk_reaches_upstream(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _rsa_keypair()
    completion: Final = json.dumps(
        {
            "id": "msg_allowlist_a4",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "a4-reply"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
    ).encode()

    def jwks(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    def provider(request: Request) -> Reply:
        return Reply(body=completion)

    with wire_server(jwks) as keys_server, wire_server(provider) as provider_server, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "RS256", subject, kid="rsa")
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": keys_server.url},
            config=_jwt_auth_config(tmp_path),
            workers=2,
        ) as owned:
            model: Final = scenario.model(model="anthropic/claude-3-5-haiku-latest", api_base=provider_server.url)
            scenario.cleanups.callback(scenario.delete_user, subject)
            sdk: Final = Anthropic(
                base_url=str(owned.gateway.client.base_url).rstrip("/"),
                auth_token=token,
                timeout=30,
                max_retries=0,
            )
            message: Final = sdk.messages.create(
                model=model, max_tokens=8, messages=[{"role": "user", "content": "a4"}]
            )
            text: Final = message.content[0].text
            assert "a4-reply" in text, message
            received: Final = provider_server.drain()
            assert received, "upstream never received the request"


def test_rs256_responses_request_through_openai_async_sdk_reaches_upstream(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _rsa_keypair()
    response_object: Final = json.dumps(
        {
            "id": "resp_allowlist_a5",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": "msg_a5",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "a5-reply", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }
    ).encode()

    def jwks(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    def provider(request: Request) -> Reply:
        return Reply(body=response_object)

    with wire_server(jwks) as keys_server, wire_server(provider) as provider_server, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "RS256", subject, kid="rsa")
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": keys_server.url},
            config=_jwt_auth_config(tmp_path),
            workers=2,
        ) as owned:
            model: Final = scenario.model(api_base=provider_server.url)
            scenario.cleanups.callback(scenario.delete_user, subject)

            async def call() -> str:
                sdk: Final = AsyncOpenAI(base_url=f"{owned.gateway.client.base_url}/v1", api_key=token, timeout=30)
                response: Final = await sdk.responses.create(model=model, input="a5")
                return response.id

            identity: Final = asyncio.run(call())
            assert identity.startswith("resp_"), identity
            received: Final = provider_server.drain()
            assert received, "upstream never received the request"


def test_es256_signed_token_is_accepted(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _ec_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = jwt.encode(
            {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="ES256",
            headers={"kid": "ec"},
        )
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist es256"}]},
                key=token,
            )
            assert response.status_code == 200, response.text


def test_hs256_token_against_oct_jwks_key_is_rejected(gateway: Gateway, tmp_path: Path) -> None:
    secret: Final = b"integration-hs256-jwt-secret-0123456789abcdef"
    oct_key: Final = {
        "kty": "oct",
        "kid": "sym",
        "alg": "HS256",
        "k": base64.urlsafe_b64encode(secret).rstrip(b"=").decode(),
    }

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([oct_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = jwt.encode(
            {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
            secret,
            algorithm="HS256",
            headers={"kid": "sym"},
        )
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist hs256"}]},
                key=token,
            )
            assert response.status_code == 401, response.text
            assert "error" in response.text, response.text


def test_token_signed_eddsa_with_rs256_header_is_rejected(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _eddsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _alg_lied_token(private_key, "ed", subject)
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist lied"}]},
                key=token,
            )
            assert response.status_code == 401, response.text


def test_request_without_authorization_header_is_rejected(gateway: Gateway, tmp_path: Path) -> None:
    _, jwks_key = _rsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            response: Final = owned.gateway.client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist noauth"}]},
            )
            assert response.status_code == 401, response.text


def test_empty_jwks_document_rejects_tokens(gateway: Gateway, tmp_path: Path) -> None:
    private_key, _jwks_key = _rsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = jwt.encode(
            {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": "rsa"},
        )
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist empty"}]},
                key=token,
            )
            assert response.status_code == 401, response.text


def test_five_eddsa_requests_warn_at_most_once_per_worker(gateway: Gateway, tmp_path: Path) -> None:
    private_key, jwks_key = _eddsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "EdDSA", subject, kid="ed")
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            for _ in range(5):
                response: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist burst"}]},
                    key=token,
                )
                assert response.status_code == 200, response.text
            log_text: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: _deprecation_lines(text) != [],
                seconds=30,
            )
            warnings: Final = _deprecation_lines(log_text)
            assert 1 <= len(warnings) <= 2, warnings


def test_master_key_request_still_works_on_jwt_enabled_proxy(gateway: Gateway, tmp_path: Path) -> None:
    _, jwks_key = _rsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist master"}]},
            )
            assert response.status_code == 200, response.text


def _spend_row_counts(identifiers: set[str]) -> dict[str, int]:
    if not identifiers:
        return {}
    rows: Final = read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s::text[])',
        ("{" + ",".join(sorted(identifiers)) + "}",),
    )
    counts: Final = {identifier: 0 for identifier in identifiers}
    for row in rows:
        identifier = str(row["request_id"])
        if identifier in counts:
            counts[identifier] += 1
    return counts


def test_mixed_jwt_burst_survives_jwks_restart_and_writes_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    rsa_private, rsa_jwks_key = _rsa_keypair()
    ed_private, ed_jwks_key = _eddsa_keypair()
    port: Final = _free_port()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([rsa_jwks_key, ed_jwks_key])

    with gateway.scenario() as scenario:
        rs_token: Final = jwt.encode(
            {"sub": f"integration-jwt-{uuid.uuid4().hex}", "iat": int(time.time()), "exp": int(time.time()) + 900},
            rsa_private,
            algorithm="RS256",
            headers={"kid": "rsa"},
        )
        ed_token: Final = jwt.encode(
            {"sub": f"integration-jwt-{uuid.uuid4().hex}", "iat": int(time.time()), "exp": int(time.time()) + 900},
            ed_private,
            algorithm="EdDSA",
            headers={"kid": "ed"},
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": f"http://127.0.0.1:{port}/jwks"},
            config=_jwt_auth_config(tmp_path),
            workers=2,
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(
                scenario.delete_user, str(jwt.decode(rs_token, options={"verify_signature": False})["sub"])
            )
            scenario.cleanups.callback(
                scenario.delete_user, str(jwt.decode(ed_token, options={"verify_signature": False})["sub"])
            )

            def chat(token: str, index: int, stream: bool = False) -> httpx.Response:
                with owned.gateway.client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": f"d1-{index}"}],
                        **({"stream": True} if stream else {}),
                    },
                    headers={"Authorization": f"Bearer {token}"},
                ) as response:
                    return httpx.Response(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        content=response.read(),
                    )

            with wire_server(respond, port=port):
                warmup_rs: Final = chat(rs_token, 0)
                warmup_ed: Final = chat(ed_token, 1)
                assert warmup_rs.status_code == 200, warmup_rs.text
                assert warmup_ed.status_code == 200, warmup_ed.text

            statuses: Final = []
            bodies: Final = []
            with ThreadPoolExecutor(max_workers=15) as pool:
                futures = [
                    pool.submit(chat, rs_token if index % 2 == 0 else ed_token, index, index % 5 == 0)
                    for index in range(30)
                ]
                for future in futures:
                    result = future.result(timeout=60)
                    statuses.append(result.status_code)
                    bodies.append(result.text)
            assert all(status in (200, 401) for status in statuses), (statuses, bodies[:3])
            assert any(status == 200 for status in statuses), statuses

            with wire_server(respond, port=port):
                fresh_rs: Final = chat(rs_token, 100)
                fresh_ed: Final = chat(ed_token, 101)
                assert fresh_rs.status_code == 200, fresh_rs.text
                assert fresh_ed.status_code == 200, fresh_ed.text

            succeeded: Final = {
                json.loads(text)["id"]
                for status, text in zip(statuses, bodies)
                if status == 200 and "id" in text and text.strip().startswith("{")
            }
            succeeded.update(json.loads(text)["id"] for text in (fresh_rs.text, fresh_ed.text) if "id" in text)
            counts: Final = eventually(
                lambda: _spend_row_counts(succeeded),
                lambda value: value == {identifier: 1 for identifier in succeeded},
                seconds=70,
            )
            assert all(count == 1 for count in counts.values()), counts


def test_burst_survives_killed_worker_and_stays_alive(gateway: Gateway, tmp_path: Path) -> None:
    rsa_private, rsa_jwks_key = _rsa_keypair()
    ed_private, ed_jwks_key = _eddsa_keypair()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply_keys([rsa_jwks_key, ed_jwks_key])

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        rs_token: Final = jwt.encode(
            {"sub": f"integration-jwt-{uuid.uuid4().hex}", "iat": int(time.time()), "exp": int(time.time()) + 900},
            rsa_private,
            algorithm="RS256",
            headers={"kid": "rsa"},
        )
        ed_token: Final = jwt.encode(
            {"sub": f"integration-jwt-{uuid.uuid4().hex}", "iat": int(time.time()), "exp": int(time.time()) + 900},
            ed_private,
            algorithm="EdDSA",
            headers={"kid": "ed"},
        )
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path), workers=2
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(
                scenario.delete_user, str(jwt.decode(rs_token, options={"verify_signature": False})["sub"])
            )
            scenario.cleanups.callback(
                scenario.delete_user, str(jwt.decode(ed_token, options={"verify_signature": False})["sub"])
            )

            def chat(token: str, index: int) -> httpx.Response:
                return owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": f"d2-{index}"}]},
                    key=token,
                )

            with ThreadPoolExecutor(max_workers=12) as pool:
                futures = [pool.submit(chat, rs_token if index % 2 == 0 else ed_token, index) for index in range(24)]
                children: Final = psutil.Process(owned.process.pid).children(recursive=True)
                assert children, "owned proxy reported no worker children"
                children[0].kill()
                results: Final = []
                for future in futures:
                    try:
                        result = future.result(timeout=60)
                        results.append(result.status_code)
                    except httpx.TransportError:
                        results.append(0)
            assert all(status in (200, 401, 0) for status in results), results
            assert any(status == 200 for status in results), results
            liveliness: Final = eventually(
                lambda: owned.gateway.client.get("/health/liveliness"),
                lambda response: response.status_code == 200,
                seconds=45,
            )
            assert liveliness.status_code == 200, liveliness.text

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Final
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, write_rows
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server

_MODEL: Final = "litellm-test-native-unpriced"
_FALLBACK_MODEL: Final = "litellm-test-native-fallback"
_CATALOG: Final = {"models": [{"slug": _MODEL, "supported_reasoning_levels": ["medium"], "native_field": True}]}


@dataclass(frozen=True, slots=True)
class CodexRig:
    gateway: Gateway
    provider: Wire
    sink: Wire
    tunnel: ForwardProxy
    log: Path
    cancellation: Event


def _reply(request: Request, cancellation: Event) -> Reply:
    if request.target.startswith("/backend-api/codex/models"):
        return Reply(body=json.dumps(_CATALOG).encode(), headers={"x-request-id": "native-catalog"})
    body: Final = json.loads(request.body)
    marker: Final = body["input"]
    if marker.startswith("redirect-"):
        return Reply(
            status=307, body=b'{"error":"redirect"}', headers={"location": "https://untrusted.example/collect"}
        )
    if marker.startswith("unauthorized-"):
        return Reply(status=401, body=b'{"error":{"message":"synthetic expired token"}}')
    if marker.startswith("error-"):
        return Reply(status=429, body=b'{"error":{"message":"synthetic quota"}}', headers={"retry-after": "3"})
    terminal: Final = {
        "id": f"resp_{marker}",
        "object": "response",
        "created_at": 1786374786,
        "model": body["model"],
        "status": "completed",
        "output": [
            {
                "id": f"msg_{marker}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": marker, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }
    if not body.get("stream"):
        return Reply(body=json.dumps(terminal).encode(), headers={"x-request-id": marker})
    first: Final = f'event: response.output_text.delta\ndata: {{"type":"response.output_text.delta","delta":"{marker}"}}\n\n'.encode()
    last: Final = f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': terminal})}\n\n".encode()
    return Reply(
        content_type="text/event-stream",
        chunks=(first, last),
        gate_after_first=cancellation if marker.startswith("cancel-") else None,
        headers={"x-request-id": marker, "x-codex-primary-used-percent": "7"},
    )


@pytest.fixture(scope="module")
def codex_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CodexRig]:
    directory: Final = tmp_path_factory.mktemp("codex-native")
    cert, key = write_self_signed_cert(directory, ("chatgpt.com",))
    cancellation: Final = Event()

    def reply(request: Request) -> Reply:
        return _reply(request, cancellation)

    def callback(_: Request) -> Reply:
        return Reply()

    with wire_server(reply, tls=server_context(cert, key)) as provider, wire_server(callback) as sink:
        port: Final = urlsplit(provider.url).port
        assert port is not None
        with tunnelling_forward_proxy(port) as tunnel:
            config: Final = directory / "codex.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "general_settings": {
                            "master_key": "os.environ/LITELLM_MASTER_KEY",
                            "database_url": "os.environ/DATABASE_URL",
                            "store_prompts_in_spend_logs": True,
                            "proxy_batch_write_at": 1,
                        },
                        "litellm_settings": {
                            "callbacks": ["generic_api"],
                            "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
                            "model_alias_map": {_MODEL: "must-not-rewrite"},
                            "disable_aiohttp_transport": True,
                        },
                        "model_list": [
                            {
                                "model_name": model,
                                "litellm_params": {"model": f"openai/{model}", "api_key": "synthetic-unused"},
                            }
                            for model in (_MODEL, _FALLBACK_MODEL)
                        ],
                    }
                )
            )
            with owned_proxy_process(
                session_gateway,
                directory,
                {
                    "HTTPS_PROXY": tunnel.url,
                    "SSL_VERIFY": str(cert),
                    "SSL_CERT_FILE": str(cert),
                    "NO_PROXY": "127.0.0.1,localhost,::1",
                    "GENERIC_LOGGER_ENDPOINT": sink.url,
                    "LITELLM_LOG": "DEBUG",
                },
                config=config,
            ) as owned:
                yield CodexRig(owned.gateway, provider, sink, tunnel, owned.log, cancellation)
            cancellation.set()


def _headers(key: str, subscription: str) -> dict[str, str]:
    return {
        "x-litellm-api-key": key,
        "authorization": f"Bearer {subscription}",
        "chatgpt-account-id": f"account-{sha256(subscription.encode()).hexdigest()}",
    }


def test_form_media_type_cannot_authorize_a_different_json_model(codex_rig: CodexRig) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    payload: Final = json.dumps({"model": _FALLBACK_MODEL, "input": f"&model={_MODEL}&"}).encode()
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL])
        response: Final = candidate.client.post(
            "/backend-api/codex/responses",
            content=payload,
            headers={**_headers(key, f"synthetic-{marker}"), "content-type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code in (400, 403), response.text
        assert not tuple(
            request for request in codex_rig.provider.drain() if marker in request.headers.get("authorization", "")
        )


@pytest.mark.parametrize("encoding", ["zstd", "gzip"])
def test_encoded_requests_are_rejected_without_upstream_traffic(codex_rig: CodexRig, encoding: str) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL])
        response: Final = candidate.client.post(
            "/backend-api/codex/responses",
            json={"model": _MODEL, "input": marker},
            headers={**_headers(key, f"synthetic-{marker}"), "content-encoding": encoding},
        )
        assert response.status_code == 415, response.text
        assert "enable_request_compression = false" in response.json()["detail"]
        assert not tuple(
            request for request in codex_rig.provider.drain() if marker in request.headers.get("authorization", "")
        )


def test_authorized_budget_fallback_is_the_model_forwarded_upstream(codex_rig: CodexRig) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL, _FALLBACK_MODEL])
        write_rows(
            'UPDATE "LiteLLM_VerificationToken" SET model_max_budget = %s::jsonb, budget_fallbacks = %s::jsonb WHERE token = %s',
            (
                json.dumps({_MODEL: {"max_budget": 0.0, "budget_duration": "1h"}}),
                json.dumps({_MODEL: [_FALLBACK_MODEL]}),
                sha256(key.encode()).hexdigest(),
            ),
        )
        candidate.post("/key/update", {"key": key, "key_alias": f"fallback-{marker}"})
        response: Final = candidate.client.post(
            "/backend-api/codex/responses",
            json={"model": _MODEL, "input": marker},
            headers=_headers(key, f"synthetic-{marker}"),
        )
        assert response.status_code == 200, response.text
        observed: Final = tuple(
            request for request in codex_rig.provider.drain() if marker in request.headers.get("authorization", "")
        )
        assert len(observed) == 1
        assert json.loads(observed[0].body)["model"] == _FALLBACK_MODEL
        assert response.json()["model"] == _FALLBACK_MODEL


def test_concurrent_callers_rotate_tokens_without_crossing_accounts(codex_rig: CodexRig) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    with candidate.scenario() as scenario:
        first: Final = scenario.key(models=[_MODEL])
        second: Final = scenario.key(models=[_MODEL])
        callers: Final = (
            (first, f"synthetic-a-{marker}"),
            (second, f"synthetic-b-{marker}"),
            (first, f"synthetic-rotated-{marker}"),
        )

        def catalog(caller: tuple[str, str]) -> httpx.Response:
            return candidate.client.get(
                "/backend-api/codex/models?client_version=synthetic-version", headers=_headers(*caller)
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses: Final = tuple(pool.map(catalog, callers))
        assert tuple(response.json() for response in responses) == (_CATALOG,) * len(callers)
        observed: Final = tuple(
            request for request in codex_rig.provider.drain() if marker in request.headers.get("authorization", "")
        )
        assert len(observed) == len(callers)
        assert {request.headers["authorization"] for request in observed} == {f"Bearer {token}" for _, token in callers}
        for request in observed:
            subscription: Final = request.headers["authorization"].removeprefix("Bearer ")
            assert request.headers["chatgpt-account-id"] == _headers("unused", subscription)["chatgpt-account-id"]
            assert request.target == "/backend-api/codex/models?client_version=synthetic-version"
            assert first not in str(request.headers) and second not in str(request.headers)
        assert all(target == "chatgpt.com:443" for target in codex_rig.tunnel.targets())


@pytest.mark.parametrize("stream", [False, True])
def test_native_usage_is_attributed_and_secrets_are_absent_from_logs_and_rows(
    codex_rig: CodexRig, stream: bool
) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    subscription: Final = f"synthetic-subscription-{marker}"
    with candidate.scenario() as scenario:
        team: Final = scenario.team(models=[_MODEL])
        user: Final = scenario.user()
        key: Final = scenario.key(models=[_MODEL], user_id=user, team_id=team)
        payload: Final = {
            "model": _MODEL,
            "input": marker,
            "stream": stream,
            "custom_body": {"target": "https://untrusted.example/collect"},
        }
        response: Final = candidate.client.post(
            "/backend-api/codex/responses",
            json=payload,
            headers={
                **_headers(key, subscription),
                "content-type": "application/json",
                "x-alias": f"Bearer {key}",
                "x-pass-authorization": f"Bearer {key}",
            },
        )
        assert response.status_code == 200, response.text
        assert marker in response.text
        assert f"resp_{marker}" in response.text
        assert response.headers["x-request-id"] == marker
        observed: Final = tuple(request for request in codex_rig.provider.drain() if marker.encode() in request.body)
        assert len(observed) == 1
        assert json.loads(observed[0].body) == payload
        assert observed[0].headers["authorization"] == f"Bearer {subscription}"
        assert key not in str(observed[0].headers)
        assert "content-encoding" not in observed[0].headers
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT api_key, "user", team_id, prompt_tokens, completion_tokens, total_tokens, spend, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (f"resp_{marker}",),
            ),
            lambda rows: len(rows) == 1,
            seconds=30,
        )
        assert rows[0]["api_key"] == sha256(key.encode()).hexdigest()
        assert rows[0]["user"] == user
        assert rows[0]["team_id"] == team
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"], rows[0]["total_tokens"], rows[0]["spend"]) == (
            11,
            4,
            15,
            0.0,
        )
        assert subscription not in str(rows) and key not in str(rows)
        batches: Final = eventually(
            lambda: tuple(codex_rig.sink.drain()),
            lambda batches: any(marker.encode() in request.body for request in batches),
            seconds=15,
        )
        assert subscription not in str(batches) and key not in str(batches)
        assert subscription not in codex_rig.log.read_text()
        assert key not in codex_rig.log.read_text()


def test_denied_models_and_invalid_keys_never_reach_provider(codex_rig: CodexRig) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL])
        denied: Final = candidate.client.post(
            "/backend-api/codex/responses",
            json={"model": "synthetic-denied-model", "input": marker},
            headers=_headers(key, f"synthetic-{marker}"),
        )
        invalid: Final = candidate.client.get(
            "/backend-api/codex/models", headers=_headers("synthetic-invalid-key", marker)
        )
        missing: Final = candidate.client.get(
            "/backend-api/codex/models", headers={"authorization": f"Bearer {marker}"}
        )
        route_key: Final = scenario.key(models=[_MODEL], allowed_routes=["/chat/completions"])
        denied_route: Final = candidate.client.post(
            "/backend-api/codex/responses",
            json={"model": _MODEL, "input": marker},
            headers=_headers(route_key, f"synthetic-{marker}"),
        )
        assert denied_route.status_code in {401, 403}, denied_route.text
        assert denied.status_code in {401, 403}, denied.text
        assert invalid.status_code in {401, 403}, invalid.text
        assert missing.status_code == 401
        assert not any(
            marker.encode() in request.body or marker in str(request.headers) for request in codex_rig.provider.drain()
        )


@pytest.mark.parametrize("prefix,status", [("redirect", 307), ("error", 429), ("unauthorized", 401)])
def test_upstream_errors_and_redirects_are_not_followed(codex_rig: CodexRig, prefix: str, status: int) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = f"{prefix}-{uuid4().hex}"
    subscription: Final = f"synthetic-secret-{uuid4().hex}"
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL])
        response: Final = candidate.client.post(
            "/backend-api/codex/responses", json={"model": _MODEL, "input": marker}, headers=_headers(key, subscription)
        )
        assert response.status_code == status, response.text
        expected: Final = {
            "redirect": "redirect",
            "error": {"message": "synthetic quota"},
            "unauthorized": {"message": "synthetic expired token"},
        }
        assert response.json()["error"] == expected[prefix]
        tunnels: Final = codex_rig.tunnel.targets()
        assert all(target == "chatgpt.com:443" for target in tunnels)
        assert subscription not in codex_rig.log.read_text()


def test_client_cancellation_releases_the_keys_parallel_request_reservation(codex_rig: CodexRig) -> None:
    candidate: Final = codex_rig.gateway
    marker: Final = uuid4().hex
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=[_MODEL], max_parallel_requests=1)
        try:
            with candidate.client.stream(
                "POST",
                "/backend-api/codex/responses",
                json={"model": _MODEL, "input": f"cancel-{marker}", "stream": True},
                headers=_headers(key, f"synthetic-{marker}"),
            ) as response:
                assert response.status_code == 200
                assert marker.encode() in next(response.iter_bytes())
        finally:
            codex_rig.cancellation.set()
        followup: Final = eventually(
            lambda: candidate.client.post(
                "/backend-api/codex/responses",
                json={"model": _MODEL, "input": marker, "stream": True},
                headers=_headers(key, f"synthetic-{marker}"),
            ),
            lambda response: response.status_code != 429,
        )
        assert followup.status_code == 200, followup.text
        assert f"resp_{marker}" in followup.text
        assert eventually(codex_rig.provider.open_connections, lambda count: count == 0) == 0

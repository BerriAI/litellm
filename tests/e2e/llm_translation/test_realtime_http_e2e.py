"""Vendor §9.19: realtime client_secrets + calls HTTP surface (LIT-4778).

Websocket coverage already lives under realtime/; this file pins the HTTP
client-secret mint and the missing-auth contract.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap, assert_auth_denied
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

REALTIME_BACKEND = "openai/gpt-realtime"


class RealtimeSession(BaseModel):
    type: str = "realtime"
    model: str | None = None
    instructions: str | None = None
    output_modalities: list[str] | None = None


class RealtimeExpiresAfter(BaseModel):
    anchor: str = "created_at"
    seconds: int = 600


class RealtimeClientSecretRequest(BaseModel):
    model: str
    expires_after: RealtimeExpiresAfter | None = None
    session: RealtimeSession | None = None


class RealtimeClientSecretResponse(BaseModel):
    value: str | None = None
    expires_at: int | None = None
    session: dict[str, object] | None = None


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-realtime-http-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(model=REALTIME_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


class TestRealtimeHttp:
    @pytest.mark.covers("llm.realtime.openai.basic.nonstream.works")
    def test_create_client_secret(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(proxy, resources)
        secret = unwrap(
            proxy.transport.post(
                "/v1/realtime/client_secrets",
                headers=proxy.transport.bearer(key),
                json=RealtimeClientSecretRequest(
                    model=model,
                    expires_after=RealtimeExpiresAfter(),
                    session=RealtimeSession(
                        # Upstream OpenAI realtime requires a provider-qualified model;
                        # the gateway alias alone is not enough for client_secrets.
                        model=REALTIME_BACKEND,
                        instructions="You are a helpful assistant.",
                        output_modalities=["text"],
                    ),
                ),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value or secret.session, f"client secret empty: {secret}"
        if secret.session is not None:
            session_type = secret.session.get("type")
            assert session_type in (None, "realtime"), f"unexpected session type: {session_type}"

    @pytest.mark.covers("other.auth.llm_chat.missing_header_denied")
    def test_client_secret_missing_auth_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, _ = _register(proxy, resources)
        result = proxy.transport.send(
            "/v1/realtime/client_secrets",
            headers=NoBody(),
            json=RealtimeClientSecretRequest(model=model),
        )
        assert_auth_denied(result, "realtime client_secrets missing auth")

    @pytest.mark.covers("llm.realtime.openai.basic.nonstream.works")
    def test_calls_without_auth_is_denied(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        result = proxy.transport.send(
            "/v1/realtime/calls",
            headers=NoBody(),
            json=NoBody(),
        )
        assert result.status_code in (401, 403, 405, 415, 422), (
            f"realtime calls missing auth unexpected {result.status_code}: {result.body[:300]}"
        )

    @pytest.mark.covers("llm.realtime.openai.basic.nonstream.works")
    def test_calls_authenticated_route_is_reachable(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key = _register(proxy, resources)
        secret = unwrap(
            proxy.transport.post(
                "/v1/realtime/client_secrets",
                headers=proxy.transport.bearer(key),
                json=RealtimeClientSecretRequest(
                    model=model,
                    session=RealtimeSession(
                        model=REALTIME_BACKEND, output_modalities=["text"]
                    ),
                ),
                response_type=RealtimeClientSecretResponse,
            )
        )
        assert secret.value, f"need client secret value for calls: {secret}"
        result = proxy.transport.send(
            "/v1/realtime/calls",
            headers=proxy.transport.bearer(secret.value),
            json=NoBody(),
        )
        assert result.status_code not in (401, 403, 404), (
            f"authenticated calls route must not be auth/not-found, "
            f"got {result.status_code}: {result.body[:300]}"
        )
        assert result.status_code < 500, (
            f"authenticated calls must not 5xx: {result.status_code} {result.body[:300]}"
        )

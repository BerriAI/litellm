from functools import partial
from pathlib import Path
from secrets import token_urlsafe
from typing import Final

import httpx2
import pytest
from integration._support.oauth_server import oauth_server
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.mcp_server.outbound_credentials.adapter import to_server_spec, to_subject
from litellm.proxy.mcp_server.outbound_credentials.client_credentials import ClientCredentialsTokenSource
from litellm.proxy.mcp_server.outbound_credentials.dual_cache_token_backend import DualCacheTokenCacheBackend
from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
from litellm.proxy.mcp_server.outbound_credentials.result import Ok
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import OAuthTokenCacheCodec
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


async def _call(server: MCPServer, upstream: Wire, redis: RedisCache) -> tuple[str, ...]:
    spec: Final = to_server_spec(server)
    assert spec is not None
    codec: Final = OAuthTokenCacheCodec(encrypt_value_helper, partial(decrypt_value_helper, key="m2m"))
    backend: Final = DualCacheTokenCacheBackend(DualCache(redis_cache=redis), codec)
    provider: Final = UpstreamCredentialProvider(
        client_credentials_source=ClientCredentialsTokenSource(backend=backend)
    )
    result: Final = await provider.resolve_credentials(to_subject(None, None), spec)
    assert isinstance(result, Ok)
    async with httpx2.AsyncClient(auth=result.ok, trust_env=False) as client:
        response: Final = await client.get(upstream.url)
        assert response.status_code == 200
    return tuple(request.headers["authorization"] for request in upstream.drain())


@pytest.mark.asyncio
async def test_resolved_profiles_share_their_own_token_and_retry_only_the_rejected_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", token_urlsafe())
    with oauth_server() as issuer, owned_redis(tmp_path) as redis:

        def respond(request: Request) -> Reply:
            token: Final = request.headers["authorization"].removeprefix("Bearer ")
            return Reply(status=200 if issuer.is_live(token) else 401)

        with wire_server(respond) as upstream:
            secret: Final = token_urlsafe()
            issuer.clients["app"] = secret
            default: Final = MCPServer(
                server_id="server",
                name="tickets",
                transport=MCPTransport.http,
                url=upstream.url,
                auth_type=MCPAuth.oauth2,
                oauth2_flow="client_credentials",
                token_url=issuer.issuer + "/token",
                client_id="app",
                client_secret=secret,
            )
            named: Final = default.model_copy(update={"auth_profile_id": "Build"})
            shared: Final = RedisCache(host=redis.host, port=redis.port)
            try:
                (default_token,) = await _call(default, upstream, shared)
                (original,) = await _call(named, upstream, shared)
                assert default_token != original
                with issuer.lock:
                    issuer.revoked.add(original.removeprefix("Bearer "))
                rejected, replacement = await _call(
                    named.model_copy(update={"auth_profile_id": "BUILD"}), upstream, shared
                )
                assert rejected == original
                assert replacement not in (original, default_token)
                assert await _call(named, upstream, shared) == (replacement,)
                assert await _call(default, upstream, shared) == (default_token,)
                assert len(issuer.token_requests()) == 3
            finally:
                await shared.disconnect()

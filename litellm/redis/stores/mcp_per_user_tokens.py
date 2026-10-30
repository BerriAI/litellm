import ast
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.constants import MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.redis.transport.contract import RedisFailure, RedisTransport

TOKEN_KEY_PREFIX: Final = MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX + ":"
_BOUND_PREFIX: Final = "litellm-bound-oauth-v1:"
_BOUND_PAYLOAD: Final = TypeAdapter(dict[str, str])
_STRING: Final = TypeAdapter(str)


def per_user_token_key(user_id: str, server_id: str, *, prefix: str = TOKEN_KEY_PREFIX) -> str:
    return f"{prefix}{user_id}:{server_id}"


@dataclass(frozen=True, slots=True)
class OAuthTokenCacheCodec:
    encrypt: Callable[[str], str]
    decrypt: Callable[[str], str | None]

    def encode(self, token: OAuthToken) -> str:
        if token.identity_binding_proof is not None:
            return self.encrypt(
                _BOUND_PREFIX
                + json.dumps(
                    {"access_token": token.access_token, "identity_binding_proof": token.identity_binding_proof}
                )
            )
        return self.encrypt(token.access_token)

    def decode(self, blob: str) -> OAuthToken | None:
        access_token: Final = self.decrypt(blob)
        if not access_token:
            return None
        if access_token.startswith(_BOUND_PREFIX):
            try:
                payload: Final = _BOUND_PAYLOAD.validate_json(access_token[len(_BOUND_PREFIX) :])
            except ValidationError:
                return None
            bearer: Final = payload.get("access_token")
            proof: Final = payload.get("identity_binding_proof")
            if not bearer or not proof:
                return None
            return OAuthToken(access_token=bearer, identity_binding_proof=proof)
        return OAuthToken(access_token=access_token, refresh_token=None)


class McpPerUserTokenStore(Protocol):
    async def read(self, user_id: str, server_id: str) -> str | None | RedisFailure: ...

    async def write(
        self, user_id: str, server_id: str, ciphertext: str, *, ttl: timedelta, defer: bool = False
    ) -> bool | RedisFailure: ...

    async def delete(self, user_id: str, server_id: str) -> int | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportMcpPerUserTokenStore:
    transport: RedisTransport
    prefix: str = TOKEN_KEY_PREFIX

    async def read(self, user_id: str, server_id: str) -> str | None | RedisFailure:
        reply: Final = await self.transport.get(per_user_token_key(user_id, server_id, prefix=self.prefix))
        if reply is None or isinstance(reply, RedisFailure):
            return reply
        try:
            text: Final = reply.decode("utf-8")
            try:
                return _STRING.validate_json(text)
            except ValidationError:
                return _STRING.validate_python(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            return RedisFailure(kind="decode", message="Invalid MCP per-user token cache value")

    async def write(
        self, user_id: str, server_id: str, ciphertext: str, *, ttl: timedelta, defer: bool = False
    ) -> bool | RedisFailure:
        key: Final = per_user_token_key(user_id, server_id, prefix=self.prefix)
        wire: Final = json.dumps(ciphertext)
        batch: Final = self.transport.request_batch() if defer else None
        if batch is not None:
            batch.set_seconds(key, wire, ttl=ttl)
            return True
        return await self.transport.set_seconds(key, wire, ttl=ttl)

    async def delete(self, user_id: str, server_id: str) -> int | RedisFailure:
        return await self.transport.delete(per_user_token_key(user_id, server_id, prefix=self.prefix))

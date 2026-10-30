import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import (
    GITHUB_COPILOT_DEFAULT_HOST,
    USER_PROVIDER_CREDENTIAL_CACHE_PREFIX,
    USER_PROVIDER_CREDENTIAL_NOT_CONNECTED,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.redis.transport.contract import RedisFailure, RedisTransport

_JSON: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class InvalidCredentialValue:
    pass


class GithubCopilotUserConnectionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    access_token: str = Field(repr=False)
    github_login: str
    github_host: str = GITHUB_COPILOT_DEFAULT_HOST


def user_provider_credential_key(user_id: str, credential_name: str) -> str:
    pair: Final = json.dumps([user_id, credential_name], separators=(",", ":"))
    return f"{USER_PROVIDER_CREDENTIAL_CACHE_PREFIX}:{pair}"


def encode_credential(payload: GithubCopilotUserConnectionPayload) -> str:
    return encrypt_value_helper(payload.model_dump_json())


def decode_credential(stored: str) -> GithubCopilotUserConnectionPayload | None:
    decrypted: Final = decrypt_value_helper(
        value=stored, key="user_provider_credential", exception_type="debug", return_original_value=False
    )
    if decrypted is None:
        return None
    try:
        return GithubCopilotUserConnectionPayload.model_validate_json(decrypted)
    except ValidationError:
        return None


def _decode_reply(reply: bytes | None | RedisFailure) -> str | None | InvalidCredentialValue | RedisFailure:
    if not isinstance(reply, bytes):
        return reply
    try:
        decoded: Final = reply.decode("utf-8")
        try:
            value: Final = _JSON.validate_json(decoded)
        except ValidationError:
            legacy: Final = _JSON.validate_python(ast.literal_eval(decoded))
            return legacy if isinstance(legacy, str) or legacy is None else InvalidCredentialValue()
        return value if isinstance(value, str) or value is None else InvalidCredentialValue()
    except (ValueError, SyntaxError):
        return RedisFailure(kind="decode", message="Invalid user provider credential cache value")


class UserProviderCredentialStore(Protocol):
    async def read(self, user_id: str, credential_name: str) -> str | None | InvalidCredentialValue | RedisFailure: ...

    async def fill(
        self, user_id: str, credential_name: str, ciphertext: str | None, *, ttl: timedelta
    ) -> bool | RedisFailure: ...

    async def connect(
        self, user_id: str, credential_name: str, payload: GithubCopilotUserConnectionPayload, *, ttl: timedelta
    ) -> bool | RedisFailure: ...

    async def disconnect(self, user_id: str, credential_name: str, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def delete(self, user_id: str, credential_name: str) -> int | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportUserProviderCredentialStore:
    transport: RedisTransport

    async def read(self, user_id: str, credential_name: str) -> str | None | InvalidCredentialValue | RedisFailure:
        return _decode_reply(await self.transport.get(user_provider_credential_key(user_id, credential_name)))

    async def fill(
        self, user_id: str, credential_name: str, ciphertext: str | None, *, ttl: timedelta
    ) -> bool | RedisFailure:
        return await self.transport.set_seconds(
            user_provider_credential_key(user_id, credential_name),
            json.dumps(ciphertext if ciphertext is not None else USER_PROVIDER_CREDENTIAL_NOT_CONNECTED),
            ttl=ttl,
            nx=True,
        )

    async def connect(
        self, user_id: str, credential_name: str, payload: GithubCopilotUserConnectionPayload, *, ttl: timedelta
    ) -> bool | RedisFailure:
        try:
            ciphertext: Final = encode_credential(payload)
        except Exception:  # noqa: BLE001  # the caller handles encryption failures like failed writes
            return RedisFailure(kind="decode", message="Cannot encode user provider credential")
        return await self.transport.set_seconds(
            user_provider_credential_key(user_id, credential_name), json.dumps(ciphertext), ttl=ttl
        )

    async def disconnect(self, user_id: str, credential_name: str, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.transport.set_seconds(
            user_provider_credential_key(user_id, credential_name),
            json.dumps(USER_PROVIDER_CREDENTIAL_NOT_CONNECTED),
            ttl=ttl,
        )

    async def delete(self, user_id: str, credential_name: str) -> int | RedisFailure:
        return await self.transport.delete(user_provider_credential_key(user_id, credential_name))

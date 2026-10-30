import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_SETTINGS: Final = TypeAdapter(dict[str, object])


def saml_metadata_key(source: str) -> str:
    return f"saml_idp_settings:{hashlib.sha256(source.encode()).hexdigest()}"


class SAMLMetadataStore(Protocol):
    async def get(self, source: str) -> Mapping[str, object] | None | RedisFailure: ...

    async def put(self, source: str, settings: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSAMLMetadataStore:
    transport: RedisTransport

    async def get(self, source: str) -> Mapping[str, object] | None | RedisFailure:
        raw: Final = await self.transport.get(saml_metadata_key(source))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _SETTINGS.validate_json(raw)
        except ValidationError:
            try:
                return _SETTINGS.validate_python(ast.literal_eval(raw.decode("utf-8")))
            except (ValueError, SyntaxError, UnicodeError):
                return RedisFailure(kind="decode", message="invalid SAML IdP settings")

    async def put(self, source: str, settings: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        reply: Final = await self.transport.set_seconds(saml_metadata_key(source), json.dumps(settings), ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None

import asyncio
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.authenticator import (
    TOKEN_EXPIRY_SKEW_SECONDS,
    ChatGPTOAuthClient,
    arefresh_chatgpt_tokens,
    chatgpt_login_values,
    new_chatgpt_oauth_client,
)
from litellm.llms.chatgpt.common_utils import ChatGPTAuthError
from litellm.llms.chatgpt.subscription_login import (
    CHATGPT_OWNER_KEY_HASH_KEY,
    CHATGPT_OWNER_TEAM_ID_KEY,
    CHATGPT_PROVIDER,
    is_chatgpt_subscription_login,
)
from litellm.proxy.common_utils.credential_hydration import decrypted_credential, stored_credential_provider
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    InProcessRefreshCoordinator,
    OAuthToken,
    RefreshCoordinator,
)
from litellm.types.utils import CredentialItem

_COORDINATOR_NAMESPACE: Final = "chatgpt_subscription"
_IN_PROCESS_COORDINATOR: Final = InProcessRefreshCoordinator()
_NO_LOGINS: Final[Mapping[str, str]] = MappingProxyType({})
_CREDENTIAL_INFO_ADAPTER: Final = TypeAdapter(Mapping[str, object])


_REFRESH_TOKEN_KEY: Final = "chatgpt_refresh_token"


class CredentialRows(Protocol):
    async def find_by_name(self, credential_name: str) -> CredentialItem | None: ...

    async def update_if_value_unchanged(
        self, credential_name: str, value_key: str, stored_value: str, data: Mapping[str, object]
    ) -> bool: ...


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _credential_info(credential: CredentialItem | None) -> Mapping[str, object]:
    return _CREDENTIAL_INFO_ADAPTER.validate_python(
        (credential.credential_info if credential is not None else None)  # pyright: ignore[reportUnknownMemberType]  # untyped credential field
        or {}
    )


def chatgpt_login_owner(credential: CredentialItem) -> tuple[str | None, str | None]:
    info: Final = _credential_info(credential)
    team_id: Final = _non_empty_str(info.get(CHATGPT_OWNER_TEAM_ID_KEY))
    key_hash: Final = _non_empty_str(info.get(CHATGPT_OWNER_KEY_HASH_KEY))
    if (team_id is None) == (key_hash is None):
        return None, None
    return team_id, key_hash


def _owned_by(credential: CredentialItem, team_id: str | None, key_hash: str | None) -> bool:
    if not is_chatgpt_subscription_login(credential):
        return False
    owner_team_id, owner_key_hash = chatgpt_login_owner(credential)
    return (owner_team_id is not None and owner_team_id == team_id) or (
        owner_key_hash is not None and owner_key_hash == key_hash
    )


def owned_chatgpt_subscription_logins(team_id: str | None, key_hash: str | None) -> tuple[CredentialItem, ...]:
    return tuple(credential for credential in litellm.credential_list if _owned_by(credential, team_id, key_hash))


def _expires_at(values: Mapping[str, object]) -> float | None:
    raw: Final = values.get("chatgpt_expires_at")
    if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
        return None
    try:
        parsed: Final = float(raw)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _fresh_access_token(values: Mapping[str, object], now: float) -> str | None:
    access_token: Final = _non_empty_str(values.get("chatgpt_access_token"))
    expires_at: Final = _expires_at(values)
    if access_token is None or expires_at is None or now >= expires_at - TOKEN_EXPIRY_SKEW_SECONDS:
        return None
    return access_token


def _in_memory_values(credential_name: str) -> Mapping[str, object] | None:
    current: Final = CredentialAccessor.find_credential(credential_name)
    return MappingProxyType(dict(current.credential_values)) if current is not None else None


@dataclass(frozen=True, slots=True)
class ChatGPTLoginRefresher:
    rows: CredentialRows
    oauth_client: ChatGPTOAuthClient
    coordinator: RefreshCoordinator
    shared_coordinator: RefreshCoordinator | None = None
    clock: Callable[[], float] = time.time
    encrypt: Callable[[str], str] = encrypt_value_helper

    async def access_token(self, owned: CredentialItem) -> str | None:
        owner: Final = chatgpt_login_owner(owned)
        refresh: Final = partial(self._refresh, owned.credential_name, owner)
        reread: Final = partial(self._reread, owned.credential_name, owner)
        flight: Final = json.dumps([owned.credential_name, *owner])
        shared: Final = self.shared_coordinator
        token: Final = await self.coordinator.run(
            _COORDINATOR_NAMESPACE,
            flight,
            refresh if shared is None else partial(shared.run, _COORDINATOR_NAMESPACE, flight, refresh, reread),
            reread,
        )
        return token.access_token if token is not None else None

    async def _stored(self, credential_name: str) -> CredentialItem | None:
        try:
            return await self.rows.find_by_name(credential_name)
        except Exception:  # noqa: BLE001  # a DB outage fails this login closed instead of breaking the request
            verbose_proxy_logger.warning(
                "Could not read ChatGPT subscription login %s from the database", credential_name
            )
            return None

    def _remember(
        self, stored: CredentialItem, values: Mapping[str, object], seen: Mapping[str, object] | None
    ) -> None:
        if seen is None or _in_memory_values(stored.credential_name) != seen:
            return
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name=stored.credential_name,
                    display_name=stored.display_name,
                    credential_values=dict(values),
                    credential_info=dict(_credential_info(stored)),
                )
            ]
        )

    def _fresh_from(
        self, stored: CredentialItem, seen: Mapping[str, object] | None
    ) -> tuple[OAuthToken | None, Mapping[str, object]]:
        values: Final = decrypted_credential(stored).credential_values
        access_token: Final = _fresh_access_token(values, self.clock())
        if access_token is None:
            return None, values
        self._remember(stored, values, seen)
        return OAuthToken(access_token=access_token, expires_at=_expires_at(values)), values

    async def _reread(self, credential_name: str, owner: tuple[str | None, str | None]) -> OAuthToken | None:
        seen: Final = _in_memory_values(credential_name)
        stored: Final = await self._stored(credential_name)
        if stored is None or not _owned_by(stored, *owner):
            return None
        token, _ = self._fresh_from(stored, seen)
        return token

    async def _refresh(self, credential_name: str, owner: tuple[str | None, str | None]) -> OAuthToken | None:
        seen: Final = _in_memory_values(credential_name)
        stored: Final = await self._stored(credential_name)
        if stored is None or not _owned_by(stored, *owner):
            return None
        fresh, values = self._fresh_from(stored, seen)
        if fresh is not None:
            return fresh
        refresh_token: Final = _non_empty_str(values.get(_REFRESH_TOKEN_KEY))
        stored_refresh_token: Final = _non_empty_str(stored.credential_values.get(_REFRESH_TOKEN_KEY))
        if refresh_token is None or stored_refresh_token is None:
            return None
        try:
            tokens: Final = await arefresh_chatgpt_tokens(self.oauth_client, refresh_token)
        except ChatGPTAuthError as exc:
            verbose_proxy_logger.warning(
                "ChatGPT subscription login %s could not be refreshed (status %s). Sign in again for it.",
                credential_name,
                exc.status_code,
            )
            return None
        login_values: Final = chatgpt_login_values(tokens)
        if not await self._save(stored, stored_refresh_token, login_values):
            return await self._reread(credential_name, owner)
        refreshed_values: Final = {**values, **login_values}
        self._remember(stored, refreshed_values, seen)
        return OAuthToken(access_token=tokens.access_token, expires_at=_expires_at(refreshed_values))

    async def _save(self, stored: CredentialItem, stored_refresh_token: str, login_values: Mapping[str, str]) -> bool:
        encrypted: Final = {key: self.encrypt(value) for key, value in login_values.items()}
        try:
            return await self.rows.update_if_value_unchanged(
                stored.credential_name,
                _REFRESH_TOKEN_KEY,
                stored_refresh_token,
                {"credential_values": json.dumps({**stored.credential_values, **encrypted})},
            )
        except Exception:  # noqa: BLE001  # the rotated tokens still serve this replica until they expire
            verbose_proxy_logger.warning(
                "ChatGPT subscription login %s was refreshed but could not be saved to the database. "
                "Sign in again for it if requests start failing after the new token expires.",
                stored.credential_name,
            )
            return True


def runtime_chatgpt_login_refresher() -> ChatGPTLoginRefresher | None:
    from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient
    from litellm.proxy.mcp_server.outbound_credentials.runtime_refresh_coordinator import (
        runtime_refresh_coordinator,
    )
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.credentials_repository import CredentialsRepository

    if prisma_client is None:
        return None
    return ChatGPTLoginRefresher(
        rows=CredentialsRepository(WriterPinnedClient(prisma_client.db)),
        oauth_client=new_chatgpt_oauth_client(),
        coordinator=_IN_PROCESS_COORDINATOR,
        shared_coordinator=runtime_refresh_coordinator(),
    )


async def resolve_chatgpt_subscription_logins(
    team_id: str | None,
    key_hash: str | None,
    refresher_factory: Callable[[], ChatGPTLoginRefresher | None] = runtime_chatgpt_login_refresher,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, str]:
    owned: Final = owned_chatgpt_subscription_logins(team_id, key_hash)
    if not owned:
        return _NO_LOGINS
    now: Final = clock()
    fresh: Final = {
        credential.credential_name: token
        for credential in owned
        if (token := _fresh_access_token(credential.credential_values, now)) is not None
    }
    stale: Final = tuple(credential for credential in owned if credential.credential_name not in fresh)
    refresher: Final = refresher_factory() if stale else None
    if refresher is None:
        return MappingProxyType(fresh)
    refreshed: Final = await asyncio.gather(*(refresher.access_token(credential) for credential in stale))
    return MappingProxyType(
        {
            **fresh,
            **{
                credential.credential_name: token
                for credential, token in zip(stale, refreshed, strict=True)
                if token is not None
            },
        }
    )


class CredentialWrites(Protocol):
    async def find_by_name(self, credential_name: str) -> CredentialItem | None: ...

    async def create(self, data: Mapping[str, object]) -> object: ...

    async def update_by_name(self, credential_name: str, data: Mapping[str, object]) -> object: ...


_OWNER_KEYS: Final = frozenset({CHATGPT_OWNER_TEAM_ID_KEY, CHATGPT_OWNER_KEY_HASH_KEY})


def blocks_chatgpt_login(existing: CredentialItem | None) -> bool:
    if existing is None or not existing.credential_values or is_chatgpt_subscription_login(existing):
        return False
    return stored_credential_provider(_credential_info(existing).get("custom_llm_provider")) != CHATGPT_PROVIDER


async def save_chatgpt_login(
    rows: CredentialWrites,
    credential_name: str,
    display_name: str | None,
    login_values: Mapping[str, str],
    owner: tuple[str | None, str | None],
    actor: str,
    encrypt: Callable[[str], str] = encrypt_value_helper,
) -> CredentialItem:
    existing: Final = await rows.find_by_name(credential_name)
    owner_team_id, owner_key_hash = owner
    kept_info: Final = {key: value for key, value in _credential_info(existing).items() if key not in _OWNER_KEYS}
    credential_info: Final = {
        **kept_info,
        "custom_llm_provider": CHATGPT_PROVIDER,
        **({CHATGPT_OWNER_TEAM_ID_KEY: owner_team_id} if owner_team_id is not None else {}),
        **({CHATGPT_OWNER_KEY_HASH_KEY: owner_key_hash} if owner_key_hash is not None else {}),
    }
    saved_display_name: Final = display_name or (existing.display_name if existing else None)
    stored_fields: Final = {
        "credential_values": json.dumps({key: encrypt(value) for key, value in login_values.items()}),
        "credential_info": json.dumps(credential_info),
        **({"display_name": saved_display_name} if saved_display_name is not None else {}),
        "updated_by": actor,
    }
    if existing is None:
        await rows.create({"credential_name": credential_name, **stored_fields, "created_by": actor})
    else:
        await rows.update_by_name(credential_name, stored_fields)
    saved: Final = CredentialItem(
        credential_name=credential_name,
        display_name=saved_display_name,
        credential_values=dict(login_values),
        credential_info=credential_info,
    )
    CredentialAccessor.upsert_credentials([saved])
    return saved

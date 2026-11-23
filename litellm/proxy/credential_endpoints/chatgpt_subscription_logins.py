import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.authenticator import (
    TOKEN_EXPIRY_SKEW_SECONDS,
    TOKEN_REFRESH_TIMEOUT_SECONDS,
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
    chatgpt_credential_info,
    chatgpt_login_owner,
    is_chatgpt_subscription_login,
    owns_chatgpt_login,
)
from litellm.proxy.common_utils.credential_hydration import decrypted_credential, stored_credential_provider
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.db.advisory_lock import AdvisoryLockTransaction, AdvisoryLockUnavailable
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    InProcessRefreshCoordinator,
    OAuthToken,
    RefreshCoordinator,
)
from litellm.types.utils import CredentialItem

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

_COORDINATOR_NAMESPACE: Final = "chatgpt_subscription"
_IN_PROCESS_COORDINATOR: Final = InProcessRefreshCoordinator()


_REFRESH_TOKEN_KEY: Final = "chatgpt_refresh_token"
_SAVE_RETRY_DELAYS_SECONDS: Final = (1.0, 2.0)
_REFRESH_LOCK_TIMEOUT: Final = timedelta(seconds=TOKEN_REFRESH_TIMEOUT_SECONDS + sum(_SAVE_RETRY_DELAYS_SECONDS) + 15)

_SAVEPOINT_SQL: Final = "SAVEPOINT chatgpt_login_save"
_ROLLBACK_TO_SAVEPOINT_SQL: Final = "ROLLBACK TO SAVEPOINT chatgpt_login_save"
_RELEASE_SAVEPOINT_SQL: Final = "RELEASE SAVEPOINT chatgpt_login_save"

_InMemoryLogin: TypeAlias = tuple[Mapping[str, object], Mapping[str, object]]


class CredentialRows(Protocol):
    async def find_by_name(self, credential_name: str) -> CredentialItem | None: ...

    async def update_if_value_unchanged(
        self,
        credential_name: str,
        value_match: tuple[str, str],
        info_match: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> bool: ...


class SharedRefresh(Protocol):
    async def run(
        self,
        flight: str,
        refresh: Callable[[CredentialRows], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None: ...


@dataclass(frozen=True, slots=True)
class CoordinatedRefresh:
    coordinator: RefreshCoordinator
    rows: CredentialRows

    async def run(
        self,
        flight: str,
        refresh: Callable[[CredentialRows], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        return await self.coordinator.run(_COORDINATOR_NAMESPACE, flight, partial(refresh, self.rows), reread)


class RefreshLock(Protocol):
    async def while_held(
        self, key: str, action: Callable[[AdvisoryLockTransaction], Awaitable[OAuthToken | None]]
    ) -> OAuthToken | None | AdvisoryLockUnavailable: ...


@dataclass(frozen=True, slots=True)
class _SavepointRows:
    rows: CredentialRows
    transaction: AdvisoryLockTransaction

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        return await self.rows.find_by_name(credential_name)

    async def update_if_value_unchanged(
        self,
        credential_name: str,
        value_match: tuple[str, str],
        info_match: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> bool:
        await self.transaction.execute_raw(_SAVEPOINT_SQL)
        try:
            saved: Final = await self.rows.update_if_value_unchanged(credential_name, value_match, info_match, data)
        except Exception:
            await self.transaction.execute_raw(_ROLLBACK_TO_SAVEPOINT_SQL)
            raise
        await self.transaction.execute_raw(_RELEASE_SAVEPOINT_SQL)
        return saved


@dataclass(frozen=True, slots=True)
class LockedRefresh:
    lock: RefreshLock
    rows_in: Callable[[AdvisoryLockTransaction], CredentialRows]

    async def run(
        self,
        flight: str,
        refresh: Callable[[CredentialRows], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        key: Final = json.dumps([_COORDINATOR_NAMESPACE, flight])
        match await self.lock.while_held(
            key, lambda transaction: refresh(_SavepointRows(self.rows_in(transaction), transaction))
        ):
            case AdvisoryLockUnavailable():
                return await reread()
            case token:
                return token


@dataclass(frozen=True, slots=True)
class _TransactionClient:
    db: AdvisoryLockTransaction


class _Save(Enum):
    SAVED = "saved"
    MISSED = "missed"
    FAILED = "failed"


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _owner_condition(owner: tuple[str | None, str | None]) -> tuple[str, str] | None:
    team_id, key_hash = owner
    if team_id is not None:
        return CHATGPT_OWNER_TEAM_ID_KEY, team_id
    if key_hash is not None:
        return CHATGPT_OWNER_KEY_HASH_KEY, key_hash
    return None


def owned_chatgpt_subscription_logins(team_id: str | None, key_hash: str | None) -> tuple[CredentialItem, ...]:
    return tuple(
        credential for credential in litellm.credential_list if owns_chatgpt_login(credential, team_id, key_hash)
    )


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


def fresh_chatgpt_subscription_logins(
    owned: Iterable[CredentialItem], clock: Callable[[], float] = time.time
) -> Mapping[str, str]:
    now: Final = clock()
    return MappingProxyType(
        {
            credential.credential_name: token
            for credential in owned
            if (token := _fresh_access_token(credential.credential_values, now)) is not None
        }
    )


def _in_memory_login(credential_name: str) -> _InMemoryLogin | None:
    current: Final = CredentialAccessor.find_credential(credential_name)
    if current is None:
        return None
    return MappingProxyType(dict(current.credential_values)), MappingProxyType(dict(chatgpt_credential_info(current)))


@dataclass(frozen=True, slots=True)
class ChatGPTLoginRefresher:
    rows: CredentialRows
    oauth_client: ChatGPTOAuthClient
    coordinator: RefreshCoordinator
    shared_refresh: SharedRefresh | None = None
    clock: Callable[[], float] = time.time
    encrypt: Callable[[str], str] = encrypt_value_helper
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    save_retry_delays: tuple[float, ...] = _SAVE_RETRY_DELAYS_SECONDS

    async def access_token(self, owned: CredentialItem) -> str | None:
        owner: Final = chatgpt_login_owner(owned)
        refresh: Final = partial(self._refresh, owned.credential_name, owner)
        reread: Final = partial(self._reread, owned.credential_name, owner, self.rows)
        flight: Final = json.dumps([owned.credential_name, *owner])
        shared: Final = self.shared_refresh
        token: Final = await self.coordinator.run(
            _COORDINATOR_NAMESPACE,
            flight,
            partial(refresh, self.rows) if shared is None else partial(shared.run, flight, refresh, reread),
            reread,
        )
        return token.access_token if token is not None else None

    async def _stored(self, credential_name: str, rows: CredentialRows) -> CredentialItem | None:
        try:
            return await rows.find_by_name(credential_name)
        except Exception:  # noqa: BLE001  # a DB outage fails this login closed instead of breaking the request
            verbose_proxy_logger.warning(
                "Could not read ChatGPT subscription login %s from the database", credential_name
            )
            return None

    def _remember(
        self,
        stored: CredentialItem,
        values: Mapping[str, object],
        seen: _InMemoryLogin | None,
    ) -> None:
        if seen is None or _in_memory_login(stored.credential_name) != seen:
            return
        CredentialAccessor.upsert_credentials(
            [
                CredentialItem(
                    credential_name=stored.credential_name,
                    display_name=stored.display_name,
                    credential_values=dict(values),
                    credential_info=dict(chatgpt_credential_info(stored)),
                )
            ]
        )

    def _fresh_from(
        self, stored: CredentialItem, seen: _InMemoryLogin | None
    ) -> tuple[OAuthToken | None, Mapping[str, object]]:
        values: Final = decrypted_credential(stored).credential_values
        access_token: Final = _fresh_access_token(values, self.clock())
        if access_token is None:
            return None, values
        self._remember(stored, values, seen)
        return OAuthToken(access_token=access_token, expires_at=_expires_at(values)), values

    async def _reread(
        self, credential_name: str, owner: tuple[str | None, str | None], rows: CredentialRows
    ) -> OAuthToken | None:
        seen: Final = _in_memory_login(credential_name)
        stored: Final = await self._stored(credential_name, rows)
        if stored is None or not owns_chatgpt_login(stored, *owner):
            return None
        token, _ = self._fresh_from(stored, seen)
        return token

    async def _refresh(
        self, credential_name: str, owner: tuple[str | None, str | None], rows: CredentialRows
    ) -> OAuthToken | None:
        seen: Final = _in_memory_login(credential_name)
        stored: Final = await self._stored(credential_name, rows)
        owner_condition: Final = _owner_condition(owner)
        if stored is None or owner_condition is None or not owns_chatgpt_login(stored, *owner):
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
        encrypted: Final = {key: self.encrypt(value) for key, value in login_values.items()}
        saved_values: Final = {"credential_values": json.dumps({**stored.credential_values, **encrypted})}
        redeemed: Final = (_REFRESH_TOKEN_KEY, stored_refresh_token)
        refreshed_values: Final = {**values, **login_values}
        match await self._save(rows, credential_name, redeemed, owner_condition, saved_values):
            case _Save.SAVED:
                self._remember(stored, refreshed_values, seen)
            case _Save.FAILED:
                _log_unsaved_refresh(credential_name)
                self._remember(stored, refreshed_values, seen)
            case _Save.MISSED:
                return await self._hand_over_to_new_owner(rows, credential_name, owner, redeemed, saved_values)
        return OAuthToken(access_token=tokens.access_token, expires_at=_expires_at(refreshed_values))

    async def _hand_over_to_new_owner(
        self,
        rows: CredentialRows,
        credential_name: str,
        owner: tuple[str | None, str | None],
        redeemed: tuple[str, str],
        saved_values: Mapping[str, object],
    ) -> OAuthToken | None:
        match await self._save(rows, credential_name, redeemed, None, saved_values):
            case _Save.SAVED:
                return None
            case _Save.FAILED:
                _log_unsaved_refresh(credential_name)
                return None
            case _Save.MISSED:
                return await self._reread(credential_name, owner, rows)

    async def _save(
        self,
        rows: CredentialRows,
        credential_name: str,
        redeemed: tuple[str, str],
        owner_condition: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> _Save:
        for delay in self.save_retry_delays:
            if (
                outcome := await self._try_save(rows, credential_name, redeemed, owner_condition, data)
            ) is not _Save.FAILED:
                return outcome
            await self.sleep(delay)
        return await self._try_save(rows, credential_name, redeemed, owner_condition, data)

    async def _try_save(
        self,
        rows: CredentialRows,
        credential_name: str,
        redeemed: tuple[str, str],
        owner_condition: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> _Save:
        try:
            saved: Final = await rows.update_if_value_unchanged(credential_name, redeemed, owner_condition, data)
        except Exception:  # noqa: BLE001  # a failed save is retried, then the refreshed tokens serve from memory
            verbose_proxy_logger.warning("Saving refreshed ChatGPT subscription login %s failed", credential_name)
            return _Save.FAILED
        return _Save.SAVED if saved else _Save.MISSED


def _log_unsaved_refresh(credential_name: str) -> None:
    verbose_proxy_logger.error(
        "ChatGPT subscription login %s was refreshed but could not be saved to the database. "
        "Sign in again for it if requests start failing after the new token expires.",
        credential_name,
    )


def runtime_chatgpt_login_refresher() -> ChatGPTLoginRefresher | None:
    from litellm.proxy.db.advisory_lock import PrismaAdvisoryLock
    from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient, writer_wrapper
    from litellm.proxy.mcp_server.outbound_credentials.runtime_refresh_coordinator import (
        runtime_refresh_coordinator,
    )
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.credentials_repository import CredentialsRepository

    if prisma_client is None:
        return None
    rows: Final = CredentialsRepository(WriterPinnedClient(prisma_client.db))
    redis_coordinator: Final = runtime_refresh_coordinator()
    return ChatGPTLoginRefresher(
        rows=rows,
        oauth_client=new_chatgpt_oauth_client(),
        coordinator=_IN_PROCESS_COORDINATOR,
        shared_refresh=(
            CoordinatedRefresh(redis_coordinator, rows)
            if redis_coordinator is not None
            else LockedRefresh(
                PrismaAdvisoryLock(writer_wrapper(prisma_client.db), timeout=_REFRESH_LOCK_TIMEOUT),
                rows_in=lambda transaction: CredentialsRepository(_TransactionClient(transaction)),
            )
        ),
    )


async def owned_chatgpt_subscription_access_token(
    credential_name: str,
    team_id: str | None,
    key_hash: str | None,
    refresher_factory: Callable[[], ChatGPTLoginRefresher | None] = runtime_chatgpt_login_refresher,
    clock: Callable[[], float] = time.time,
) -> str | None:
    credential: Final = CredentialAccessor.find_credential(credential_name)
    if credential is None or not owns_chatgpt_login(credential, team_id, key_hash):
        return None
    fresh: Final = _fresh_access_token(credential.credential_values, clock())
    if fresh is not None:
        return fresh
    try:
        refresher: Final = refresher_factory()
        return await refresher.access_token(credential) if refresher is not None else None
    except Exception:  # noqa: BLE001  # a refresh failure fails only this login closed, never the request
        verbose_proxy_logger.exception("Failed to refresh ChatGPT subscription login %s", credential_name)
        return None


class CredentialWrites(Protocol):
    async def find_by_name(self, credential_name: str) -> CredentialItem | None: ...

    async def create(self, data: Mapping[str, object]) -> object: ...

    async def update_if_unchanged(
        self,
        credential_name: str,
        expected_values: Mapping[str, object],
        expected_info: Mapping[str, object],
        data: Mapping[str, object],
    ) -> bool: ...


_OWNER_KEYS: Final = frozenset({CHATGPT_OWNER_TEAM_ID_KEY, CHATGPT_OWNER_KEY_HASH_KEY})
_SIGN_IN_SAVE_ATTEMPTS: Final = 3


@dataclass(frozen=True, slots=True)
class _RowChanged:
    pass


def blocks_chatgpt_login(existing: CredentialItem | None) -> bool:
    if existing is None or not existing.credential_values or is_chatgpt_subscription_login(existing):
        return False
    return stored_credential_provider(chatgpt_credential_info(existing).get("custom_llm_provider")) != CHATGPT_PROVIDER


async def save_chatgpt_login(
    rows: CredentialWrites,
    credential_name: str,
    display_name: str | None,
    login_values: Mapping[str, str],
    owner: tuple[str | None, str | None],
    actor: str,
    encrypt: Callable[[str], str] = encrypt_value_helper,
) -> CredentialItem | None:
    attempt: Final = partial(
        _save_chatgpt_login_once, rows, credential_name, display_name, login_values, owner, actor, encrypt
    )
    for _ in range(_SIGN_IN_SAVE_ATTEMPTS):
        match await attempt():
            case _RowChanged():
                continue
            case saved:
                return saved
    return None


async def _save_chatgpt_login_once(
    rows: CredentialWrites,
    credential_name: str,
    display_name: str | None,
    login_values: Mapping[str, str],
    owner: tuple[str | None, str | None],
    actor: str,
    encrypt: Callable[[str], str],
) -> CredentialItem | None | _RowChanged:
    existing: Final = await rows.find_by_name(credential_name)
    if blocks_chatgpt_login(existing):
        return None
    owner_team_id, owner_key_hash = owner
    kept_info: Final = {
        key: value for key, value in chatgpt_credential_info(existing).items() if key not in _OWNER_KEYS
    }
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
    elif not await rows.update_if_unchanged(
        credential_name, existing.credential_values, chatgpt_credential_info(existing), stored_fields
    ):
        return _RowChanged()
    saved: Final = CredentialItem(
        credential_name=credential_name,
        display_name=saved_display_name,
        credential_values=dict(login_values),
        credential_info=credential_info,
    )
    CredentialAccessor.upsert_credentials([saved])
    return saved


class CredentialInfoLookup(Protocol):
    async def has_info_value(self, info_key: str, value: str) -> bool: ...


def primary_credential_rows(prisma_client: "PrismaClient | None") -> CredentialInfoLookup | None:
    from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient
    from litellm.repositories.credentials_repository import CredentialsRepository

    return None if prisma_client is None else CredentialsRepository(WriterPinnedClient(prisma_client.db))


async def owner_has_chatgpt_login(rows: CredentialInfoLookup | None, team_id: str | None, key_hash: str | None) -> bool:
    if owned_chatgpt_subscription_logins(team_id, key_hash):
        return True
    condition: Final = _owner_condition((team_id, key_hash))
    return rows is not None and condition is not None and await rows.has_info_value(*condition)


class CredentialRepoints(Protocol):
    async def repoint_info_value(self, info_key: str, previous_value: str, new_value: str) -> int: ...


async def repoint_chatgpt_logins_to_key(rows: CredentialRepoints, previous_key_hash: str, new_key_hash: str) -> None:
    await rows.repoint_info_value(CHATGPT_OWNER_KEY_HASH_KEY, previous_key_hash, new_key_hash)
    CredentialAccessor.upsert_credentials(
        [
            credential.model_copy(
                update={
                    "credential_info": {**chatgpt_credential_info(credential), CHATGPT_OWNER_KEY_HASH_KEY: new_key_hash}
                }
            )
            for credential in litellm.credential_list
            if chatgpt_credential_info(credential).get(CHATGPT_OWNER_KEY_HASH_KEY) == previous_key_hash
        ]
    )

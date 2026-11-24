import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

import httpx
import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.authenticator import TOKEN_EXPIRY_SKEW_SECONDS
from litellm.llms.chatgpt.common_utils import CHATGPT_OAUTH_TOKEN_URL, OPENAI_AUTH_CLAIM_KEY
from litellm.llms.chatgpt.subscription_login import fresh_chatgpt_access_token
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.chatgpt_subscription_logins import (
    ChatGPTLoginRefresher,
    CoordinatedRefresh,
    LockedRefresh,
    RefusedRefreshTokens,
    blocks_chatgpt_login,
    owned_chatgpt_subscription_access_token,
    owned_chatgpt_subscription_logins,
    save_chatgpt_login,
)
from litellm.proxy.db.advisory_lock import AdvisoryLockTransaction, AdvisoryLockUnavailable, PrismaAdvisoryLock
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import InProcessRefreshCoordinator, OAuthToken
from litellm.types.utils import CredentialItem

NOW: Final = 1_900_000_000.0
FRESH_EXPIRES_AT: Final = str(int(NOW) + 3600)
STALE_EXPIRES_AT: Final = str(int(NOW) + TOKEN_EXPIRY_SKEW_SECONDS)
REFRESHED_EXPIRES_AT: Final = int(NOW) + 7200


def _now() -> float:
    return NOW


def _jwt(claims: Mapping[str, object]) -> str:
    def segment(value: Mapping[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'RS256', 'typ': 'JWT'})}.{segment(claims)}.c2lnbmF0dXJl"


def _access_token(tag: str, exp: int = REFRESHED_EXPIRES_AT, account_id: str = "acct-1") -> str:
    return _jwt({"exp": exp, "jti": tag, OPENAI_AUTH_CLAIM_KEY: {"chatgpt_account_id": account_id}})


def _id_token(account_id: str = "acct-1") -> str:
    return _jwt({"email": "owner@example.com", OPENAI_AUTH_CLAIM_KEY: {"chatgpt_account_id": account_id}})


def _login_values(
    access_token: str, refresh_token: str, expires_at: str | None, account_id: str = "acct-1"
) -> dict[str, str]:
    return {
        "chatgpt_access_token": access_token,
        "chatgpt_refresh_token": refresh_token,
        "chatgpt_id_token": _id_token(account_id),
        "chatgpt_account_id": account_id,
        **({"chatgpt_expires_at": expires_at} if expires_at is not None else {}),
    }


def _login(
    name: str,
    values: Mapping[str, str],
    team_id: str | None = None,
    key_hash: str | None = None,
    **extra_info: str,
) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values=dict(values),
        credential_info={
            "custom_llm_provider": "chatgpt",
            **({"chatgpt_owner_team_id": team_id} if team_id is not None else {}),
            **({"chatgpt_owner_key_hash": key_hash} if key_hash is not None else {}),
            **extra_info,
        },
    )


def _stored(credential: CredentialItem) -> CredentialItem:
    return credential.model_copy(
        update={
            "credential_values": {
                key: encrypt_value_helper(str(value)) for key, value in credential.credential_values.items()
            }
        }
    )


def _decrypted(values: Mapping[str, object]) -> dict[str, str | None]:
    return {key: decrypt_value_helper(str(value), key) for key, value in values.items()}


def _row_after_write(row: CredentialItem, data: Mapping[str, object]) -> CredentialItem:
    return row.model_copy(
        update={
            field: json.loads(value) if field in ("credential_values", "credential_info") else value
            for field, value in data.items()
            if field in ("credential_values", "credential_info", "display_name") and isinstance(value, str)
        }
    )


class _CredentialRows:
    def __init__(
        self,
        *rows: CredentialItem,
        failing_writes: int = 0,
        reads_fail: bool = False,
        rewritten_after_reads: Sequence[CredentialItem] = (),
    ) -> None:
        self.rows: dict[str, CredentialItem] = {row.credential_name: row for row in rows}
        self.rewritten_after_reads = list(rewritten_after_reads)
        self.updates: list[tuple[str, Mapping[str, object]]] = []
        self.compared: list[tuple[tuple[str, str], tuple[str, str] | None]] = []
        self.reads: list[str] = []
        self.creates: list[Mapping[str, object]] = []
        self.failing_writes = failing_writes
        self.reads_fail = reads_fail

    def _attempt_write(self) -> None:
        if self.failing_writes > 0:
            self.failing_writes -= 1
            raise ConnectionError("the database is unreachable")

    def put(self, row: CredentialItem) -> None:
        self.rows[row.credential_name] = row

    def remove(self, credential_name: str) -> None:
        self.rows.pop(credential_name)

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        self.reads.append(credential_name)
        if self.reads_fail:
            raise ConnectionError("the database is unreachable")
        found: Final = self.rows.get(credential_name)
        if self.rewritten_after_reads:
            self.put(self.rewritten_after_reads.pop(0))
        return found

    async def update_if_value_unchanged(
        self,
        credential_name: str,
        value_match: tuple[str, str],
        info_match: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> bool:
        self.updates.append((credential_name, data))
        self.compared.append((value_match, info_match))
        self._attempt_write()
        current: Final = self.rows.get(credential_name)
        value_key, stored_value = value_match
        if (
            current is None
            or current.credential_values.get(value_key) != stored_value
            or (info_match is not None and current.credential_info.get(info_match[0]) != info_match[1])
        ):
            return False
        self.rows[credential_name] = _row_after_write(current, data)
        return True

    async def update_if_unchanged(
        self,
        credential_name: str,
        expected_values: Mapping[str, object],
        expected_info: Mapping[str, object],
        data: Mapping[str, object],
    ) -> bool:
        self.updates.append((credential_name, data))
        self._attempt_write()
        current: Final = self.rows.get(credential_name)
        if (
            current is None
            or current.credential_values != expected_values
            or (expected_info and current.credential_info != expected_info)
        ):
            return False
        self.rows[credential_name] = _row_after_write(current, data)
        return True

    async def create(self, data: Mapping[str, object]) -> None:
        self.creates.append(data)
        name: Final = str(data["credential_name"])
        self.rows[name] = _row_after_write(
            CredentialItem(credential_name=name, credential_values={}, credential_info={}), data
        )


class _TokenEndpoint:
    def __init__(
        self,
        status: int = 200,
        body: Mapping[str, object] | None = None,
        during_post: Callable[[], None] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.during_post = during_post
        self.posts: list[tuple[str, Mapping[str, str] | None]] = []

    async def post(
        self,
        url: str,
        data: Mapping[str, str] | str | bytes | None = None,
        json: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        self.posts.append((url, json))
        await asyncio.sleep(0)
        if self.during_post is not None:
            self.during_post()
        request: Final = httpx.Request("POST", url)
        response: Final = httpx.Response(self.status, json=self.body, request=request)
        if response.is_error:
            raise httpx.HTTPStatusError(
                f"{self.status} from {url}: {response.text}", request=request, response=response
            )
        return response


def _refreshed(
    tag: str, refresh_token: str | None = "rt-issued-by-refresh", during_post: Callable[[], None] | None = None
) -> tuple[_TokenEndpoint, str]:
    access_token: Final = _access_token(tag)
    return (
        _TokenEndpoint(
            body={
                "access_token": access_token,
                "id_token": _id_token(),
                **({"refresh_token": refresh_token} if refresh_token is not None else {}),
            },
            during_post=during_post,
        ),
        access_token,
    )


class _LosingCoordinator:
    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        return await reread()


class _SharedCoordinatorWithoutLock:
    def __init__(self) -> None:
        self.runs = 0

    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[OAuthToken | None]],
        reread: Callable[[], Awaitable[OAuthToken | None]],
    ) -> OAuthToken | None:
        self.runs += 1
        return await refresh()


class _LockTransaction:
    def __init__(self) -> None:
        self.aborted = False

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return [{"locked": 1}]

    async def execute_raw(self, query: str, *args: object) -> int:
        if query.startswith("ROLLBACK TO SAVEPOINT"):
            self.aborted = False
        elif self.aborted:
            raise RuntimeError("current transaction is aborted, commands ignored until end of transaction block")
        return 0


class _RowsOnTransaction:
    def __init__(self, rows: _CredentialRows, transaction: _LockTransaction) -> None:
        self.rows = rows
        self.transaction = transaction

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        return await self.rows.find_by_name(credential_name)

    async def update_if_value_unchanged(
        self,
        credential_name: str,
        value_match: tuple[str, str],
        info_match: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> bool:
        if self.transaction.aborted:
            raise RuntimeError("current transaction is aborted, commands ignored until end of transaction block")
        try:
            return await self.rows.update_if_value_unchanged(credential_name, value_match, info_match, data)
        except ConnectionError:
            self.transaction.aborted = True
            raise


class _DatabaseLock:
    def __init__(self) -> None:
        self.held: dict[str, asyncio.Lock] = {}
        self.transaction = _LockTransaction()

    async def while_held(
        self, key: str, action: Callable[[AdvisoryLockTransaction], Awaitable[OAuthToken | None]]
    ) -> OAuthToken | None | AdvisoryLockUnavailable:
        async with self.held.setdefault(key, asyncio.Lock()):
            return await action(self.transaction)


class _DatabaseThatCannotCommit:
    def __init__(self) -> None:
        self.transaction = _LockTransaction()

    @asynccontextmanager
    async def tx(self, *, timeout: timedelta) -> AsyncIterator[_LockTransaction]:
        yield self.transaction
        raise ConnectionError("the commit did not reach the database")


class _UnreachableDatabaseLock:
    async def while_held(
        self, key: str, action: Callable[[AdvisoryLockTransaction], Awaitable[OAuthToken | None]]
    ) -> OAuthToken | None | AdvisoryLockUnavailable:
        return AdvisoryLockUnavailable()


def _rows_in_lock_transaction(
    lock: _DatabaseLock, rows: _CredentialRows
) -> Callable[[AdvisoryLockTransaction], _CredentialRows]:
    def rows_in(transaction: AdvisoryLockTransaction) -> _CredentialRows:
        assert transaction is lock.transaction
        return rows

    return rows_in


def _recording_sleep(waits: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        waits.append(delay)

    return sleep


def _refresher(
    rows: _CredentialRows,
    oauth_client: _TokenEndpoint,
    coordinator: InProcessRefreshCoordinator | _LosingCoordinator | None = None,
    shared_refresh: CoordinatedRefresh | LockedRefresh | None = None,
    waits: list[float] | None = None,
    refused: RefusedRefreshTokens | None = None,
) -> ChatGPTLoginRefresher:
    return ChatGPTLoginRefresher(
        rows=rows,
        oauth_client=oauth_client,
        coordinator=coordinator or InProcessRefreshCoordinator(),
        shared_refresh=shared_refresh,
        clock=_now,
        sleep=_recording_sleep(waits if waits is not None else []),
        refused=refused or RefusedRefreshTokens(),
    )


async def _resolve(
    team_id: str | None, key_hash: str | None, refresher: ChatGPTLoginRefresher | None = None
) -> Mapping[str, str]:
    names: Final = tuple(credential.credential_name for credential in litellm.credential_list)
    tokens: Final = await asyncio.gather(
        *(
            owned_chatgpt_subscription_access_token(
                name, team_id, key_hash, refresher_factory=lambda: refresher, clock=_now
            )
            for name in names
        )
    )
    return {name: token for name, token in zip(names, tokens, strict=True) if token is not None}


def _served(credential_name: str) -> Mapping[str, object]:
    credential: Final = CredentialAccessor.find_credential(credential_name)
    assert credential is not None
    return credential.credential_values


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "chatgpt-subscription-logins-test-salt")
    monkeypatch.setattr(litellm, "credential_list", [])


def _serve(monkeypatch, *credentials: CredentialItem) -> None:
    monkeypatch.setattr(litellm, "credential_list", list(credentials))


@pytest.fixture
def proxy_log(caplog):
    caplog.set_level(logging.DEBUG, logger=verbose_proxy_logger.name)
    verbose_proxy_logger.addHandler(caplog.handler)
    yield caplog
    verbose_proxy_logger.removeHandler(caplog.handler)


def _logged_text(caplog: pytest.LogCaptureFixture) -> str:
    return caplog.text + "\n".join(record.getMessage() for record in caplog.records)


async def test_a_fresh_login_is_served_from_memory_without_the_database_or_the_token_endpoint(monkeypatch):
    access_token: Final = _access_token("fresh")
    _serve(monkeypatch, _login("login-a", _login_values(access_token, "rt-a", FRESH_EXPIRES_AT), team_id="team-a"))
    rows: Final = _CredentialRows()
    oauth: Final = _TokenEndpoint(status=500)

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": access_token}
    assert oauth.posts == []
    assert (rows.updates, rows.creates) == ([], [])


async def test_a_stale_login_another_replica_already_refreshed_is_served_from_the_row(monkeypatch):
    memory_values: Final = _login_values(_access_token("old", exp=int(NOW)), "rt-old", STALE_EXPIRES_AT)
    row_values: Final = _login_values(_access_token("refreshed-elsewhere"), "rt-current", FRESH_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", memory_values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, team_id="team-a")))
    oauth: Final = _TokenEndpoint(status=500)

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": row_values["chatgpt_access_token"]}
    assert oauth.posts == []
    assert rows.updates == []
    assert _served("login-a") == row_values


async def test_a_stale_login_redeems_the_rows_refresh_token_and_saves_the_result_encrypted(monkeypatch):
    memory_values: Final = _login_values(_access_token("old"), "rt-memory-older", STALE_EXPIRES_AT)
    row_values: Final = _login_values(_access_token("old-row"), "rt-row-current", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", memory_values, team_id="team-a"))
    stored_row: Final = _stored(
        _login("login-a", {**row_values, "api_base": "https://chatgpt.com/backend-api/codex"}, team_id="team-a")
    )
    rows: Final = _CredentialRows(stored_row)
    oauth, new_access_token = _refreshed("new", refresh_token="rt-rotated")

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": new_access_token}
    assert [(url, (body or {}).get("refresh_token")) for url, body in oauth.posts] == [
        (CHATGPT_OAUTH_TOKEN_URL, "rt-row-current")
    ]
    assert [name for name, _ in rows.updates] == ["login-a"]
    written: Final = rows.updates[0][1]["credential_values"]
    assert isinstance(written, str)
    saved: Final = json.loads(written)
    expected_login: Final = {
        "chatgpt_access_token": new_access_token,
        "chatgpt_refresh_token": "rt-rotated",
        "chatgpt_id_token": _id_token(),
        "chatgpt_account_id": "acct-1",
        "chatgpt_expires_at": str(REFRESHED_EXPIRES_AT),
    }
    for key, plaintext in expected_login.items():
        assert saved[key] != plaintext
        assert decrypt_value_helper(saved[key], key) == plaintext
    for secret in (new_access_token, "rt-rotated", _id_token()):
        assert secret not in written
    assert saved["api_base"] == stored_row.credential_values["api_base"]
    assert _served("login-a")["chatgpt_access_token"] == new_access_token
    assert _served("login-a")["chatgpt_refresh_token"] == "rt-rotated"


async def test_a_refresh_response_without_a_refresh_token_keeps_the_old_one(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-still-valid", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    oauth, new_access_token = _refreshed("new", refresh_token=None)

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": new_access_token}
    assert _decrypted(rows.rows["login-a"].credential_values)["chatgpt_refresh_token"] == "rt-still-valid"
    assert _served("login-a")["chatgpt_refresh_token"] == "rt-still-valid"


@pytest.mark.parametrize("expires_at", ["nan", "inf", "1e400", "soon"])
async def test_a_login_whose_expiry_is_not_a_finite_time_is_refreshed(monkeypatch, expires_at):
    values: Final = _login_values(_access_token("old"), "rt-a", expires_at)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    oauth, new_access_token = _refreshed("new")

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": new_access_token}
    assert len(oauth.posts) == 1


async def test_concurrent_requests_for_one_stale_login_redeem_it_once(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    oauth, new_access_token = _refreshed("new")
    refresher: Final = _refresher(rows, oauth)

    resolved: Final = await asyncio.gather(*(_resolve("team-a", None, refresher) for _ in range(8)))

    assert resolved == [{"login-a": new_access_token}] * 8
    assert len(oauth.posts) == 1
    assert len(rows.updates) == 1


async def test_a_refresh_lock_loser_serves_the_token_the_winner_saved(monkeypatch):
    memory_values: Final = _login_values(_access_token("before-winner"), "rt-before-winner", STALE_EXPIRES_AT)
    row_values: Final = _login_values(_access_token("winner"), "rt-winner", FRESH_EXPIRES_AT)
    held: Final = _login("login-a", memory_values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, team_id="team-a")))
    oauth: Final = _TokenEndpoint(status=500)

    token: Final = await _refresher(rows, oauth, _LosingCoordinator()).access_token(held)

    assert token == row_values["chatgpt_access_token"]
    assert oauth.posts == []
    assert _served("login-a") == row_values


async def test_a_refresh_lock_loser_serves_nothing_while_the_row_is_still_stale():
    row_values: Final = _login_values(_access_token("stale"), "rt-stale", STALE_EXPIRES_AT)
    held: Final = _login("login-a", row_values, team_id="team-a")
    rows: Final = _CredentialRows(_stored(held))
    oauth: Final = _TokenEndpoint(status=500)

    token: Final = await _refresher(rows, oauth, _LosingCoordinator()).access_token(held)

    assert token is None
    assert oauth.posts == []


async def test_a_failed_redeem_serves_no_login_and_logs_no_secret(monkeypatch, proxy_log):
    stale_access_token: Final = _access_token("stale-access")
    other_access_token: Final = _access_token("other")
    memory_values: Final = _login_values(stale_access_token, "rt-memory-copy", STALE_EXPIRES_AT)
    row_values: Final = _login_values(stale_access_token, "rt-row-revoked", STALE_EXPIRES_AT)
    _serve(
        monkeypatch,
        _login("login-a", memory_values, team_id="team-a"),
        _login("login-b", _login_values(other_access_token, "rt-b", FRESH_EXPIRES_AT), team_id="team-a"),
    )
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, team_id="team-a")))
    leaked_in_body: Final = _jwt({"sub": "token-echoed-in-error-body"})
    oauth: Final = _TokenEndpoint(
        status=400,
        body={"error": "invalid_grant", "error_description": f"refresh token already used: {leaked_in_body}"},
    )

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-b": other_access_token}
    assert len(oauth.posts) == 1
    assert rows.updates == []
    logged: Final = _logged_text(proxy_log)
    assert "login-a" in logged
    for secret in (stale_access_token, "rt-memory-copy", "rt-row-revoked", leaked_in_body, "already used"):
        assert secret not in logged
        assert secret not in repr(resolved)


@pytest.mark.parametrize(
    ("status", "posts_after_two_requests"),
    [(400, 1), (401, 1), (429, 2), (503, 2)],
    ids=["refused-400", "refused-401", "rate-limited", "unavailable"],
)
async def test_a_refused_refresh_token_is_not_sent_again(monkeypatch, status, posts_after_two_requests):
    values: Final = _login_values(_access_token("stale"), "rt-revoked", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    oauth: Final = _TokenEndpoint(status=status, body={"error": "invalid_grant"})
    refused: Final = RefusedRefreshTokens()

    first: Final = await _resolve("team-a", None, _refresher(rows, oauth, refused=refused))
    second: Final = await _resolve("team-a", None, _refresher(rows, oauth, refused=refused))

    assert first == second == {}
    assert len(oauth.posts) == posts_after_two_requests


async def test_signing_in_again_after_a_refused_refresh_serves_the_new_login(monkeypatch):
    values: Final = _login_values(_access_token("stale"), "rt-revoked", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    refused: Final = RefusedRefreshTokens()
    await _resolve("team-a", None, _refresher(rows, _TokenEndpoint(status=400), refused=refused))
    signed_in_again: Final = _login_values(_access_token("stale-again"), "rt-new-sign-in", STALE_EXPIRES_AT)
    rows.put(_stored(_login("login-a", signed_in_again, team_id="team-a")))
    oauth, new_access_token = _refreshed("new")

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth, refused=refused))

    assert resolved == {"login-a": new_access_token}
    assert len(oauth.posts) == 1


async def test_a_save_that_keeps_failing_is_retried_then_serves_the_refreshed_token(monkeypatch, proxy_log):
    values: Final = _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")), failing_writes=10)
    oauth, new_access_token = _refreshed("new", refresh_token="rt-rotated")
    waits: Final[list[float]] = []

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth, waits=waits))

    assert resolved == {"login-a": new_access_token}
    assert len(oauth.posts) == 1
    assert len(rows.updates) == 3
    assert len(waits) == 2
    assert all(delay > 0 for delay in waits)
    assert _served("login-a")["chatgpt_access_token"] == new_access_token
    unsaved_levels: Final = {
        record.levelno for record in proxy_log.records if "could not be saved" in record.getMessage()
    }
    assert unsaved_levels == {logging.ERROR}
    logged: Final = _logged_text(proxy_log)
    assert "login-a" in logged
    for secret in (new_access_token, "rt-rotated", "rt-a"):
        assert secret not in logged


async def test_a_save_that_fails_once_is_retried_and_lands_in_the_row(monkeypatch, proxy_log):
    values: Final = _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")), failing_writes=1)
    oauth, new_access_token = _refreshed("new", refresh_token="rt-rotated")
    waits: Final[list[float]] = []

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth, waits=waits))

    assert resolved == {"login-a": new_access_token}
    assert len(oauth.posts) == 1
    assert len(rows.updates) == 2
    assert len(waits) == 1
    saved: Final = _decrypted(rows.rows["login-a"].credential_values)
    assert (saved["chatgpt_access_token"], saved["chatgpt_refresh_token"]) == (new_access_token, "rt-rotated")
    assert [record for record in proxy_log.records if record.levelno >= logging.ERROR] == []


async def test_concurrent_calls_redeem_once_when_the_shared_lock_refreshes_without_locking(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    oauth, new_access_token = _refreshed("new")
    shared: Final = _SharedCoordinatorWithoutLock()
    database_lock: Final = _DatabaseLock()
    locked: Final = LockedRefresh(database_lock, rows_in=_rows_in_lock_transaction(database_lock, rows))
    refresher: Final = _refresher(rows, oauth, shared_refresh=CoordinatedRefresh(shared, locked))

    tokens: Final = await asyncio.gather(*(refresher.access_token(held) for _ in range(5)))

    assert tokens == [new_access_token] * 5
    assert len(oauth.posts) == 1
    assert shared.runs == 1


async def test_two_replicas_whose_shared_lock_lets_both_refresh_redeem_a_stale_login_once_through_the_database_lock(
    monkeypatch,
):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    oauth, new_access_token = _refreshed("new")
    shared: Final = _SharedCoordinatorWithoutLock()
    database_lock: Final = _DatabaseLock()
    locked: Final = LockedRefresh(database_lock, rows_in=_rows_in_lock_transaction(database_lock, rows))
    replicas: Final = tuple(
        _refresher(rows, oauth, shared_refresh=CoordinatedRefresh(shared, locked)) for _ in range(2)
    )

    tokens: Final = await asyncio.gather(*(replica.access_token(held) for replica in replicas))

    assert tokens == [new_access_token, new_access_token]
    assert shared.runs == 2
    assert len(oauth.posts) == 1


async def test_a_refresh_that_loses_its_write_to_a_new_sign_in_serves_the_new_sign_in(monkeypatch):
    old_values: Final = _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT)
    signed_in_values: Final = _login_values(_access_token("signed-in-again"), "rt-signed-in-again", FRESH_EXPIRES_AT)
    held: Final = _login("login-a", old_values, team_id="team-a")
    _serve(monkeypatch, held)
    stored_row: Final = _stored(held)
    rows: Final = _CredentialRows(stored_row)
    oauth, redeemed_access_token = _refreshed(
        "redeemed",
        refresh_token="rt-redeemed",
        during_post=lambda: rows.put(_stored(_login("login-a", signed_in_values, team_id="team-a"))),
    )

    token: Final = await _refresher(rows, oauth).access_token(held)

    assert token == signed_in_values["chatgpt_access_token"]
    redeemed: Final = ("chatgpt_refresh_token", stored_row.credential_values["chatgpt_refresh_token"])
    assert rows.compared == [(redeemed, ("chatgpt_owner_team_id", "team-a")), (redeemed, None)]
    assert _decrypted(rows.rows["login-a"].credential_values) == signed_in_values
    assert _served("login-a") == signed_in_values
    assert redeemed_access_token != token


async def test_a_refresh_that_loses_its_write_to_another_owners_sign_in_serves_nothing(monkeypatch):
    old_values: Final = _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT)
    team_b_values: Final = _login_values(_access_token("team-b-seat"), "rt-team-b", FRESH_EXPIRES_AT)
    held: Final = _login("login-a", old_values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    oauth, _ = _refreshed(
        "redeemed",
        refresh_token="rt-redeemed",
        during_post=lambda: rows.put(_stored(_login("login-a", team_b_values, team_id="team-b"))),
    )

    token: Final = await _refresher(rows, oauth).access_token(held)

    assert token is None
    assert _decrypted(rows.rows["login-a"].credential_values) == team_b_values
    assert _served("login-a") == old_values


@pytest.mark.parametrize("row_deleted", [False, True], ids=["memory-only", "memory-and-row"])
async def test_a_login_deleted_while_its_refresh_is_in_flight_stays_deleted(monkeypatch, row_deleted):
    values: Final = _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))

    def delete() -> None:
        _serve(monkeypatch)
        if row_deleted:
            rows.remove("login-a")

    oauth, _ = _refreshed("new", during_post=delete)

    await _refresher(rows, oauth).access_token(held)

    assert len(oauth.posts) == 1
    assert CredentialAccessor.find_credential("login-a") is None


async def test_a_login_deleted_before_its_refresh_starts_is_not_added_back(monkeypatch):
    memory_values: Final = _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT)
    row_values: Final = _login_values(_access_token("row"), "rt-row", FRESH_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", memory_values, team_id="team-a"))
    refresher: Final = _refresher(
        _CredentialRows(_stored(_login("login-a", row_values, team_id="team-a"))), _TokenEndpoint(status=500)
    )

    def delete_then_refresh() -> ChatGPTLoginRefresher:
        _serve(monkeypatch)
        return refresher

    await owned_chatgpt_subscription_access_token(
        "login-a", "team-a", None, refresher_factory=delete_then_refresh, clock=_now
    )

    assert CredentialAccessor.find_credential("login-a") is None


async def test_a_database_outage_serves_no_stale_login_and_keeps_the_others(monkeypatch, proxy_log):
    other_access_token: Final = _access_token("other")
    stale_values: Final = _login_values(_access_token("stale"), "rt-stale", STALE_EXPIRES_AT)
    _serve(
        monkeypatch,
        _login("login-a", stale_values, team_id="team-a"),
        _login("login-b", _login_values(other_access_token, "rt-b", FRESH_EXPIRES_AT), team_id="team-a"),
    )
    rows: Final = _CredentialRows(_stored(_login("login-a", stale_values, team_id="team-a")), reads_fail=True)
    oauth: Final = _TokenEndpoint(status=500)

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-b": other_access_token}
    assert oauth.posts == []
    logged: Final = _logged_text(proxy_log)
    assert "login-a" in logged
    assert "rt-stale" not in logged


@pytest.mark.parametrize("owner_key", ["chatgpt_owner_team_id", "chatgpt_owner_key_hash"], ids=["team", "key"])
@pytest.mark.parametrize("real_owner_first", [True, False], ids=["real-owner-first", "stale-owner-first"])
async def test_callers_holding_different_owners_never_share_one_refresh(owner_key, real_owner_first):
    row_values: Final = _login_values(_access_token("owner-b-seat"), "rt-owner-b", FRESH_EXPIRES_AT)
    stale_values: Final = _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT)
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, **{owner_key: "owner-b"})))
    refresher: Final = _refresher(rows, _TokenEndpoint(status=500))
    held_by_owner_b: Final = _login("login-a", stale_values, **{owner_key: "owner-b"})
    held_by_owner_a: Final = _login("login-a", stale_values, **{owner_key: "owner-a"})
    callers: Final = (held_by_owner_b, held_by_owner_a) if real_owner_first else (held_by_owner_a, held_by_owner_b)

    tokens: Final = await asyncio.gather(*(refresher.access_token(caller) for caller in callers))

    assert {caller.credential_info[owner_key]: token for caller, token in zip(callers, tokens)} == {
        "owner-b": row_values["chatgpt_access_token"],
        "owner-a": None,
    }


def _ownership_fixture() -> tuple[CredentialItem, ...]:
    def fresh(tag: str) -> dict[str, str]:
        return _login_values(_access_token(tag), f"rt-{tag}", FRESH_EXPIRES_AT)

    return (
        _login("login-team-a", fresh("team-a"), team_id="team-a"),
        _login("login-team-b", fresh("team-b"), team_id="team-b"),
        _login("login-key-k", fresh("key-k"), key_hash="hash-k"),
        _login("login-both-owners", fresh("both"), team_id="team-a", key_hash="hash-k"),
        _login("login-no-owner", fresh("none")),
        CredentialItem(
            credential_name="openai-team-a",
            credential_values={"api_key": "sk-not-a-subscription"},
            credential_info={"custom_llm_provider": "openai", "chatgpt_owner_team_id": "team-a"},
        ),
    )


@pytest.mark.parametrize(
    ("team_id", "key_hash", "expected"),
    [
        ("team-a", "hash-unbound", {"login-team-a"}),
        (None, "hash-k", {"login-key-k"}),
        ("team-b", "hash-k", {"login-team-b", "login-key-k"}),
        ("team-a", "hash-k", {"login-team-a", "login-key-k"}),
        (None, None, set()),
        ("team-unbound", None, set()),
    ],
    ids=["team-a", "key-k", "team-b-and-key-k", "team-a-and-key-k", "no-owner", "unbound-team"],
)
async def test_a_caller_gets_only_the_logins_its_team_or_key_owns(monkeypatch, team_id, key_hash, expected):
    credentials: Final = _ownership_fixture()
    _serve(monkeypatch, *credentials)
    tokens: Final = {
        credential.credential_name: credential.credential_values.get("chatgpt_access_token")
        for credential in credentials
    }

    rows: Final = _CredentialRows()
    oauth: Final = _TokenEndpoint(status=500)

    resolved: Final = await _resolve(team_id, key_hash, _refresher(rows, oauth))
    bound: Final = {
        credential.credential_name: token
        for credential in owned_chatgpt_subscription_logins(team_id, key_hash)
        if (token := fresh_chatgpt_access_token(credential.credential_values, NOW)) is not None
    }

    assert resolved == bound == {name: tokens[name] for name in expected}
    assert (rows.updates, oauth.posts) == ([], [])


@pytest.mark.parametrize(
    "expires_at",
    [None, "not-a-number", ""],
    ids=["missing", "unparsable", "blank"],
)
async def test_a_login_with_an_unknown_expiry_is_refreshed(monkeypatch, expires_at):
    values: Final = _login_values(_access_token("unknown-expiry"), "rt-a", expires_at)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")))
    oauth, new_access_token = _refreshed("new")

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert len(oauth.posts) == 1
    assert resolved == {"login-a": new_access_token}


@pytest.mark.parametrize(
    ("existing", "blocked"),
    [
        (None, False),
        (
            CredentialItem(
                credential_name="taken",
                credential_values={"api_key": "sk-openai"},
                credential_info={"custom_llm_provider": "openai"},
            ),
            True,
        ),
        (
            CredentialItem(credential_name="taken", credential_values={"api_key": "sk-unknown"}, credential_info={}),
            True,
        ),
        (
            CredentialItem(
                credential_name="taken", credential_values={}, credential_info={"custom_llm_provider": "openai"}
            ),
            False,
        ),
        (
            CredentialItem(
                credential_name="taken",
                credential_values={"api_base": "https://chatgpt.com/backend-api/codex"},
                credential_info={"custom_llm_provider": "chatgpt"},
            ),
            False,
        ),
        (
            CredentialItem(
                credential_name="taken",
                credential_values={"chatgpt_access_token": "at", "chatgpt_refresh_token": "rt"},
                credential_info={},
            ),
            False,
        ),
    ],
    ids=["free", "other-provider", "no-provider", "no-values", "chatgpt-provider", "existing-login"],
)
def test_a_sign_in_is_blocked_only_by_another_providers_credential(existing, blocked):
    assert blocks_chatgpt_login(existing) is blocked


async def test_saving_a_new_login_creates_an_encrypted_credential_its_owner_can_use():
    rows: Final = _CredentialRows()
    access_token: Final = _access_token("signed-in")
    values: Final = _login_values(access_token, "rt-signed-in", FRESH_EXPIRES_AT)

    await save_chatgpt_login(rows, "login-new", "Team A ChatGPT", values, ("team-a", None), "admin-user")

    assert rows.updates == []
    assert len(rows.creates) == 1
    created: Final = rows.creates[0]
    written_values: Final = created["credential_values"]
    assert isinstance(written_values, str)
    saved: Final = json.loads(written_values)
    assert saved.keys() == values.keys()
    for key, plaintext in values.items():
        assert saved[key] != plaintext
        assert decrypt_value_helper(saved[key], key) == plaintext
    for secret in (access_token, "rt-signed-in"):
        assert secret not in written_values
    assert json.loads(str(created["credential_info"])) == {
        "custom_llm_provider": "chatgpt",
        "chatgpt_owner_team_id": "team-a",
    }
    assert (created["credential_name"], created["display_name"], created["created_by"]) == (
        "login-new",
        "Team A ChatGPT",
        "admin-user",
    )
    assert await _resolve("team-a", None) == {"login-new": access_token}
    assert await _resolve("team-b", "hash-k") == {}


async def test_saving_a_login_over_another_providers_credential_writes_nothing(monkeypatch):
    existing: Final = CredentialItem(
        credential_name="openai-prod",
        credential_values={"api_key": encrypt_value_helper("sk-openai-prod")},
        credential_info={"custom_llm_provider": "openai"},
    )
    served: Final = existing.model_copy(update={"credential_values": {"api_key": "sk-openai-prod"}})
    _serve(monkeypatch, served)
    rows: Final = _CredentialRows(existing)
    values: Final = _login_values(_access_token("signed-in"), "rt-signed-in", FRESH_EXPIRES_AT)

    saved: Final = await save_chatgpt_login(rows, "openai-prod", None, values, ("team-a", None), "admin-user")

    assert saved is None
    assert (rows.updates, rows.creates) == ([], [])
    assert rows.rows == {"openai-prod": existing}
    assert litellm.credential_list == [served]


async def test_saving_again_on_the_same_name_replaces_the_owner_and_keeps_the_rest(monkeypatch):
    old_values: Final = _login_values(_access_token("old"), "rt-old", FRESH_EXPIRES_AT)
    existing: Final = _login("login-a", old_values, team_id="team-a", note="billing seat 3").model_copy(
        update={"display_name": "Old label"}
    )
    _serve(monkeypatch, existing)
    rows: Final = _CredentialRows(_stored(existing))
    new_access_token: Final = _access_token("rotated")
    new_values: Final = _login_values(new_access_token, "rt-rotated", FRESH_EXPIRES_AT)

    await save_chatgpt_login(rows, "login-a", None, new_values, (None, "hash-k"), "admin-user")

    assert rows.creates == []
    assert [name for name, _ in rows.updates] == ["login-a"]
    written: Final = rows.updates[0][1]
    assert json.loads(str(written["credential_info"])) == {
        "note": "billing seat 3",
        "custom_llm_provider": "chatgpt",
        "chatgpt_owner_key_hash": "hash-k",
    }
    assert written["display_name"] == "Old label"
    assert _decrypted(json.loads(str(written["credential_values"]))) == new_values
    assert await _resolve("team-a", None) == {}
    assert await _resolve(None, "hash-k") == {"login-a": new_access_token}


async def test_a_sign_in_never_replaces_another_providers_credential_written_after_its_check(monkeypatch):
    empty_login: Final = CredentialItem(
        credential_name="login-a", credential_values={}, credential_info={"custom_llm_provider": "chatgpt"}
    )
    other_provider: Final = CredentialItem(
        credential_name="login-a",
        credential_values={"api_key": encrypt_value_helper("sk-written-meanwhile")},
        credential_info={"custom_llm_provider": "openai"},
    )
    _serve(monkeypatch)
    rows: Final = _CredentialRows(empty_login, rewritten_after_reads=(other_provider,))
    values: Final = _login_values(_access_token("signed-in"), "rt-signed-in", FRESH_EXPIRES_AT)

    saved: Final = await save_chatgpt_login(rows, "login-a", None, values, ("team-a", None), "admin-user")

    assert saved is None
    assert rows.rows == {"login-a": other_provider}
    assert await _resolve("team-a", None) == {}


async def test_a_sign_in_saves_over_its_login_refreshed_after_its_check(monkeypatch):
    before_refresh: Final = _login("login-a", _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT), "team-a")
    refreshed: Final = _login(
        "login-a", _login_values(_access_token("refreshed"), "rt-refreshed", FRESH_EXPIRES_AT), "team-a"
    )
    _serve(monkeypatch)
    rows: Final = _CredentialRows(_stored(before_refresh), rewritten_after_reads=(_stored(refreshed),))
    signed_in_access_token: Final = _access_token("signed-in")
    values: Final = _login_values(signed_in_access_token, "rt-signed-in", FRESH_EXPIRES_AT)

    saved: Final = await save_chatgpt_login(rows, "login-a", None, values, ("team-a", None), "admin-user")

    assert saved is not None
    assert _decrypted(rows.rows["login-a"].credential_values) == values
    assert await _resolve("team-a", None) == {"login-a": signed_in_access_token}


@pytest.mark.parametrize("coordinator", [InProcessRefreshCoordinator, _LosingCoordinator], ids=["refresh", "reread"])
async def test_a_login_rebound_to_another_owner_is_not_served_to_the_old_owner(monkeypatch, coordinator):
    memory_values: Final = _login_values(_access_token("team-a-seat"), "rt-team-a", STALE_EXPIRES_AT)
    row_values: Final = _login_values(
        _access_token("team-b-seat", account_id="acct-team-b"), "rt-team-b", FRESH_EXPIRES_AT, "acct-team-b"
    )
    _serve(monkeypatch, _login("login-a", memory_values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, team_id="team-b")))
    oauth: Final = _TokenEndpoint(status=500)

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth, coordinator()))

    assert resolved == {}
    assert oauth.posts == []
    assert _served("login-a") == memory_values


@pytest.mark.parametrize(
    "new_owner",
    [{"team_id": "team-b"}, {"key_hash": "hash-k"}],
    ids=["moved-to-another-team", "moved-to-a-key"],
)
async def test_a_login_moved_to_another_owner_while_its_refresh_is_in_flight_keeps_the_new_tokens_for_the_new_owner(
    monkeypatch, new_owner
):
    values: Final = _login_values(_access_token("team-a-seat"), "rt-team-a", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    moved_info: Final = _login("login-a", values, **new_owner).credential_info

    def move_to_the_new_owner() -> None:
        rows.put(rows.rows["login-a"].model_copy(update={"credential_info": moved_info}))
        _serve(monkeypatch, held.model_copy(update={"credential_info": moved_info}))

    oauth, redeemed_access_token = _refreshed(
        "redeemed", refresh_token="rt-redeemed", during_post=move_to_the_new_owner
    )
    refresher: Final = _refresher(rows, oauth)

    old_owner_token: Final = await refresher.access_token(held)

    assert old_owner_token is None
    assert rows.rows["login-a"].credential_info == moved_info
    row_values: Final = _decrypted(rows.rows["login-a"].credential_values)
    assert (row_values["chatgpt_access_token"], row_values["chatgpt_refresh_token"]) == (
        redeemed_access_token,
        "rt-redeemed",
    )
    served: Final = CredentialAccessor.find_credential("login-a")
    assert served is not None
    assert (served.credential_info, served.credential_values) == (moved_info, values)

    new_owner_token: Final = await owned_chatgpt_subscription_access_token(
        "login-a",
        new_owner.get("team_id"),
        new_owner.get("key_hash"),
        refresher_factory=lambda: refresher,
        clock=_now,
    )

    assert new_owner_token == redeemed_access_token
    assert len(oauth.posts) == 1


async def test_a_refresh_never_writes_back_the_owner_this_replica_moved_the_login_from(monkeypatch):
    values: Final = _login_values(_access_token("team-a-seat"), "rt-team-a", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    moved: Final = _login("login-a", values, team_id="team-b")
    oauth, _ = _refreshed("redeemed", during_post=lambda: _serve(monkeypatch, moved))

    await _refresher(rows, oauth).access_token(held)

    served: Final = CredentialAccessor.find_credential("login-a")
    assert served is not None
    assert served.credential_info == moved.credential_info


async def test_a_refresh_for_one_login_leaves_the_callers_other_stale_logins_alone(monkeypatch):
    stale_a: Final = _login(
        "login-a", _login_values(_access_token("old-a"), "rt-a", STALE_EXPIRES_AT), team_id="team-a"
    )
    stale_b: Final = _login(
        "login-b", _login_values(_access_token("old-b"), "rt-b", STALE_EXPIRES_AT), team_id="team-a"
    )
    _serve(monkeypatch, stale_a, stale_b)
    rows: Final = _CredentialRows(_stored(stale_a), _stored(stale_b))
    oauth, new_access_token = _refreshed("new")

    token: Final = await owned_chatgpt_subscription_access_token(
        "login-a", "team-a", None, refresher_factory=lambda: _refresher(rows, oauth), clock=_now
    )

    assert token == new_access_token
    assert [(body or {}).get("refresh_token") for _, body in oauth.posts] == ["rt-a"]
    assert [name for name, _ in rows.updates] == ["login-a"]
    assert _served("login-b") == stale_b.credential_values


@pytest.mark.parametrize(
    ("team_id", "key_hash", "reads"),
    [("team-b", "hash-k", ["login-a"]), (None, None, [])],
    ids=["another-owner", "no-owner"],
)
async def test_a_stale_login_is_never_refreshed_for_a_caller_that_does_not_own_it(
    monkeypatch, team_id, key_hash, reads
):
    stale: Final = _login("login-a", _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT), team_id="team-a")
    _serve(monkeypatch, stale)
    rows: Final = _CredentialRows(_stored(stale))
    oauth, _ = _refreshed("new")

    token: Final = await owned_chatgpt_subscription_access_token(
        "login-a", team_id, key_hash, refresher_factory=lambda: _refresher(rows, oauth), clock=_now
    )

    assert token is None
    assert (rows.reads, rows.updates, oauth.posts) == (reads, [], [])
    assert _served("login-a") == stale.credential_values


_KEY_SEAT: Final = _login_values(_access_token("key-seat"), "rt-key-seat", FRESH_EXPIRES_AT)


@pytest.mark.parametrize(
    "held",
    [(_login("login-k", _KEY_SEAT, key_hash="hash-old"),), ()],
    ids=["held-for-the-key-before-regeneration", "not-loaded-yet"],
)
async def test_a_login_the_database_gives_the_caller_is_served_before_this_replica_syncs(monkeypatch, held):
    _serve(monkeypatch, *held)
    rows: Final = _CredentialRows(_stored(_login("login-k", _KEY_SEAT, key_hash="hash-new")))
    oauth: Final = _TokenEndpoint(status=500)
    refresher: Final = _refresher(rows, oauth)

    tokens: Final = [
        await owned_chatgpt_subscription_access_token(
            "login-k", None, key_hash, refresher_factory=lambda: refresher, clock=_now
        )
        for key_hash in ("hash-new", "hash-old")
    ]

    assert tokens == [_KEY_SEAT["chatgpt_access_token"], None]
    assert (rows.updates, oauth.posts) == ([], [])


@pytest.mark.parametrize(
    "credential_info",
    [{1: "not-a-string-key", "chatgpt_owner_team_id": "team-a"}, ["chatgpt_owner_team_id", "team-a"]],
    ids=["non_string_key", "not_a_mapping"],
)
async def test_a_login_with_malformed_credential_info_is_served_to_no_one(monkeypatch, credential_info):
    malformed: Final = _login(
        "login-a", _login_values(_access_token("a"), "rt-a", FRESH_EXPIRES_AT), team_id="team-a"
    ).model_copy(update={"credential_info": credential_info})
    _serve(monkeypatch, malformed)
    rows: Final = _CredentialRows(_stored(malformed))
    oauth: Final = _TokenEndpoint(status=500)

    token: Final = await owned_chatgpt_subscription_access_token(
        "login-a", "team-a", None, refresher_factory=lambda: _refresher(rows, oauth), clock=_now
    )

    assert token is None
    assert (rows.updates, oauth.posts) == ([], [])


async def test_a_refresh_that_fails_unexpectedly_serves_no_token(monkeypatch, proxy_log):
    stale: Final = _login("login-a", _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT), team_id="team-a")
    _serve(monkeypatch, stale)

    def unreachable_lock() -> ChatGPTLoginRefresher | None:
        raise ConnectionError("the refresh lock is unreachable")

    token: Final = await owned_chatgpt_subscription_access_token(
        "login-a", "team-a", None, refresher_factory=unreachable_lock, clock=_now
    )

    assert token is None
    assert "login-a" in _logged_text(proxy_log)


async def test_two_replicas_sharing_the_database_lock_redeem_a_stale_login_once_on_the_lock_transaction(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    pooled_rows: Final = _CredentialRows(reads_fail=True, failing_writes=99)
    transaction_rows: Final = _CredentialRows(_stored(held))
    oauth, new_access_token = _refreshed("new")
    database_lock: Final = _DatabaseLock()
    locked_refresh: Final = LockedRefresh(
        database_lock, rows_in=_rows_in_lock_transaction(database_lock, transaction_rows)
    )
    replicas: Final = tuple(_refresher(pooled_rows, oauth, shared_refresh=locked_refresh) for _ in range(2))

    tokens: Final = await asyncio.gather(*(replica.access_token(held) for replica in replicas))

    assert tokens == [new_access_token, new_access_token]
    assert len(oauth.posts) == 1
    assert (pooled_rows.reads, pooled_rows.updates) == ([], [])
    assert _decrypted(transaction_rows.rows["login-a"].credential_values)["chatgpt_access_token"] == new_access_token


async def test_a_failed_save_on_the_lock_transaction_is_retried_and_the_redeemed_login_is_stored(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    transaction_rows: Final = _CredentialRows(_stored(held), failing_writes=1)
    oauth, new_access_token = _refreshed("new")
    database_lock: Final = _DatabaseLock()
    locked_refresh: Final = LockedRefresh(
        database_lock, rows_in=lambda transaction: _RowsOnTransaction(transaction_rows, database_lock.transaction)
    )
    refresher: Final = _refresher(
        _CredentialRows(reads_fail=True, failing_writes=99), oauth, shared_refresh=locked_refresh
    )

    token: Final = await refresher.access_token(held)

    assert token == new_access_token
    assert len(oauth.posts) == 1
    assert _decrypted(transaction_rows.rows["login-a"].credential_values)["chatgpt_access_token"] == new_access_token


async def test_a_refresh_whose_lock_transaction_cannot_commit_still_serves_the_redeemed_token(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    committed_rows: Final = _CredentialRows(_stored(held))
    uncommitted_rows: Final = _CredentialRows(_stored(held))
    oauth, new_access_token = _refreshed("new")
    lock: Final = PrismaAdvisoryLock(_DatabaseThatCannotCommit(), timeout=timedelta(seconds=60))
    refresher: Final = _refresher(
        committed_rows, oauth, shared_refresh=LockedRefresh(lock, rows_in=lambda _: uncommitted_rows)
    )

    token: Final = await refresher.access_token(held)

    assert token == new_access_token
    assert len(oauth.posts) == 1
    assert _served("login-a")["chatgpt_access_token"] == new_access_token


@pytest.mark.parametrize(
    ("row_expires_at", "served_from_row"),
    [(FRESH_EXPIRES_AT, True), (STALE_EXPIRES_AT, False)],
    ids=["refreshed-elsewhere", "still-stale"],
)
async def test_a_replica_that_cannot_take_the_database_lock_only_reads_the_row(
    monkeypatch, row_expires_at, served_from_row
):
    memory_values: Final = _login_values(_access_token("old"), "rt-old", STALE_EXPIRES_AT)
    row_values: Final = _login_values(_access_token("row"), "rt-row", row_expires_at)
    held: Final = _login("login-a", memory_values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(_login("login-a", row_values, team_id="team-a")))
    oauth, _ = _refreshed("new")
    refresher: Final = _refresher(
        rows,
        oauth,
        shared_refresh=LockedRefresh(_UnreachableDatabaseLock(), rows_in=lambda _: _CredentialRows(reads_fail=True)),
    )

    token: Final = await refresher.access_token(held)

    assert token == (row_values["chatgpt_access_token"] if served_from_row else None)
    assert oauth.posts == []
    assert rows.updates == []

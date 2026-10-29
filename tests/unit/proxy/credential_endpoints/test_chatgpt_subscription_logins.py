import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.authenticator import TOKEN_EXPIRY_SKEW_SECONDS
from litellm.llms.chatgpt.common_utils import CHATGPT_OAUTH_TOKEN_URL, OPENAI_AUTH_CLAIM_KEY
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.credential_endpoints.chatgpt_subscription_logins import (
    ChatGPTLoginRefresher,
    blocks_chatgpt_login,
    resolve_chatgpt_subscription_logins,
    save_chatgpt_login,
)
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
    def __init__(self, *rows: CredentialItem, writes_fail: bool = False, reads_fail: bool = False) -> None:
        self.rows: dict[str, CredentialItem] = {row.credential_name: row for row in rows}
        self.updates: list[tuple[str, Mapping[str, object]]] = []
        self.compared: list[tuple[str, str]] = []
        self.reads: list[str] = []
        self.creates: list[Mapping[str, object]] = []
        self.writes_fail = writes_fail
        self.reads_fail = reads_fail

    def put(self, row: CredentialItem) -> None:
        self.rows[row.credential_name] = row

    def remove(self, credential_name: str) -> None:
        self.rows.pop(credential_name)

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        self.reads.append(credential_name)
        if self.reads_fail:
            raise ConnectionError("the database is unreachable")
        return self.rows.get(credential_name)

    async def update_if_value_unchanged(
        self, credential_name: str, value_key: str, stored_value: str, data: Mapping[str, object]
    ) -> bool:
        self.updates.append((credential_name, data))
        self.compared.append((value_key, stored_value))
        if self.writes_fail:
            raise ConnectionError("the database is unreachable")
        current: Final = self.rows.get(credential_name)
        if current is None or current.credential_values.get(value_key) != stored_value:
            return False
        self.rows[credential_name] = _row_after_write(current, data)
        return True

    async def update_by_name(self, credential_name: str, data: Mapping[str, object]) -> None:
        self.updates.append((credential_name, data))
        if self.writes_fail:
            raise ConnectionError("the database is unreachable")
        self.rows[credential_name] = _row_after_write(self.rows[credential_name], data)

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


def _refresher(
    rows: _CredentialRows,
    oauth_client: _TokenEndpoint,
    coordinator: InProcessRefreshCoordinator | _LosingCoordinator | None = None,
    shared_coordinator: _SharedCoordinatorWithoutLock | None = None,
) -> ChatGPTLoginRefresher:
    return ChatGPTLoginRefresher(
        rows=rows,
        oauth_client=oauth_client,
        coordinator=coordinator or InProcessRefreshCoordinator(),
        shared_coordinator=shared_coordinator,
        clock=_now,
    )


async def _resolve(
    team_id: str | None, key_hash: str | None, refresher: ChatGPTLoginRefresher | None = None
) -> Mapping[str, str]:
    return await resolve_chatgpt_subscription_logins(
        team_id, key_hash, refresher_factory=lambda: refresher, clock=_now
    )


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


async def test_a_failed_save_still_serves_the_refreshed_token(monkeypatch, proxy_log):
    values: Final = _login_values(_access_token("old"), "rt-a", STALE_EXPIRES_AT)
    _serve(monkeypatch, _login("login-a", values, team_id="team-a"))
    rows: Final = _CredentialRows(_stored(_login("login-a", values, team_id="team-a")), writes_fail=True)
    oauth, new_access_token = _refreshed("new", refresh_token="rt-rotated")

    resolved: Final = await _resolve("team-a", None, _refresher(rows, oauth))

    assert resolved == {"login-a": new_access_token}
    assert len(oauth.posts) == 1
    assert len(rows.updates) == 1
    assert _served("login-a")["chatgpt_access_token"] == new_access_token
    logged: Final = _logged_text(proxy_log)
    assert "login-a" in logged
    for secret in (new_access_token, "rt-rotated", "rt-a"):
        assert secret not in logged


async def test_concurrent_calls_redeem_once_when_the_shared_lock_refreshes_without_locking(monkeypatch):
    values: Final = _login_values(_access_token("old"), "rt-single-use", STALE_EXPIRES_AT)
    held: Final = _login("login-a", values, team_id="team-a")
    _serve(monkeypatch, held)
    rows: Final = _CredentialRows(_stored(held))
    oauth, new_access_token = _refreshed("new")
    shared: Final = _SharedCoordinatorWithoutLock()
    refresher: Final = _refresher(rows, oauth, shared_coordinator=shared)

    tokens: Final = await asyncio.gather(*(refresher.access_token(held) for _ in range(5)))

    assert tokens == [new_access_token] * 5
    assert len(oauth.posts) == 1
    assert shared.runs == 1


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
    assert rows.compared == [("chatgpt_refresh_token", stored_row.credential_values["chatgpt_refresh_token"])]
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

    await resolve_chatgpt_subscription_logins("team-a", None, refresher_factory=delete_then_refresh, clock=_now)

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

    assert resolved == {name: tokens[name] for name in expected}
    assert (rows.reads, oauth.posts) == ([], [])


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

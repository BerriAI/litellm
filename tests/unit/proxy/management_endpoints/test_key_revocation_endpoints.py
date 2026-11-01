import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi.testclient import TestClient

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    KEY_REVOKE_MAX_KEYS_PER_REQUEST,
    KEY_REVOKE_MAX_REQUESTS_PER_SOURCE,
    KEY_REVOKE_WINDOW_SECONDS,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_AuditLogs, LitellmTableNames, UserAPIKeyAuth
from litellm.proxy.auth.key_revoke_throttle import KeyRevokeThrottle
from litellm.proxy.deps.database import get_database
from litellm.proxy.management_endpoints.key_revocation_endpoints import (
    get_key_revocation_audit,
    get_key_revoke_throttle,
)
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import hash_token
from litellm.repositories.database import Database
from tests._master_key import MASTER_KEY
from tests._support.native_database import native_database

client: Final = TestClient(app)
NATIVE: Final = native_database()

LIVE: Final = "sk-live-0123456789abcdefABCDEF"
OTHER: Final = "sk-other_0123456789abcdef"
BLOCKED: Final = "sk-blocked-0123456789abcdef"
NEVER_EXISTED: Final = "sk-never-existed-0123456789"
LIVE_HASH: Final = hash_token(LIVE)
OTHER_HASH: Final = hash_token(OTHER)
BLOCKED_HASH: Final = hash_token(BLOCKED)
STORAGE_ERROR: Final = "connection to db-internal-7.corp:5432 reset while reading LiteLLM_VerificationToken"
ACCEPTED: Final = (202, b"{}")


class _TokenTable:
    def __init__(self, rows: Mapping[str, Mapping[str, object]], fail_on: str | None) -> None:
        self.rows: Final = {token: dict(row) for token, row in rows.items()}
        self.updates: Final[list[tuple[str, ...]]] = []
        self.fail_on: Final = fail_on

    async def find_many(self, where: Mapping[str, Mapping[str, Sequence[str]]]) -> list[dict[str, object]]:
        if self.fail_on == "find":
            raise RuntimeError(STORAGE_ERROR)
        return [dict(self.rows[token]) for token in where["token"]["in"] if token in self.rows]

    async def update_many(self, data: Mapping[str, object], where: Mapping[str, Mapping[str, Sequence[str]]]) -> int:
        if self.fail_on == "update":
            raise RuntimeError(STORAGE_ERROR)
        tokens: Final = tuple(where["token"]["in"])
        self.updates.append(tokens)
        for token in tokens:
            self.rows[token].update(data)
        return len(tokens)

    def blocked(self, hashed_token: str) -> bool:
        return self.rows[hashed_token]["blocked"] is True


def _row(hashed_token: str, blocked: bool | None = None) -> dict[str, object]:
    return {"token": hashed_token, "blocked": blocked, "team_id": f"team-{hashed_token[:8]}"}


def _install(
    monkeypatch: pytest.MonkeyPatch, rows: Mapping[str, Mapping[str, object]], fail_on: str | None = None
) -> _TokenTable:
    table: Final = _TokenTable(rows, fail_on)
    prisma: Final = SimpleNamespace(writer_db=SimpleNamespace(litellm_verificationtoken=table))
    monkeypatch.setitem(app.dependency_overrides, get_database, lambda: Database(prisma=prisma, native=NATIVE))
    return table


def _revoke(keys: object, headers: Mapping[str, str] | None = None) -> tuple[int, bytes]:
    response: Final = client.post("/key/revoke", json={"keys": keys}, headers=headers)
    return response.status_code, response.content


@pytest.fixture(autouse=True)
def throttle(monkeypatch: pytest.MonkeyPatch) -> KeyRevokeThrottle:
    fresh: Final = KeyRevokeThrottle(source="203.0.113.7", counters=InMemoryCache(), redis_cache=None)
    monkeypatch.setitem(app.dependency_overrides, get_key_revoke_throttle, lambda: fresh)
    return fresh


@pytest.fixture(autouse=True)
def audit_rows(monkeypatch: pytest.MonkeyPatch) -> list[LiteLLM_AuditLogs]:
    recorded: Final[list[LiteLLM_AuditLogs]] = []

    async def record(rows: Sequence[LiteLLM_AuditLogs]) -> None:
        recorded.extend(rows)

    monkeypatch.setitem(app.dependency_overrides, get_key_revocation_audit, lambda: record)
    monkeypatch.setattr(litellm, "store_audit_logs", True)
    return recorded


def test_a_single_live_key_is_blocked_and_answered_with_an_empty_202(monkeypatch):
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH), OTHER_HASH: _row(OTHER_HASH)})

    assert _revoke(LIVE) == ACCEPTED
    assert table.blocked(LIVE_HASH)
    assert isinstance(table.rows[LIVE_HASH]["settings_updated_at"], datetime)
    assert not table.blocked(OTHER_HASH)


def test_the_log_names_the_hashed_token_and_source_but_never_the_raw_key(monkeypatch, caplog):
    _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})
    verbose_proxy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name):
            assert _revoke([LIVE, NEVER_EXISTED]) == ACCEPTED
    finally:
        verbose_proxy_logger.removeHandler(caplog.handler)

    messages: Final = tuple(record.getMessage() for record in caplog.records)
    assert any(LIVE_HASH in message and "203.0.113.7" in message for message in messages), messages
    assert not any(LIVE in message or NEVER_EXISTED in message for message in messages), messages


def test_every_key_in_a_full_batch_is_blocked(monkeypatch):
    keys: Final = tuple(f"sk-batch-{i:03d}" for i in range(KEY_REVOKE_MAX_KEYS_PER_REQUEST))
    table: Final = _install(monkeypatch, {hash_token(key): _row(hash_token(key)) for key in keys})

    assert _revoke(list(keys)) == ACCEPTED
    assert [table.blocked(hash_token(key)) for key in keys] == [True] * KEY_REVOKE_MAX_KEYS_PER_REQUEST


def test_revoking_evicts_the_cached_key_object(monkeypatch):
    _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})
    cache: Final = proxy_server.user_api_key_cache
    cache.set_cache(key=LIVE_HASH, value=UserAPIKeyAuth(token=LIVE_HASH).model_dump())
    assert cache.get_cache(key=LIVE_HASH) is not None

    assert _revoke(LIVE) == ACCEPTED
    assert cache.get_cache(key=LIVE_HASH) is None


def test_duplicate_entries_are_blocked_with_one_update_and_one_audit_row(monkeypatch, audit_rows):
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})

    assert _revoke([LIVE, LIVE, LIVE]) == ACCEPTED
    assert table.updates == [(LIVE_HASH,)]
    assert [row.object_id for row in audit_rows] == [LIVE_HASH]


@pytest.mark.parametrize(
    "body",
    [
        {"keys": []},
        {"keys": [f"sk-batch-{i:03d}" for i in range(KEY_REVOKE_MAX_KEYS_PER_REQUEST + 1)]},
        {"keys": [LIVE, 5]},
        {"keys": 5},
        {"keys": None},
        {},
    ],
    ids=["empty-list", "over-the-cap", "non-string-entry", "not-a-string-or-list", "null", "missing"],
)
def test_schema_violations_are_rejected_without_spending_the_allowance(monkeypatch, body):
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})

    statuses: Final = tuple(
        client.post("/key/revoke", json=body).status_code for _ in range(KEY_REVOKE_MAX_REQUESTS_PER_SOURCE)
    )
    assert statuses == (422,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert not table.blocked(LIVE_HASH)
    assert _revoke(LIVE) == ACCEPTED
    assert table.blocked(LIVE_HASH)


def test_a_hashed_token_never_blocks_its_key(monkeypatch):
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH), OTHER_HASH: _row(OTHER_HASH)})

    assert _revoke([LIVE_HASH, OTHER]) == ACCEPTED
    assert not table.blocked(LIVE_HASH)
    assert table.blocked(OTHER_HASH)


@pytest.mark.parametrize(
    "junk",
    [
        "sk-" + "a" * 98,
        "sk-has space",
        "sk-trailing-newline\n",
        "SK-UPPERCASE-PREFIX",
        "sk-",
        "sk-semi;colon",
        "not-a-key",
        "",
    ],
    ids=[
        "101-chars",
        "space",
        "trailing-newline",
        "uppercase-prefix",
        "bare-prefix",
        "semicolon",
        "no-prefix",
        "empty",
    ],
)
def test_a_junk_entry_blocks_nothing_even_when_a_row_has_its_hash(monkeypatch, junk):
    table: Final = _install(monkeypatch, {hash_token(junk): _row(hash_token(junk))})

    assert _revoke(junk) == ACCEPTED
    assert not table.blocked(hash_token(junk))


@pytest.mark.parametrize("key", ["sk-" + "a" * 97, "sk-under_score-and-dash"], ids=["100-chars", "underscore-dash"])
def test_every_raw_key_shape_up_to_100_chars_is_revocable(monkeypatch, key):
    table: Final = _install(monkeypatch, {hash_token(key): _row(hash_token(key))})

    assert _revoke(key) == ACCEPTED
    assert table.blocked(hash_token(key))


def test_an_already_blocked_key_gets_no_second_update_or_audit_row(monkeypatch, audit_rows):
    table: Final = _install(monkeypatch, {BLOCKED_HASH: _row(BLOCKED_HASH, blocked=True)})

    assert _revoke(BLOCKED) == ACCEPTED
    assert table.blocked(BLOCKED_HASH)
    assert table.updates == []
    assert audit_rows == []


@pytest.mark.parametrize(
    "authorization",
    [f"Bearer {MASTER_KEY}", "Bearer sk-not-a-real-key", "Basic Zm9vOmJhcg==", "garbage"],
    ids=["master-key", "unknown-key", "basic", "garbage"],
)
def test_an_authorization_header_changes_nothing(monkeypatch, authorization):
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})

    assert _revoke(LIVE, headers={"Authorization": authorization}) == ACCEPTED
    assert table.blocked(LIVE_HASH)


def test_without_a_database_the_request_is_accepted_and_does_nothing(monkeypatch, audit_rows):
    monkeypatch.setitem(app.dependency_overrides, get_database, lambda: None)

    assert _revoke(LIVE) == ACCEPTED
    assert audit_rows == []


def test_without_a_database_requests_still_spend_the_allowance(monkeypatch):
    monkeypatch.setitem(app.dependency_overrides, get_database, lambda: None)

    admitted: Final = tuple(_revoke(LIVE) for _ in range(KEY_REVOKE_MAX_REQUESTS_PER_SOURCE))
    assert admitted == (ACCEPTED,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert _revoke(LIVE)[0] == 429


@pytest.mark.parametrize("fail_on", ["find", "update"])
def test_a_storage_failure_answers_a_generic_retryable_500(monkeypatch, audit_rows, fail_on):
    _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)}, fail_on=fail_on)

    response: Final = client.post("/key/revoke", json={"keys": LIVE})

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "message": "Could not process the revocation request. Retry it.",
            "type": "internal_server_error",
            "param": None,
            "code": "500",
        }
    }
    assert STORAGE_ERROR not in response.text
    assert audit_rows == []


def test_each_newly_blocked_key_gets_an_audit_row_naming_the_public_revocation(monkeypatch, audit_rows):
    _install(
        monkeypatch,
        {
            LIVE_HASH: _row(LIVE_HASH),
            OTHER_HASH: _row(OTHER_HASH, blocked=False),
            BLOCKED_HASH: _row(BLOCKED_HASH, blocked=True),
        },
    )

    assert _revoke([LIVE, OTHER, BLOCKED]) == ACCEPTED
    assert sorted(
        (row.object_id, row.action, row.changed_by, row.changed_by_api_key, row.table_name) for row in audit_rows
    ) == sorted(
        (hashed, "blocked", "litellm_public_key_revocation", None, LitellmTableNames.KEY_TABLE_NAME)
        for hashed in (LIVE_HASH, OTHER_HASH)
    )
    assert len({row.id for row in audit_rows}) == 2
    before_values: Final = tuple(json.loads(str(row.before_value)) for row in audit_rows)
    assert sorted((before["team_id"], before["blocked"]) for before in before_values) == sorted(
        [(f"team-{LIVE_HASH[:8]}", None), (f"team-{OTHER_HASH[:8]}", False)]
    )


def test_with_audit_logging_off_keys_are_blocked_without_audit_rows(monkeypatch, audit_rows):
    monkeypatch.setattr(litellm, "store_audit_logs", False)
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})

    assert _revoke(LIVE) == ACCEPTED
    assert table.blocked(LIVE_HASH)
    assert audit_rows == []


def test_the_response_is_byte_identical_whatever_the_submitted_keys_are(monkeypatch):
    table: Final = _install(
        monkeypatch,
        {
            LIVE_HASH: _row(LIVE_HASH),
            OTHER_HASH: _row(OTHER_HASH),
            BLOCKED_HASH: _row(BLOCKED_HASH, blocked=True),
            hash_token("sk-mixed-live-key"): _row(hash_token("sk-mixed-live-key")),
        },
    )
    submissions: Final[tuple[object, ...]] = (
        LIVE,
        BLOCKED,
        NEVER_EXISTED,
        OTHER_HASH,
        "not a key at all",
        [LIVE, BLOCKED, NEVER_EXISTED, OTHER_HASH, "junk", "sk-mixed-live-key"],
    )

    def answer(keys: object) -> tuple[int, bytes, str | None, str | None]:
        response: Final = client.post("/key/revoke", json={"keys": keys})
        return (
            response.status_code,
            response.content,
            response.headers.get("content-type"),
            response.headers.get("content-length"),
        )

    answers: Final = tuple(answer(keys) for keys in submissions)

    assert answers == ((202, b"{}", "application/json", "2"),) * len(submissions)
    assert table.blocked(LIVE_HASH)
    assert table.blocked(hash_token("sk-mixed-live-key"))
    assert not table.blocked(OTHER_HASH)


@pytest.mark.parametrize("filler", [NEVER_EXISTED, "not a key", OTHER_HASH], ids=["unknown-key", "junk", "hash"])
def test_past_the_hourly_allowance_a_request_is_refused_and_blocks_nothing(monkeypatch, filler):
    table: Final = _install(monkeypatch, {LIVE_HASH: _row(LIVE_HASH)})

    admitted: Final = tuple(_revoke(filler) for _ in range(KEY_REVOKE_MAX_REQUESTS_PER_SOURCE))
    refused: Final = client.post("/key/revoke", json={"keys": LIVE})

    assert admitted == (ACCEPTED,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == str(KEY_REVOKE_WINDOW_SECONDS)
    assert refused.json() == {
        "error": {
            "message": "Too many revocation requests from this address. Retry later.",
            "type": "rate_limit_error",
            "param": None,
            "code": "429",
        }
    }
    assert not table.blocked(LIVE_HASH)

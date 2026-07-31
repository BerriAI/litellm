from __future__ import annotations

import logging

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.resolvers.divergence import (
    SpendIdentity,
    log_identity_divergence,
    spend_identity_divergence,
)


def _key(**overrides: str | None) -> UserAPIKeyAuth:
    base: dict[str, str | None] = {
        "token": "hashed-token",
        "user_id": "u-1",
        "team_id": "t-1",
        "org_id": "o-1",
    }
    base.update(overrides)
    return UserAPIKeyAuth(**base)


def _consumed(**overrides: str | None) -> SpendIdentity:
    base: dict[str, str | None] = {
        "user_id": "u-1",
        "team_id": "t-1",
        "org_id": "o-1",
    }
    base.update(overrides)
    return SpendIdentity(**base)


def test_consumed_matches_resolved_is_silent(caplog):
    key = _key()
    consumed = _consumed()

    assert (
        spend_identity_divergence(
            SpendIdentity(user_id="u-1", team_id="t-1", org_id="o-1"),
            consumed,
        )
        == ()
    )

    with caplog.at_level(logging.WARNING):
        log_identity_divergence(key, consumed)
    assert caplog.records == []


def test_empty_consumed_metadata_against_populated_key_warns_naming_fields(caplog):
    key = _key()
    consumed = _consumed(user_id=None, team_id=None, org_id=None)

    diverged = spend_identity_divergence(
        SpendIdentity(user_id="u-1", team_id="t-1", org_id="o-1"),
        consumed,
    )
    by_field = {d.field: (d.resolved_value, d.consumed_value) for d in diverged}

    assert set(by_field) == {"user_id", "team_id", "org_id"}
    assert by_field["user_id"] == ("u-1", None)
    assert by_field["team_id"] == ("t-1", None)
    assert by_field["org_id"] == ("o-1", None)

    with caplog.at_level(logging.WARNING):
        log_identity_divergence(key, consumed)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "user_id" in message and "team_id" in message and "org_id" in message


def test_different_consumed_user_id_warns(caplog):
    key = _key()
    consumed = _consumed(user_id="u-OTHER")

    diverged = spend_identity_divergence(
        SpendIdentity(user_id="u-1", team_id="t-1", org_id="o-1"),
        consumed,
    )
    by_field = {d.field: (d.resolved_value, d.consumed_value) for d in diverged}

    assert by_field == {"user_id": ("u-1", "u-OTHER")}

    with caplog.at_level(logging.WARNING):
        log_identity_divergence(key, consumed)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "user_id" in message and "u-1" in message and "u-OTHER" in message

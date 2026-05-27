"""Regression tests for LIT-2569: auto-extend expiration on key regenerate
when the existing key is already expired and the caller did not supply a
new duration.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_prisma_capturing_update():
    captured = {}
    mock = AsyncMock()

    async def _fake_update(*, where, data):
        captured["data"] = data

        class R:
            def __init__(self, d):
                self._d = d

            def __iter__(self):
                return iter(self._d.items())

        return R(
            {
                "token": data["token"],
                "key_name": data["key_name"],
                "user_id": "user-1",
                "expires": data.get("expires"),
            }
        )

    mock.db.litellm_verificationtoken.update = _fake_update
    mock.db.litellm_verificationtoken.create = AsyncMock(return_value=None)
    mock.jsonify_object = MagicMock(side_effect=lambda data: data)
    mock._captured = captured
    return mock


def _make_expired_key(*, days_ago_created=31, days_ago_expired=1):
    from litellm.proxy._types import LiteLLM_VerificationToken

    now = datetime.now(timezone.utc)
    return LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=now - timedelta(days=days_ago_created),
        expires=now - timedelta(days=days_ago_expired),
    )


async def _run_regenerate(existing_key, data):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _execute_virtual_key_regeneration,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-admin",
        user_id="admin-1",
    )
    mock_prisma = _mock_prisma_capturing_update()

    with (
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.get_new_token",
            new_callable=AsyncMock,
            return_value="sk-newtoken1234ab12",
        ),
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints._insert_deprecated_key",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.KeyManagementEventHooks.async_key_rotated_hook",
            new_callable=AsyncMock,
        ),
    ):
        await _execute_virtual_key_regeneration(
            prisma_client=mock_prisma,
            key_in_db=existing_key,
            hashed_api_key="abc123",
            key="abc123",
            data=data,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=MagicMock(),
        )

    return mock_prisma._captured["data"]


@pytest.mark.asyncio
async def test_lit2569_regenerate_expired_key_without_duration_auto_extends():
    """Headline: regenerating an already-expired key without supplying
    duration produces a key whose expires is in the future, with
    the original lifetime (~30 days) preserved."""
    from litellm.proxy._types import RegenerateKeyRequest

    existing_key = _make_expired_key(days_ago_created=31, days_ago_expired=1)
    data = RegenerateKeyRequest()  # NO duration

    written = await _run_regenerate(existing_key, data)

    new_expires = written.get("expires")
    assert new_expires is not None, (
        "BUG: expires not set on regenerate of expired key -- new key "
        "would inherit the existing (expired) date."
    )
    assert new_expires > datetime.now(timezone.utc), (
        f"BUG: regenerated key still expired (new expires={new_expires!r})"
    )
    # Original lifetime (~30d) should be preserved (within tolerance for
    # the test runners wall-clock drift between key creation and now).
    delta = new_expires - datetime.now(timezone.utc)
    assert timedelta(days=29) < delta < timedelta(days=31), (
        f"Expected ~30d lifetime, got {delta}"
    )


@pytest.mark.asyncio
async def test_lit2569_regenerate_non_expired_key_unchanged():
    """A key whose expires is still in the future must NOT be
    auto-extended. This guards against regressions for callers who rely
    on /key/regenerate being a no-op on the expiration field."""
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    now = datetime.now(timezone.utc)
    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=now - timedelta(days=5),
        expires=now + timedelta(days=25),  # still valid for 25 days
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())

    assert "expires" not in written, (
        'non-expired key was auto-extended; expires=' + repr(written.get('expires'))
    )


@pytest.mark.asyncio
async def test_lit2569_regenerate_explicit_duration_still_wins():
    """When the caller explicitly supplies duration, the auto-extend
    branch must not fire -- the supplied value takes precedence."""
    from litellm.proxy._types import RegenerateKeyRequest

    existing_key = _make_expired_key()
    data = RegenerateKeyRequest(duration="7d")

    written = await _run_regenerate(existing_key, data)
    new_expires = written.get("expires")
    assert new_expires is not None
    delta = new_expires - datetime.now(timezone.utc)
    assert timedelta(days=6) < delta < timedelta(days=8), (
        f"Expected ~7d from explicit duration, got {delta}"
    )


@pytest.mark.asyncio
async def test_lit2569_regenerate_non_expiring_key_unchanged():
    """A key with expires=None (never expires) must remain
    non-expiring after regenerate without duration."""
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=datetime.now(timezone.utc) - timedelta(days=5),
        expires=None,
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())

    assert "expires" not in written, (
        "non-expiring key was given an expires by the LIT-2569 branch"
    )


@pytest.mark.asyncio
async def test_lit2569_regenerate_expired_key_with_string_expires_parses_iso():
    """Covers the defensive `isinstance(_existing_expires, str)` branch:
    if Prisma surfaces ``expires`` as an ISO-8601 string instead of a
    datetime, the auto-extend logic must still fire."""
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    now = datetime.now(timezone.utc)
    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=now - timedelta(days=31),
        # ISO-8601 string with trailing Z, the shape datetime.fromisoformat()
        # used to choke on prior to py3.11. The branch normalizes "Z" to "+00:00".
        expires=(now - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())
    new_expires = written.get("expires")
    assert new_expires is not None, (
        "expected auto-extend to fire when expires is provided as an ISO string"
    )
    assert new_expires > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_lit2569_regenerate_expired_key_tz_naive_expires_treated_as_utc():
    """Defensive branch: if either ``expires`` or ``created_at`` is
    timezone-naive (legacy rows / SQLite-backed installs), the helper
    must coerce to UTC so the ``< now`` comparison is well-defined."""
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    now = datetime.now(timezone.utc)
    # Construct expires/created_at without tzinfo
    expires_naive = (now - timedelta(days=1)).replace(tzinfo=None)
    created_naive = (now - timedelta(days=31)).replace(tzinfo=None)
    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=created_naive,
        expires=expires_naive,
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())
    new_expires = written.get("expires")
    assert new_expires is not None
    assert new_expires > datetime.now(timezone.utc)
    delta = new_expires - datetime.now(timezone.utc)
    assert timedelta(days=29) < delta < timedelta(days=31)


@pytest.mark.asyncio
async def test_lit2569_regenerate_expired_key_missing_created_at_falls_back_to_30d():
    """Defensive branch: if ``created_at`` is missing on the row (very
    old keys that pre-date that column), the helper falls back to a
    30-day lifetime so the regenerated key is still usable."""
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    now = datetime.now(timezone.utc)
    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=None,
        expires=now - timedelta(days=1),
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())
    new_expires = written.get("expires")
    assert new_expires is not None
    delta = new_expires - datetime.now(timezone.utc)
    # 30-day fallback (within tolerance)
    assert timedelta(days=29) < delta < timedelta(days=31), (
        f"Expected 30d fallback, got {delta}"
    )


@pytest.mark.asyncio
async def test_lit2569_auto_extend_respects_upperbound_duration():
    """LIT-2569 + Veria review: when ``upperbound_key_generate_params.duration``
    is configured, the auto-extend lifetime is capped to that bound. Without
    this cap a caller could revive a 30-day key on a deployment that has
    since been locked down to e.g. ``duration="1d"`` by omitting
    ``duration`` from the regenerate request.
    """
    import litellm
    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.types.proxy.management_endpoints.ui_sso import (
        LiteLLM_UpperboundKeyGenerateParams,
    )

    existing_key = _make_expired_key(days_ago_created=31, days_ago_expired=1)
    saved = litellm.upperbound_key_generate_params
    try:
        litellm.upperbound_key_generate_params = (
            LiteLLM_UpperboundKeyGenerateParams(duration="1d")
        )
        written = await _run_regenerate(existing_key, RegenerateKeyRequest())
    finally:
        litellm.upperbound_key_generate_params = saved

    new_expires = written.get("expires")
    assert new_expires is not None
    delta = new_expires - datetime.now(timezone.utc)
    # Cap is 1d -- new expires should be ~1d out, NOT the original ~30d.
    assert delta < timedelta(days=2), (
        f"BUG: auto-extend bypassed upperbound cap; new expires {delta} > 1d"
    )
    assert delta > timedelta(hours=23)


@pytest.mark.asyncio
async def test_lit2569_auto_extend_caps_lifetime_at_30d_no_creep():
    """LIT-2569 + Greptile review: prevent lifetime creep on successive
    regenerations. ``created_at`` does not change on regenerate, so a naive
    ``expires - created_at`` lifetime calculation grows on each cycle
    (30d -> 61d -> 123d). Cap the auto-extended lifetime at a 30-day
    default when no upperbound is configured.
    """
    from litellm.proxy._types import LiteLLM_VerificationToken, RegenerateKeyRequest

    now = datetime.now(timezone.utc)
    # Simulate a key that has already been regenerated twice with lifetime
    # creep: ``expires - created_at`` is now ~90d even though it started
    # life as a 30-day key.
    existing_key = LiteLLM_VerificationToken(
        token="abc123",
        user_id="user-1",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
        created_at=now - timedelta(days=120),
        expires=now - timedelta(days=1),  # still expired
    )

    written = await _run_regenerate(existing_key, RegenerateKeyRequest())
    new_expires = written.get("expires")
    assert new_expires is not None
    delta = new_expires - datetime.now(timezone.utc)
    # Must be capped at the 30-day default, not the naive 119d
    # (created_at -> expires) span.
    assert delta < timedelta(days=31), (
        f"BUG: lifetime creep -- new expires {delta} > 30d cap"
    )
    assert delta > timedelta(days=29)

import pytest
from unittest.mock import AsyncMock, patch
import litellm
from litellm.proxy.auth.user_api_key_auth import (
    _run_post_custom_auth_checks,
    update_valid_token_with_end_user_params,
)
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_without_end_user_id():
    # common_checks now runs in the user_api_key_auth wrapper via
    # _run_centralized_common_checks, gated by
    # custom_auth_run_common_checks for custom-auth deployments. The
    # helper itself no longer calls common_checks.
    valid_token = UserAPIKeyAuth(token="test_token")

    # Default: common_checks should NOT be called inside the helper
    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ) as mock_common:
        mock_common.return_value = True
        result = await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        assert result.token == "test_token"
        assert getattr(result, "end_user_id", None) is None
        mock_common.assert_not_awaited()

    # With opt-in flag: still not from the helper — the centralized gate
    # in the wrapper handles it.
    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
        ) as mock_common,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        mock_common.return_value = True
        result = await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        assert result.token == "test_token"
        mock_common.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_with_end_user_budget_exceeded():
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="test_user",
        end_user_model_max_budget={
            "gpt-4": {"budget_limit": 10.0, "time_period": "1d"}
        },
    )
    request_data = {"model": "gpt-4"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ):
        with patch(
            "litellm.proxy.proxy_server.model_max_budget_limiter.is_end_user_within_model_budget",
            new_callable=AsyncMock,
        ) as mock_budget_check:
            mock_budget_check.side_effect = litellm.BudgetExceededError(
                message="Exceeded budget", current_cost=20.0, max_budget=10.0
            )

            with pytest.raises(litellm.BudgetExceededError):
                await _run_post_custom_auth_checks(
                    valid_token=valid_token,
                    request=None,
                    request_data=request_data,
                    route="/v1/chat/completions",
                    parent_otel_span=None,
                )
            mock_budget_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_auth_enforces_end_user_budget_when_common_checks_skipped():
    # custom-auth deployments with custom_auth_run_common_checks unset skip
    # common_checks() (and its end-user budget enforcement) in the centralized
    # gate, so the helper must enforce the end-user budget itself. Regression:
    # an over-budget end user must be rejected on this path.
    valid_token = UserAPIKeyAuth(token="test_token", end_user_id="customer-1")
    over_budget_end_user = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    async def mock_get_current_spend(
        counter_key, fallback_spend, max_budget=None, **kwargs
    ):
        if counter_key == "spend:end_user:customer-1":
            return 5.0
        return fallback_spend

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
            new_callable=AsyncMock,
            return_value=over_budget_end_user,
        ),
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data={"model": "gpt-4"},
                route="/v1/chat/completions",
                parent_otel_span=None,
            )


@pytest.mark.asyncio
async def test_custom_auth_defers_end_user_budget_to_common_checks_when_enabled():
    # With custom_auth_run_common_checks set, the wrapper's common_checks()
    # enforces the end-user budget, so the helper must not double-enforce it.
    valid_token = UserAPIKeyAuth(token="test_token", end_user_id="customer-1")
    end_user_obj = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
            new_callable=AsyncMock,
            return_value=end_user_obj,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth._check_end_user_budget",
            new_callable=AsyncMock,
        ) as mock_check,
        patch(
            "litellm.proxy.auth.user_api_key_auth._enforce_key_and_fallback_model_access",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={"model": "gpt-4"},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_auth_token_budget_still_loads_and_caches_unrestricted_end_user():
    """
    A token-supplied end_user_max_budget must leave the end-user row in cache.

    Custom auth can set that budget for a customer whose own row carries no budget, block, region
    or permission, which keeps the row out of the cached restricted-id registry that lets auth skip
    the read. The end-user spend counter seeds from this cache entry, so skipping the read would
    cold-start the counter at 0 and under-count a customer who has already spent 100.
    """
    from unittest.mock import MagicMock

    from litellm.proxy.auth.user_api_key_auth import _lookup_end_user_and_apply_budget
    from litellm.proxy.common_utils.user_api_key_cache import (
        UserApiKeyCache,
        end_user_cache_key,
    )

    end_user_row = MagicMock()
    end_user_row.user_id = "customer-1"
    end_user_row.dict = lambda: {
        "user_id": "customer-1",
        "blocked": False,
        "spend": 100.0,
    }

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_endusertable.find_unique = AsyncMock(return_value=end_user_row)
    cache = UserApiKeyCache()

    _, end_user_object = await _lookup_end_user_and_apply_budget(
        valid_token=UserAPIKeyAuth(
            token="test_token",
            end_user_id="customer-1",
            end_user_max_budget=50.0,
        ),
        route="/v1/chat/completions",
        parent_otel_span=None,
        prisma_client=mock_prisma,
        user_api_key_cache=cache,
        proxy_logging_obj=MagicMock(),
    )

    assert end_user_object is not None
    assert end_user_object.spend == 100.0
    assert await cache.async_get_cache(key=end_user_cache_key("customer-1")) is not None


def test_update_valid_token_does_not_override_custom_auth_values_with_none():
    """
    Greptile feedback: if custom auth sets end_user_model_max_budget on the token,
    but the DB end_user has no model_max_budget in their budget table, the DB lookup
    should NOT clear the custom-auth-provided value.
    """
    custom_auth_budget = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="user_1",
        end_user_tpm_limit=100,
        end_user_rpm_limit=50,
        end_user_model_max_budget=custom_auth_budget,
    )

    # Simulate DB lookup that found the end_user but budget table has no limits set
    end_user_params = {
        "end_user_id": "user_1",
        "allowed_model_region": None,
        # No tpm_limit, rpm_limit, or model_max_budget from DB
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    # Custom-auth-provided values should be preserved, not cleared to None
    assert result.end_user_tpm_limit == 100
    assert result.end_user_rpm_limit == 50
    assert result.end_user_model_max_budget == custom_auth_budget
    assert result.end_user_id == "user_1"


def test_update_valid_token_db_values_override_custom_auth_when_set():
    """
    When the DB budget table has explicit values, they should override
    whatever the custom auth function set (DB is source of truth).
    """
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="user_1",
        end_user_tpm_limit=100,
        end_user_model_max_budget={"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}},
    )

    db_budget = {"gpt-4": {"budget_limit": 20.0, "time_period": "1d"}}
    end_user_params = {
        "end_user_id": "user_1",
        "end_user_tpm_limit": 500,
        "end_user_model_max_budget": db_budget,
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    # DB values should win
    assert result.end_user_tpm_limit == 500
    assert result.end_user_model_max_budget == db_budget


# ---------------------------------------------------------------------------
# Regression test: end-user rpm_limit must be stamped on cache-hit requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_builder_stamps_end_user_rpm_limit_on_cache_hit():
    """
    Builder-level regression test.

    When a virtual key is served from the in-memory cache (cache hit), the
    `if valid_token is None:` DB-miss block is skipped entirely, including the
    raw assignments that copy end_user_rpm_limit / end_user_tpm_limit onto the
    token.  Without the unconditional update_valid_token_with_end_user_params
    call added by this fix, the token reaches parallel_request_limiter_v3 with
    both fields as None, so no rate-limit bucket is created and every request
    succeeds regardless of the configured rpm_limit.

    This test fails without the fix and passes with it.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import Request
    from starlette.datastructures import URL

    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_EndUserTable,
        LitellmUserRoles,
    )
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    from litellm.proxy.proxy_server import hash_token
    import litellm as _litellm

    api_key = "sk-test-rpm-cache-hit-builder"
    hashed_key = hash_token(api_key)

    # Token as it sits in the cache — no end_user fields populated
    cached_token = UserAPIKeyAuth(
        api_key=api_key,
        token=hashed_key,
        user_role=LitellmUserRoles.INTERNAL_USER,
        end_user_rpm_limit=None,
        end_user_tpm_limit=None,
    )

    # End-user budget object — not used in this test path (get_end_user_object returns None)
    # The budget is provided via max_end_user_budget_id default budget instead

    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.delete_cache = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    _attrs = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    _originals = {k: getattr(_proxy_server_mod, k, None) for k in _attrs}
    _original_max_end_user_budget_id = getattr(_litellm, "max_end_user_budget_id", None)

    try:
        for k, v in _attrs.items():
            setattr(_proxy_server_mod, k, v)
        # Trigger the default-budget branch inside the builder
        _litellm.max_end_user_budget_id = "tier-default"

        request = Request(scope={"type": "http"})
        request._url = URL(url="/v1/chat/completions")

        # _resolve_key is called twice:
        #   1. check_cache_only=True  → returns cached_token (cache HIT)
        #   2. check_cache_only=False → must NOT be reached; raise to prove it
        # This forces the cache-hit code path and skips the DB-miss block
        # (including its raw end_user_rpm_limit assignments).
        call_count = {"n": 0}

        async def resolve_side_effect(hashed_token):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return cached_token   # cache hit
            raise AssertionError("DB fetch must not be reached on a cache hit")

        with (
            patch(  # test-quality-ok: must intercept cache-layer to simulate cache-hit without a real Redis/DB
                "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
                side_effect=resolve_side_effect,
            ),
            # End-user object is None — simulates the case where the end user
            # row doesn't exist yet but max_end_user_budget_id provides limits.
            # In this path _end_user_object stays None so valid_token_dict does
            # NOT get updated with end_user_params at the bottom of the builder,
            # making the unconditional update_valid_token_with_end_user_params
            # call the only place where the limits reach the returned token.
            patch(  # test-quality-ok: must control DB return to isolate cache-hit path without a real DB
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(  # test-quality-ok: must supply end_user_id without a real request/DB round-trip
                "litellm.proxy.auth.user_api_key_auth.resolve_and_validate_end_user_id",
                new_callable=AsyncMock,
                return_value="alice@example.com",
            ),
            # Simulate max_end_user_budget_id path: default budget provides limits
            patch(  # test-quality-ok: must return a known budget without a real DB to assert rpm_limit propagation
                "litellm.proxy.auth.auth_checks.get_default_end_user_budget",
                new_callable=AsyncMock,
                return_value=LiteLLM_BudgetTable(rpm_limit=5, tpm_limit=1000),
            ),
        ):
            result = await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {api_key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={"user": "alice@example.com"},
            )

        # Fix: limits must be stamped even when the key came from cache.
        # Without the unconditional update_valid_token_with_end_user_params call,
        # end_user_rpm_limit stays None and parallel_request_limiter_v3 never
        # creates a rate-limit bucket — all requests succeed regardless of limit.
        assert result.end_user_rpm_limit == 5, (
            "end_user_rpm_limit must be non-None on cache-hit requests so "
            "parallel_request_limiter_v3 enforces the customer rate limit"
        )
        assert result.end_user_tpm_limit == 1000
        assert result.end_user_id == "alice@example.com"

    finally:
        for k, v in _originals.items():
            setattr(_proxy_server_mod, k, v)
        _litellm.max_end_user_budget_id = _original_max_end_user_budget_id


def test_end_user_rpm_limit_applied_on_cache_hit():
    """
    Regression: end_user_rpm_limit / end_user_tpm_limit were only written onto
    the token inside the DB-miss branch of _user_api_key_auth_builder via the
    raw .get() assignments. On a cache hit those raw assignments run but they
    read from end_user_params which is built from get_end_user_object — the
    bug is that a cached token whose end_user_rpm_limit was previously None
    is never updated via update_valid_token_with_end_user_params unless the
    unconditional call we added is present.

    This test directly exercises update_valid_token_with_end_user_params to
    confirm:
    1. Without calling it, a cached token retains None for rpm/tpm limits.
    2. After calling it with populated end_user_params, the limits are set.

    This matches exactly what the fix does: call it unconditionally so that
    cache-hit requests (where the token object in memory has stale None values)
    get the correct limits stamped before reaching parallel_request_limiter_v3.
    """
    # Simulate a token as it sits in the in-memory cache — no end_user fields
    cached_token = UserAPIKeyAuth(
        token="hashed-sk-test",
        end_user_rpm_limit=None,
        end_user_tpm_limit=None,
        end_user_id=None,
    )

    # end_user_params as built by the early get_end_user_object call
    end_user_params_with_limits = {
        "end_user_id": "alice@example.com",
        "end_user_rpm_limit": 5,
        "end_user_tpm_limit": 1000,
    }

    # --- WITHOUT the fix: cache hit skips the unconditional call ---
    # Token in cache still has None — parallel_request_limiter_v3 skips bucket
    assert cached_token.end_user_rpm_limit is None, (
        "Precondition: cached token starts with no rpm limit"
    )

    # --- WITH the fix: unconditional call stamps the limits ---
    result = update_valid_token_with_end_user_params(
        cached_token, end_user_params_with_limits
    )

    assert result.end_user_rpm_limit == 5, (
        "end_user_rpm_limit must be stamped onto the token so "
        "parallel_request_limiter_v3 creates a rate-limit bucket"
    )
    assert result.end_user_tpm_limit == 1000
    assert result.end_user_id == "alice@example.com"


def test_end_user_rpm_limit_none_without_unconditional_update():
    """
    Demonstrates the pre-fix state: a cached token whose rpm/tpm limits are
    None stays None if update_valid_token_with_end_user_params is never called.
    This is what happened on every cache-hit request before the fix.
    """
    cached_token = UserAPIKeyAuth(
        token="hashed-sk-test",
        end_user_rpm_limit=None,
        end_user_tpm_limit=None,
    )

    # Simulate the pre-fix cache-hit path: update function is NOT called
    # (the DB-miss block's raw assignments also don't run on a cache hit)
    # So the token retains None — no bucket, no 429.
    assert cached_token.end_user_rpm_limit is None
    assert cached_token.end_user_tpm_limit is None

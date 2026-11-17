from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.routing import Route

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import (
    _run_post_custom_auth_checks,
    update_valid_token_with_end_user_params,
)
from litellm.proxy.hooks.model_max_budget_limiter import (
    PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
)
from litellm.types.videos.utils import encode_video_id_with_provider


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_without_end_user_id():
    # common_checks now runs in the user_api_key_auth wrapper via
    # _run_centralized_common_checks, gated by
    # custom_auth_run_common_checks for custom-auth deployments. The
    # helper itself no longer calls common_checks.
    valid_token = UserAPIKeyAuth(token="test_token")
    http_request: Final = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions"})

    # Default: common_checks should NOT be called inside the helper
    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ) as mock_common:
        mock_common.return_value = True
        result = await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=http_request,
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
            request=http_request,
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
    http_request: Final = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions"})
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
            "litellm.proxy.auth.user_api_key_auth.check_end_user_budget",
            new_callable=AsyncMock,
        ) as mock_check,
        patch(
            "litellm.proxy.auth.user_api_key_auth.enforce_key_and_fallback_model_access",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=http_request,
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


@pytest.mark.asyncio
async def test_custom_auth_key_default_end_user_budget_reaches_the_token_for_a_new_end_user(monkeypatch):
    """A custom-auth token that carries a key ``end_user_budget_id`` must enforce that budget on a
    brand-new end user, ahead of the proxy-wide default, from the very first request."""
    from unittest.mock import MagicMock

    from litellm.proxy.auth.user_api_key_auth import _lookup_end_user_and_apply_budget
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    monkeypatch.setattr(litellm, "max_end_user_budget_id", "global-eu-budget")
    budgets = {"global-eu-budget": 100.0, "svc-a-budget": 0.5}

    async def _find_budget(where):
        row = MagicMock()
        row.dict = lambda: {"budget_id": where["budget_id"], "max_budget": budgets[where["budget_id"]]}
        return row

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_endusertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_budgettable.find_unique = AsyncMock(side_effect=_find_budget)

    valid_token, end_user_object = await _lookup_end_user_and_apply_budget(
        valid_token=UserAPIKeyAuth(
            token="test_token",
            end_user_id="customer-new",
            metadata={"end_user_budget_id": "svc-a-budget"},
        ),
        route="/v1/chat/completions",
        parent_otel_span=None,
        prisma_client=mock_prisma,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=MagicMock(),
    )

    assert end_user_object is None
    assert valid_token.end_user_max_budget == 0.5


@pytest.mark.asyncio
async def test_custom_auth_cap_stays_below_the_key_default_end_user_budget(monkeypatch):
    """A custom auth callable that already capped the end user tighter than the key's default
    budget keeps its cap: the key default never loosens what custom auth set."""
    from unittest.mock import MagicMock

    from litellm.proxy.auth.user_api_key_auth import _lookup_end_user_and_apply_budget
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    monkeypatch.setattr(litellm, "max_end_user_budget_id", None)

    async def _find_budget(where):
        row = MagicMock()
        row.dict = lambda: {"budget_id": where["budget_id"], "max_budget": 0.5}
        return row

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_endusertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_budgettable.find_unique = AsyncMock(side_effect=_find_budget)

    valid_token, _ = await _lookup_end_user_and_apply_budget(
        valid_token=UserAPIKeyAuth(
            token="test_token",
            end_user_id="customer-new",
            end_user_max_budget=0.1,
            metadata={"end_user_budget_id": "svc-a-budget"},
        ),
        route="/v1/chat/completions",
        parent_otel_span=None,
        prisma_client=mock_prisma,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=MagicMock(),
    )

    assert valid_token.end_user_max_budget == 0.1


@pytest.mark.asyncio
async def test_custom_auth_proxy_wide_default_end_user_budget_reaches_an_uncapped_token(monkeypatch):
    """With no key default, a brand-new end user on a custom-auth token that set no cap gets the
    proxy-wide default budget's cap, the same way the virtual-key path already applies it."""
    from unittest.mock import MagicMock

    from litellm.proxy.auth.user_api_key_auth import _lookup_end_user_and_apply_budget
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    monkeypatch.setattr(litellm, "max_end_user_budget_id", "global-eu-budget")

    async def _find_budget(where):
        row = MagicMock()
        row.dict = lambda: {"budget_id": where["budget_id"], "max_budget": 100.0}
        return row

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_endusertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_budgettable.find_unique = AsyncMock(side_effect=_find_budget)

    valid_token, end_user_object = await _lookup_end_user_and_apply_budget(
        valid_token=UserAPIKeyAuth(token="test_token", end_user_id="customer-new"),
        route="/v1/chat/completions",
        parent_otel_span=None,
        prisma_client=mock_prisma,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=MagicMock(),
    )

    assert end_user_object is None
    assert valid_token.end_user_max_budget == 100.0


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


def test_end_user_budget_tpd_limit_reaches_the_token():
    from litellm.proxy.auth.user_api_key_auth import _apply_budget_limits_to_end_user_params

    end_user_params = {"end_user_id": "user_1"}
    _apply_budget_limits_to_end_user_params(
        end_user_params=end_user_params,
        budget_info=LiteLLM_BudgetTable(rpm_limit=5, tpd_limit=750000),
        end_user_id="user_1",
    )
    result = update_valid_token_with_end_user_params(UserAPIKeyAuth(token="test_token"), end_user_params)

    assert result.end_user_rpm_limit == 5
    assert result.end_user_tpd_limit == 750000


def test_update_valid_token_loads_per_model_tpm_and_rpm_limits_from_the_customer_budget():
    valid_token: Final = UserAPIKeyAuth(token="test_token")
    end_user_params: Final[dict[str, object]] = {
        "end_user_id": "acme",
        "end_user_model_tpm_limit": {"gpt-4o": 400000},
        "end_user_model_rpm_limit": {"gpt-4o": 200},
    }

    result: Final = update_valid_token_with_end_user_params(valid_token, end_user_params)

    assert result.end_user_model_tpm_limit == {"gpt-4o": 400000}
    assert result.end_user_model_rpm_limit == {"gpt-4o": 200}


def test_update_valid_token_keeps_custom_auth_per_model_limits_when_the_budget_has_none():
    valid_token: Final = UserAPIKeyAuth(
        token="test_token",
        end_user_id="acme",
        end_user_model_tpm_limit={"gpt-4o": 1000},
        end_user_model_rpm_limit={"gpt-4o": 10},
    )

    result: Final = update_valid_token_with_end_user_params(valid_token, {"end_user_id": "acme"})

    assert result.end_user_model_tpm_limit == {"gpt-4o": 1000}
    assert result.end_user_model_rpm_limit == {"gpt-4o": 10}


def test_apply_budget_limits_copies_per_model_limits_onto_end_user_params():
    from litellm.proxy.auth.user_api_key_auth import _apply_budget_limits_to_end_user_params

    end_user_params: Final[dict[str, object]] = {}
    budget: Final = LiteLLM_BudgetTable(model_tpm_limit={"gpt-4o": 400000}, model_rpm_limit={"gpt-4o": 200})

    _apply_budget_limits_to_end_user_params(end_user_params, budget, "acme")

    assert end_user_params == {
        "end_user_model_tpm_limit": {"gpt-4o": 400000},
        "end_user_model_rpm_limit": {"gpt-4o": 200},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, key_aliases, run_common_checks, expect_enforced",
    [
        ("free-key", {"free-key": "free-model"}, True, False),
        ("free-model", {"free-model": "paid-model"}, True, True),
        ("free-key", {"free-key": "free-model"}, False, True),
    ],
)
async def test_custom_auth_judges_the_zero_cost_waiver_by_the_key_alias_target(
    requested_model: str,
    key_aliases: dict[str, str],
    run_common_checks: bool,
    expect_enforced: bool,
    free_and_paid_router: litellm.Router,
) -> None:
    valid_token: Final = UserAPIKeyAuth(
        token="test_token",
        aliases=key_aliases,
        end_user_id="end-user",
        end_user_model_max_budget={requested_model: {"budget_limit": 1.0, "time_period": "1d"}},
    )
    limiter: Final = MagicMock()
    limiter.is_end_user_within_model_budget = AsyncMock(
        side_effect=litellm.BudgetExceededError(current_cost=5.0, max_budget=1.0)
    )
    http_request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "route": Route("/v1/chat/completions", endpoint=lambda request: None),
        }
    )

    async def run() -> UserAPIKeyAuth:
        return await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=http_request,
            request_data={"model": requested_model},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )

    with (
        patch("litellm.proxy.proxy_server.llm_router", free_and_paid_router),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
        patch("litellm.proxy.proxy_server.general_settings", {"custom_auth_run_common_checks": run_common_checks}),
    ):
        if expect_enforced:
            with pytest.raises(litellm.BudgetExceededError):
                await run()
            assert limiter.is_end_user_within_model_budget.await_args.kwargs["model"] == requested_model
        else:
            result: Final = await run()
            assert result.token == "test_token"
            limiter.is_end_user_within_model_budget.assert_not_awaited()


_VIDEO_ID: Final = encode_video_id_with_provider(video_id="video_raw", provider="openai", model_id="sora-2")


async def _post_custom_auth_video_request(
    valid_token: UserAPIKeyAuth, method: str, template: str, request_data: dict[str, object]
) -> UserAPIKeyAuth:
    limiter: Final = PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    await limiter.dual_cache.async_set_cache(
        key=model_budget_spend_cache_key(
            entity_type=Litellm_EntityType.KEY, entity_id=valid_token.token, budget_model="sora-2", budget_duration="1d"
        ),
        value=0.4,
    )
    path: Final = template.format(video_id=_VIDEO_ID)
    http_request: Final = Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "query_string": b"",
            "route": Route(template, endpoint=lambda request: None),
        }
    )
    with (
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        return await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=http_request,
            request_data=request_data,
            route=path,
            parent_otel_span=None,
        )


def _exhausted_model_budget_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token="test_token", model_max_budget={"sora-2": {"max_budget": 0.3, "budget_duration": "1d"}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template",
    ["/v1/videos/{video_id}", "/videos/{video_id}", "/v1/videos/{video_id}/content", "/videos/{video_id}/content"],
)
async def test_custom_auth_exhausted_model_budget_still_retrieves_an_already_paid_video(template: str) -> None:
    result: Final = await _post_custom_auth_video_request(
        _exhausted_model_budget_token(), "GET", template, {"video_id": _VIDEO_ID}
    )

    assert result.token == "test_token"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, template, request_data",
    [
        ("POST", "/v1/videos", {"model": "sora-2"}),
        ("POST", "/v1/videos/{video_id}/remix", {"video_id": _VIDEO_ID}),
        ("GET", "/v1/videos", {"model": "sora-2"}),
    ],
)
async def test_custom_auth_exhausted_model_budget_still_blocks_video_requests_other_than_retrieval(
    method: str, template: str, request_data: dict[str, object]
) -> None:
    with pytest.raises(litellm.BudgetExceededError):
        await _post_custom_auth_video_request(_exhausted_model_budget_token(), method, template, request_data)


@pytest.mark.asyncio
async def test_custom_auth_video_retrieval_still_rejects_an_expired_token() -> None:
    expired_token: Final = _exhausted_model_budget_token().model_copy(
        update={"expires": datetime(2000, 1, 1, tzinfo=timezone.utc)}
    )

    with pytest.raises(ProxyException) as exc_info:
        await _post_custom_auth_video_request(expired_token, "GET", "/v1/videos/{video_id}", {"video_id": _VIDEO_ID})

    assert exc_info.value.type == ProxyErrorTypes.expired_key

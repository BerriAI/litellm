from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm import Router
from litellm.proxy.lens.endpoints import list_agents, user_scope, validate_model, worker_supports_model
from litellm.proxy.lens.models import LensSettings


@pytest.fixture
def analysis_router(monkeypatch: pytest.MonkeyPatch) -> Router:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost})
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
            {
                "model_name": "analysis",
                "litellm_params": {
                    "model": "openai/test-analysis",
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
            {"model_name": "unpriced/*", "litellm_params": {"model": "openai/*", "api_key": "test-key"}},
        ],
        model_group_alias={"analysis-alias": "analysis"},
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    return router


@pytest.mark.parametrize("model", ("openai/test-analysis", "analysis", "analysis-alias"))
@pytest.mark.asyncio
async def test_analysis_accepts_models_served_by_configured_routes(analysis_router: Router, model: str) -> None:
    settings: Final = LensSettings(name="Research", model=model, context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert analysis_router.get_model_list(model_name=model)
    await validate_model(settings, auth)


@pytest.mark.parametrize("model", ("unconfigured", "anthropic/test-analysis"))
@pytest.mark.asyncio
async def test_analysis_rejects_models_without_a_configured_route(analysis_router: Router, model: str) -> None:
    settings: Final = LensSettings(name="Research", model=model, context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert not analysis_router.get_model_list(model_name=model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_analysis_route_resolution_preserves_key_model_restrictions(analysis_router: Router) -> None:
    settings: Final = LensSettings(name="Research", model="openai/test-analysis", context="Answer using cited sources")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, models=["analysis"])
    assert analysis_router.get_model_list(model_name=settings.model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_analysis_rejects_unpriced_wildcard_before_creating_a_run(analysis_router: Router) -> None:
    settings: Final = LensSettings(name="Research", model="unpriced/lens-unpriced-test", context="Answer questions")
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert analysis_router.get_model_list(model_name=settings.model)
    with pytest.raises(HTTPException) as error:
        await validate_model(settings, auth)
    assert error.value.status_code == 400
    assert "Pricing is not configured" in error.value.detail


@pytest.mark.parametrize("model,allowed", (("openai/test-analysis", "openai/*"), ("analysis-alias", "analysis")))
@pytest.mark.asyncio
async def test_analysis_key_accepts_wildcard_and_alias_access(
    analysis_router: Router, model: str, allowed: str
) -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, models=[allowed])
    assert analysis_router.get_model_list(model_name=model)
    await validate_model(LensSettings(name="Research", model=model, context="Answer questions"), auth)


@pytest.mark.parametrize("revoked,key_id", ((True, "a" * 64), (False, None)))
@pytest.mark.asyncio
async def test_worker_without_active_billing_cannot_take_work(revoked: bool, key_id: str | None) -> None:
    from tests.unit.proxy.lens.test_state import worker

    inactive: Final = worker().model_copy(update={"revoked": revoked, "analysis_key_id": key_id})
    settings: Final = LensSettings(name="Research", model="analysis", context="Answer questions")
    assert not await worker_supports_model(inactive, settings)


@pytest.mark.parametrize("role", (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_is_empty(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role)
    assert await list_agents(auth, None) == ()


@pytest.mark.asyncio
async def test_agent_discovery_without_trace_storage_still_requires_admin_access() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    with pytest.raises(HTTPException) as error:
        await list_agents(auth, None)
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "role",
    (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.TEAM),
)
def test_non_admin_cannot_start_analysis_spending(role: LitellmUserRoles) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth, write=True)
    assert error.value.status_code == 403


def test_admin_can_configure_lens_and_viewer_can_only_read() -> None:
    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    viewer: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    assert user_scope(admin, write=True).all_teams
    assert user_scope(viewer).all_teams


@pytest.mark.parametrize("identity", ("not-an-execution", "W10=", "WyJvdGhlciIsICIiLCAiaWQiXQ=="))
def test_invalid_explicit_execution_ids_are_rejected(identity: str) -> None:
    from litellm.proxy.lens.endpoints import validate_selection
    from tests.unit.proxy.lens.test_state import lens

    settings: Final = lens().settings.model_copy(update={"execution_ids": (identity,)})
    with pytest.raises(HTTPException) as error:
        validate_selection(settings)
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_incompatible_worker_is_rejected_before_claiming_work() -> None:
    from litellm.proxy.lens.endpoints import claim
    from tests.unit.proxy.lens.test_state import worker

    with pytest.raises(HTTPException) as error:
        await claim(worker(), protocol_version=1)
    assert error.value.status_code == 409
    assert "Upgrade" in error.value.detail


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, None))
def test_regular_keys_cannot_read_lens_results(role: LitellmUserRoles | None) -> None:
    auth: Final = UserAPIKeyAuth(user_role=role, team_id="team", token="hashed-test-key")
    with pytest.raises(HTTPException) as error:
        user_scope(auth)
    assert error.value.status_code == 403


@pytest.mark.parametrize("provider", (False, True))
def test_model_errors_reach_worker_with_status_and_redacted_provider_message(provider: bool) -> None:
    import httpx

    from litellm.proxy._types import ProxyException
    from litellm.proxy.lens.endpoints import model_failure
    from litellm.proxy.lens.worker import failure_message

    message: Final = "Token rate limit exceeded. api_key=secret-example-value-123456 Retry in 60 seconds."
    error: Final = model_failure(
        ProxyException(message, "rate_limit_error", None, 429, headers={"retry-after": "60"})
        if provider
        else HTTPException(429, message, headers={"retry-after": "60"})
    )
    request: Final = httpx.Request("POST", "https://proxy.test/lens/worker/lens/run/model")
    response: Final = httpx.Response(error.status_code, json={"detail": error.detail}, request=request)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    diagnostic: Final = failure_message(caught.value)
    assert diagnostic.startswith("Model request failed (HTTP 429):")
    assert "Token rate limit exceeded." in diagnostic
    assert "Retry in 60 seconds." in diagnostic
    assert "secret-example" not in diagnostic
    assert error.headers == {"retry-after": "60"}


@pytest.mark.asyncio
async def test_preview_reports_calendar_overflow_as_a_validation_error() -> None:
    from datetime import datetime, timezone

    from litellm.proxy.lens.endpoints import Preview, preview_sample

    body: Final = Preview(
        settings=LensSettings(name="Calendar regression", model="analysis", context="Read recorded activity"),
        as_of=datetime.min.replace(tzinfo=timezone.utc),
    )
    with pytest.raises(HTTPException) as error:
        await preview_sample(body, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), None)
    assert error.value.status_code == 422
    assert "supported calendar range" in error.value.detail

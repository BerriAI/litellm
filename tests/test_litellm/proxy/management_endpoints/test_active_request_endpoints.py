from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Response

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.hooks.active_request_registry import ActiveRequestRegistry
from litellm.proxy.management_endpoints import active_request_endpoints


class FakeRegistry:
    def __init__(self):
        self.kwargs = None

    async def list_requests(self, **kwargs):
        self.kwargs = kwargs
        return {
            "available": True,
            "reason": None,
            "items": [
                {
                    "registry_id": "reg-1", "request_id": "request-1",
                    "started_at": 100.0,
                    "model": "model-a",
                    "end_user_id": "end-user-1",
                }
            ],
            "total": 1,
            "page": kwargs["page"],
            "page_size": kwargs["page_size"],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
async def test_admin_roles_can_list_active_requests_without_caching(monkeypatch, role):
    registry = FakeRegistry()
    monkeypatch.setattr(
        active_request_endpoints,
        "_get_active_request_registry",
        lambda: registry,
    )
    response = Response()

    result = await active_request_endpoints.get_active_requests(
        response=response,
        model="model-a",
        end_user_id="end-user-1",
        page=2,
        page_size=25,
        user_api_key_dict=UserAPIKeyAuth(user_role=role),
    )

    assert result.total == 1
    assert result.items[0].end_user_id == "end-user-1"
    assert isinstance(result.generated_at, datetime)
    assert response.headers["Cache-Control"] == "no-store, max-age=0"
    assert response.headers["Pragma"] == "no-cache"
    assert registry.kwargs == {
        "model": "model-a",
        "user_id": None,
        "end_user_id": "end-user-1",
        "organization_id": None,
        "project_id": None,
        "page": 2,
        "page_size": 25,
    }


@pytest.mark.asyncio
async def test_non_admin_cannot_list_active_requests(monkeypatch):
    get_registry = SimpleNamespace(called=False)

    def unexpected_registry_lookup():
        get_registry.called = True

    monkeypatch.setattr(
        active_request_endpoints,
        "_get_active_request_registry",
        unexpected_registry_lookup,
    )

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.get_active_requests(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        )

    assert exc_info.value.status_code == 403
    assert get_registry.called is False


@pytest.mark.asyncio
async def test_missing_registry_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(
        active_request_endpoints,
        "_get_active_request_registry",
        lambda: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.get_active_requests(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_registry_failure_is_reported_as_service_unavailable(monkeypatch):
    class ExplodingRegistry:
        async def list_requests(self, **_):
            raise ConnectionError("redis down")

    monkeypatch.setattr(active_request_endpoints, "_get_active_request_registry", lambda: ExplodingRegistry())

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.get_active_requests(
            response=Response(),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.status_code == 503
    assert "active request registry" in exc_info.value.detail


def test_registry_lookup_ignores_an_unrelated_hook(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        SimpleNamespace(get_proxy_hook=lambda _: object()),
        raising=False,
    )

    assert active_request_endpoints._get_active_request_registry() is None


def test_registry_lookup_returns_the_registered_hook(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    registry = ActiveRequestRegistry(SimpleNamespace(dual_cache=SimpleNamespace(redis_cache=None)))
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        SimpleNamespace(get_proxy_hook=lambda _: registry),
        raising=False,
    )

    assert active_request_endpoints._get_active_request_registry() is registry


@pytest.mark.asyncio
async def test_only_proxy_admin_may_cancel_a_request(monkeypatch):
    cancelled = []

    class Registry:
        async def request_cancel(self, registry_id):
            cancelled.append(registry_id)
            return True

    monkeypatch.setattr(active_request_endpoints, "_get_active_request_registry", lambda: Registry())

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.cancel_active_request(
            registry_id="abc",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        )

    assert exc_info.value.status_code == 403
    assert cancelled == []


@pytest.mark.asyncio
async def test_cancelling_a_running_request_reports_the_signal_was_sent(monkeypatch):
    class Registry:
        async def request_cancel(self, registry_id):
            assert registry_id == "abc"
            return True

    monkeypatch.setattr(active_request_endpoints, "_get_active_request_registry", lambda: Registry())

    result = await active_request_endpoints.cancel_active_request(
        registry_id="abc",
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert result.cancelled is True


@pytest.mark.asyncio
async def test_cancelling_a_finished_request_is_not_found(monkeypatch):
    class Registry:
        async def request_cancel(self, registry_id):
            return False

    monkeypatch.setattr(active_request_endpoints, "_get_active_request_registry", lambda: Registry())

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.cancel_active_request(
            registry_id="gone",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cancelling_without_a_registry_is_service_unavailable(monkeypatch):
    monkeypatch.setattr(active_request_endpoints, "_get_active_request_registry", lambda: None)

    with pytest.raises(HTTPException) as exc_info:
        await active_request_endpoints.cancel_active_request(
            registry_id="abc",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert exc_info.value.status_code == 503

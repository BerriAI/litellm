"""Tests for catalog-only entries on GET /v1/models.

`general_settings.advertised_models` adds entries to the listing for models the
proxy does not serve itself. They are discovery only: no deployment is
registered, so routing is untouched and a routed model is never displaced.
"""

from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth

CATALOG_SETTINGS: Final = {"advertised_models": [{"id": "catalog-only-model", "owned_by": "example-provider"}]}


@pytest.fixture
def patched_model_list(monkeypatch):
    """Stub router + utility helpers used by `model_list`."""
    from litellm.proxy import utils as proxy_utils

    router: Final = MagicMock()
    router.get_fully_blocked_model_names = MagicMock(return_value=set())
    router.async_get_fully_unhealthy_model_names = AsyncMock(return_value=set())
    router.get_model_names = MagicMock(return_value=["routed-model"])
    router.get_model_access_groups = MagicMock(return_value={})

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})

    async def _fake_get_available_models_for_user(**kwargs):
        return ["routed-model"]

    monkeypatch.setattr(proxy_utils, "get_available_models_for_user", _fake_get_available_models_for_user)

    def _fake_create_model_info_response(model_id, provider="openai", **kwargs):
        return {"id": model_id, "object": "model", "created": 0, "owned_by": provider}

    monkeypatch.setattr(proxy_utils, "create_model_info_response", _fake_create_model_info_response)

    return router


async def _listing(**kwargs):
    response: Final = await proxy_server.model_list(user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"), **kwargs)
    return response["data"]


@pytest.mark.asyncio
async def test_catalog_entry_is_listed_alongside_routed_models(patched_model_list, monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", CATALOG_SETTINGS)

    rows: Final = await _listing()

    assert [row["id"] for row in rows] == [
        "routed-model",
        "catalog-only-model",
    ], f"the catalog entry should be listed after the routed models, got {rows}"
    assert rows[1]["owned_by"] == "example-provider", f"the configured owner should be reported, got {rows[1]}"


@pytest.mark.asyncio
async def test_routed_rows_are_untouched_by_the_catalog(patched_model_list, monkeypatch):
    """The routed model's row is identical with and without catalog entries."""
    without_catalog: Final = await _listing()

    monkeypatch.setattr(proxy_server, "general_settings", CATALOG_SETTINGS)
    with_catalog: Final = await _listing()

    assert with_catalog[0] == without_catalog[0], (
        "configuring advertised_models must not change an existing row: "
        f"{with_catalog[0]} differs from {without_catalog[0]}"
    )


@pytest.mark.asyncio
async def test_listing_is_unchanged_when_no_catalog_is_configured(patched_model_list):
    rows: Final = await _listing()

    assert rows == [{"id": "routed-model", "object": "model", "created": 0, "owned_by": "openai"}], (
        f"with no advertised_models the listing must be exactly what it was before, got {rows}"
    )


@pytest.mark.asyncio
async def test_catalog_entry_cannot_displace_a_routed_model_of_the_same_id(patched_model_list, monkeypatch):
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"advertised_models": [{"id": "routed-model", "owned_by": "impostor"}]},
    )

    rows: Final = await _listing()

    assert rows == [{"id": "routed-model", "object": "model", "created": 0, "owned_by": "openai"}], (
        f"the routed model must be listed once, unchanged, got {rows}"
    )


@pytest.mark.asyncio
async def test_access_group_only_listing_carries_no_catalog_entries(patched_model_list, monkeypatch):
    """A caller asking for access groups alone gets none: a catalog entry is not a group."""
    monkeypatch.setattr(proxy_server, "general_settings", CATALOG_SETTINGS)

    rows: Final = await _listing(only_model_access_groups=True)

    assert "catalog-only-model" not in [row["id"] for row in rows], (
        f"only_model_access_groups must not surface catalog entries, got {rows}"
    )


@pytest.mark.asyncio
async def test_catalog_entry_is_listed_for_scope_expand(patched_model_list, monkeypatch):
    from litellm.proxy.auth import model_checks
    from litellm.proxy.management_endpoints import common_utils

    async def _fake_admin(**kwargs):
        return True

    monkeypatch.setattr(common_utils, "_user_has_admin_privileges", _fake_admin)
    monkeypatch.setattr(model_checks, "get_complete_model_list", lambda **kwargs: ["routed-model"])
    monkeypatch.setattr(proxy_server, "general_settings", CATALOG_SETTINGS)

    rows: Final = await _listing(scope="expand")

    assert [row["id"] for row in rows] == [
        "routed-model",
        "catalog-only-model",
    ], f"the admin listing should carry catalog entries too, got {rows}"

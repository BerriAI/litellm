"""
Tests for the pass-through models GET /v1/models?scope=expand offers the Admin UI
allowlist pickers: with TypeSafe credentials configured, `typesafe/*` and every
priced Jev model are offered, so a team or project can allowlist them.
"""

from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


@pytest.fixture
def openai_only_router(monkeypatch) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                "model_info": {"id": "gpt-4-id"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    return router


def _priced_jev_models() -> frozenset[str]:
    return frozenset(name.removeprefix("typesafe/") for name in litellm.model_cost if name.startswith("typesafe/"))


async def _admin_picker_models() -> set[str]:
    admin: Final = UserAPIKeyAuth(api_key="sk-test", user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN)
    response: Final = await proxy_server.model_list(
        user_api_key_dict=admin, return_wildcard_routes=True, include_model_access_groups=True, scope="expand"
    )
    return {model["id"] for model in response["data"]}


@pytest.mark.asyncio
async def test_scope_expand_offers_typesafe_models_when_typesafe_is_configured(openai_only_router, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-typesafe-fake")
    jev_models: Final = _priced_jev_models()

    picker_models: Final = await _admin_picker_models()

    assert jev_models
    assert {"gpt-4", "typesafe/*", *jev_models} <= picker_models


@pytest.mark.asyncio
async def test_scope_expand_offers_no_typesafe_models_without_typesafe_credentials(openai_only_router, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    picker_models: Final = await _admin_picker_models()

    assert "gpt-4" in picker_models
    assert picker_models.isdisjoint({"typesafe/*", *_priced_jev_models()})


@pytest.mark.asyncio
async def test_default_model_list_does_not_offer_typesafe_models(openai_only_router, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-typesafe-fake")
    admin: Final = UserAPIKeyAuth(api_key="sk-test", user_id="u", user_role=LitellmUserRoles.PROXY_ADMIN)

    response: Final = await proxy_server.model_list(user_api_key_dict=admin)

    assert {model["id"] for model in response["data"]}.isdisjoint({"typesafe/*", *_priced_jev_models()})

import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from starlette.requests import Request as StarletteRequest

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import (
    LitellmUserRoles,
    NewCustomerRequest,
    ProxyException,
    UpdateCustomerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import (
    update_valid_token_with_end_user_params,  # pyright: ignore[reportUnknownVariableType]  # legacy auth helper accepts an untyped parameter dict
    user_api_key_auth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

RULES: Final = [{"primary": ["backup", "last"]}]


@pytest.mark.parametrize(
    "fallbacks",
    (
        [{"primary": ["primary"]}],
        [{"primary": ["backup"]}, {"backup": ["primary"]}],
        [{"primary": ["backup"]}, {"primary": ["last"]}],
        [{"primary": ["backup", "backup"]}],
        [{"primary": []}],
        [{"primary": [" "]}],
        [{" primary": ["backup"]}],
        [{"primary": ["backup"], "other": ["last"]}],
        ["backup"],
        [{"primary": [{"model": "backup", "api_key": "client-controlled"}]}],
    ),
)
def test_customer_fallback_rules_reject_ambiguous_or_cyclic_chains(
    fallbacks: object,
) -> None:
    for request in (NewCustomerRequest, UpdateCustomerRequest):
        with pytest.raises(ValidationError):
            request.model_validate({"user_id": "customer", "fallbacks": fallbacks})


@pytest.mark.parametrize("fallbacks", (None, [], RULES))
def test_customer_fallback_updates_preserve_the_difference_between_inheritance_and_disable(
    fallbacks: object,
) -> None:
    request: Final = UpdateCustomerRequest.model_validate({"user_id": "customer", "fallbacks": fallbacks})
    assert request.model_dump(exclude_unset=True) == {
        "user_id": "customer",
        "fallbacks": fallbacks,
    }
    assert UpdateCustomerRequest(user_id="customer").model_dump(exclude_unset=True) == {"user_id": "customer"}


def test_reusing_a_key_for_another_customer_does_not_reuse_previous_fallback_rules() -> None:
    token: Final = UserAPIKeyAuth(end_user_id="first", end_user_fallbacks=RULES)
    updated: Final = update_valid_token_with_end_user_params(token, {"end_user_id": "second"})
    assert (updated.end_user_id, updated.end_user_fallbacks) == ("second", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("builder_only", (False, True))
async def test_cached_virtual_key_loads_each_customers_saved_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
    builder_only: bool,
) -> None:
    from litellm.proxy._types import LiteLLM_EndUserTable
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder
    from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key

    server: Final = proxy_server
    cache: Final = UserApiKeyCache()
    key: Final = "sk-customer-policy-cache-test"
    stored: Final = UserAPIKeyAuth(
        token=server.hash_token(key),
        models=["primary-model"],
        user_role=LitellmUserRoles.PROXY_ADMIN if builder_only else LitellmUserRoles.INTERNAL_USER,
    )
    first_rules: Final = [{"primary-model": ["first-backup"]}]
    second_rules: Final = [{"primary-model": ["second-backup"]}]
    cache.set_cache(server.hash_token(key), stored, model_type=UserAPIKeyAuth)
    for customer, rules in (("first-customer", first_rules), ("second-customer", second_rules), ("unconfigured", None)):
        cache.set_cache(
            end_user_cache_key(customer),
            LiteLLM_EndUserTable(user_id=customer, blocked=False, fallbacks=rules),
            model_type=LiteLLM_EndUserTable,
        )
    monkeypatch.setattr(server, "master_key", "sk-unit")
    monkeypatch.setattr(server, "general_settings", {})
    monkeypatch.setattr(server, "user_api_key_cache", cache)
    monkeypatch.setattr(server, "prisma_client", MagicMock())
    monkeypatch.setattr(server, "llm_router", None)
    monkeypatch.setattr(server, "user_custom_auth", None)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", None)
    monkeypatch.setattr(litellm, "validate_end_user_id_in_db", False)

    def customer_request(customer: str | None, body: dict[str, object]) -> StarletteRequest:
        return StarletteRequest(
            {
                "type": "http",
                "method": "POST",
                "path": "/chat/completions",
                "headers": [(b"x-litellm-customer-id", customer.encode())] if customer is not None else [],
            },
            receive=AsyncMock(
                return_value={"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
            ),
        )

    client_fallbacks: Final = [{"primary-model": ["client-model-without-access"]}]
    for customer, expected, with_client_overrides in (
        ("first-customer", first_rules, True),
        ("second-customer", second_rules, True),
        ("unconfigured", None, False),
        (None, None, False),
        ("first-customer", first_rules, False),
    ):
        body: Final[dict[str, object]] = {
            "model": "primary-model",
            **(
                {"fallbacks": client_fallbacks, "router_settings_override": {"fallbacks": client_fallbacks}}
                if with_client_overrides
                else {}
            ),
        }
        request: Final = customer_request(customer, body)
        identity: Final = (
            await user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data=body,
            )
            if builder_only
            else await user_api_key_auth(request=request, api_key=f"Bearer {key}")
        )
        assert (identity.end_user_id, identity.end_user_fallbacks) == (customer, expected)

    if not builder_only:
        for customer in ("unconfigured", None):
            denied_request: Final = customer_request(
                customer,
                {"model": "primary-model", "fallbacks": client_fallbacks},
            )
            with pytest.raises(ProxyException) as denied:
                await user_api_key_auth(request=denied_request, api_key=f"Bearer {key}")
            assert denied.value.code == "403"

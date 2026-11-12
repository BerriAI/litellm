from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy._types import (
    NewCustomerRequest,
    ProxyException,
    UpdateCustomerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import (
    update_valid_token_with_end_user_params,  # pyright: ignore[reportUnknownVariableType]  # legacy auth helper accepts an untyped parameter dict
)
from litellm.proxy.common_utils.customer_fallbacks import (
    apply_customer_fallbacks,
    customer_fallback_settings,
    prepare_customer_fallback_request,
)

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


def test_customer_fallback_policy_replaces_every_error_specific_fallback_chain() -> None:
    data: Final[dict[str, object]] = {
        "model": "primary",
        "fallbacks": ["client-backup"],
        "context_window_fallbacks": [{"primary": ["client-context"]}],
        "content_policy_fallbacks": [{"primary": ["client-content"]}],
        "router_settings_override": {
            "fallbacks": [{"primary": ["client-override"]}],
            "timeout": 9,
        },
    }
    assert {**data, **apply_customer_fallbacks(data, RULES)} == {
        "model": "primary",
        "fallbacks": RULES,
        "context_window_fallbacks": RULES,
        "content_policy_fallbacks": RULES,
        "router_settings_override": {"timeout": 9},
    }


def test_auth_discards_client_fallbacks_without_treating_server_rules_as_client_allowlist_requests() -> None:
    data: Final[dict[str, object]] = {
        "model": "primary",
        "fallbacks": ["client-backup"],
        "max_tokens": 7,
    }
    assert {**data, **prepare_customer_fallback_request(data, RULES)} == {
        "model": "primary",
        "fallbacks": None,
        "max_tokens": 7,
    }


def test_unconfigured_customer_preserves_existing_request_and_routing_settings() -> None:
    data: Final[dict[str, object]] = {"model": "primary", "fallbacks": ["existing"]}
    assert {**data, **apply_customer_fallbacks(data, None)} == {"model": "primary", "fallbacks": ["existing"]}
    assert customer_fallback_settings(None) == {}


def test_explicit_empty_customer_fallbacks_disable_every_fallback_chain() -> None:
    data: Final[dict[str, object]] = {
        "model": "primary",
        "fallbacks": ["client-backup"],
    }
    assert {**data, **apply_customer_fallbacks(data, [])} == {
        "model": "primary",
        "fallbacks": [],
        "context_window_fallbacks": [],
        "content_policy_fallbacks": [],
    }


def test_customer_policy_rejects_a_client_router_without_proxy_target_guards() -> None:
    with pytest.raises(ProxyException) as rejected:
        apply_customer_fallbacks({"model": "primary", "user_config": {}}, RULES)
    assert rejected.value.code == "400"
    assert rejected.value.param == "user_config"


def test_reusing_a_key_for_another_customer_does_not_reuse_previous_fallback_rules() -> None:
    token: Final = UserAPIKeyAuth(end_user_id="first", end_user_fallbacks=RULES)
    updated: Final = update_valid_token_with_end_user_params(token, {"end_user_id": "second"})
    assert (updated.end_user_id, updated.end_user_fallbacks) == ("second", None)

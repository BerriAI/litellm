from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy._types import (
    NewCustomerRequest,
    UpdateCustomerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import (
    update_valid_token_with_end_user_params,  # pyright: ignore[reportUnknownVariableType]  # legacy auth helper accepts an untyped parameter dict
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


def test_reusing_a_key_for_another_customer_does_not_reuse_previous_fallback_rules() -> None:
    token: Final = UserAPIKeyAuth(end_user_id="first", end_user_fallbacks=RULES)
    updated: Final = update_valid_token_with_end_user_params(token, {"end_user_id": "second"})
    assert (updated.end_user_id, updated.end_user_fallbacks) == ("second", None)

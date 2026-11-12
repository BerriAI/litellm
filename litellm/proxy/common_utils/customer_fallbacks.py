from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy._types import ProxyException
from litellm.types.proxy.customer_fallbacks import CustomerFallbacks


def customer_fallback_settings(
    fallbacks: CustomerFallbacks | None,
) -> Mapping[str, CustomerFallbacks]:
    if fallbacks is None:
        return {}
    return {
        "fallbacks": fallbacks,
        "context_window_fallbacks": fallbacks,
        "content_policy_fallbacks": fallbacks,
    }


def prepare_customer_fallback_request(
    data: Mapping[str, object], fallbacks: CustomerFallbacks | None
) -> Mapping[str, object]:
    if fallbacks is None:
        return {}
    if data.get("user_config") is not None:
        raise ProxyException(
            message="user_config cannot be used with customer-specific fallback rules",
            type="invalid_request_error",
            param="user_config",
            code=400,
        )
    fields: Final = customer_fallback_settings(fallbacks)
    override: Final = as_str_mapping(data.get("router_settings_override"))
    overrides: Final = (
        {"router_settings_override": {key: value for key, value in override.items() if key not in fields}}
        if override is not None
        else {}
    )
    return {**{field: None for field in fields if field in data}, **overrides}


def apply_customer_fallbacks(data: Mapping[str, object], fallbacks: CustomerFallbacks | None) -> Mapping[str, object]:
    return {**prepare_customer_fallback_request(data, fallbacks), **customer_fallback_settings(fallbacks)}

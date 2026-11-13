from typing import Final

import pytest

from litellm.proxy._types import LiteLLM_JWTAuth, ProxyErrorTypes, ProxyException
from litellm.proxy.auth import core
from litellm.proxy.auth import denial as auth_denial
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustDenial

JWT: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"


def test_proxy_denial_keeps_its_status_type_and_param() -> None:
    error: Final = auth_denial.denial_error(
        RustDenial.Proxy(status=401, message="expired", error_type="expired_key", param="sk-...abcd")
    )

    assert isinstance(error, ProxyException)
    assert (error.message, error.type, error.param, error.code) == (
        "expired",
        ProxyErrorTypes.expired_key,
        "sk-...abcd",
        "401",
    )


@pytest.mark.parametrize(
    ("general_settings", "token", "feature"),
    [
        ({"enable_jwt_auth": True}, JWT, "JWT Auth is an enterprise only feature."),
        ({"enable_oauth2_auth": True}, "sk-opaque", "Oauth2 token validation is only available for premium users."),
    ],
    ids=["jwt", "oauth2"],
)
def test_a_licensed_path_without_a_license_is_a_403_naming_premium_user(
    general_settings: dict[str, object], token: str, feature: str
) -> None:
    settings: Final = RustAuthSettings(general_settings, LiteLLM_JWTAuth(), core.HostFacts(premium=False))
    planned: Final = RustAuth.plan(core.request(token, "/chat/completions"), settings)
    assert isinstance(planned, RustDenial)

    error: Final = auth_denial.denial_error(planned)

    assert isinstance(error, ProxyException)
    assert error.message.startswith(feature)
    assert (error.type, error.param, error.code) == (ProxyErrorTypes.auth_error, "premium_user", "403")

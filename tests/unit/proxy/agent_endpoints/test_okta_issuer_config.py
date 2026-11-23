from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy._types import JWTIssuerConfig, LiteLLM_JWTAuth
from litellm.proxy.auth.jwt import claims

ORG: Final = "example"
ORG_SERVER: Final = "https://example.okta.test"
CUSTOM: Final = ORG_SERVER + "/oauth2/default"
AUDIENCE: Final = "api://gateway"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        pytest.param(
            {"issuer": "https://acme.okta.example/other/path"}, "needs an Okta issuer", id="not_an_okta_issuer"
        ),
        pytest.param({"okta_org": None}, "needs okta_org", id="no_org"),
        pytest.param(
            {"audience": None, "disable_audience_validation": True}, "needs audience validation", id="no_audience_check"
        ),
        pytest.param(
            {"agent_identity_provider": None}, "okta_org needs agent_identity_provider", id="org_without_okta"
        ),
    ],
)
def test_an_okta_issuer_config_that_cannot_be_trusted_fails_to_load(
    settings: Mapping[str, object], message: str
) -> None:
    fields: Final = {
        "issuer": CUSTOM,
        "audience": AUDIENCE,
        "agent_identity_provider": "okta",
        "okta_org": ORG,
        **settings,
    }

    with pytest.raises(ValueError, match=message):
        JWTIssuerConfig.model_validate(fields)


@pytest.mark.parametrize("issuer", [CUSTOM, ORG_SERVER])
def test_an_opted_in_okta_issuer_loads_into_the_rust_claim_reader(issuer: str) -> None:
    jwtauth: Final = LiteLLM_JWTAuth(
        user_email_jwt_field="email",
        issuers=[JWTIssuerConfig(issuer=issuer, audience=AUDIENCE, agent_identity_provider="okta", okta_org=ORG)],
    )

    reading: Final = claims({"iss": issuer, "sub": "00u1maria", "email": "maria@example.com"}, jwtauth)

    assert reading.user_id == "maria@example.com"

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from litellm.types.proxy.agent_identity import OKTA_CUSTOM_ISSUER_PATTERN

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_JWTAuth


@dataclass(frozen=True, slots=True)
class OktaIssuer:
    org: str
    accepts_access_tokens: bool


def okta_issuers(litellm_jwtauth: "LiteLLM_JWTAuth") -> Mapping[str, OktaIssuer]:
    return MappingProxyType(
        {
            item.issuer: OktaIssuer(item.okta_org, re.fullmatch(OKTA_CUSTOM_ISSUER_PATTERN, item.issuer) is not None)
            for item in litellm_jwtauth.issuers or ()
            if item.agent_identity_provider == "okta" and item.okta_org
        }
    )

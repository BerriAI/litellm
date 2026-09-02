"""Whether the SSO provider the login callback dispatches to can capture an IdP identity assertion.

An ``oauth2_id_jag`` MCP server spends the ``id_token`` captured at SSO login as its RFC 8693
subject token. Only the generic OIDC login path reaches a token response the gateway retains one
from, so a deployment whose SSO runs through Google, Microsoft or SAML never stores an assertion
and every store-sourced ID-JAG exchange fails for every user, however many times they sign in.
Neither side can see that alone: the MCP registration knows nothing about SSO and the login knows
nothing about MCP. This module is the one shared answer both warn from.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import assert_never

from litellm.proxy.management_endpoints.sso.saml_sso import SAMLAuthHandler

_GENERIC_OIDC_REMEDY = (
    "Point SSO at the generic OIDC provider (GENERIC_CLIENT_ID), the one login path whose token "
    "response the gateway retains an id_token from"
)


class ActiveSSOProvider(str, Enum):
    google = "google"
    microsoft = "microsoft"
    generic = "generic"
    saml = "saml"
    none = "none"


def active_sso_provider() -> ActiveSSOProvider:
    """The provider the SSO callback will dispatch to.

    Mirrors the callback's precedence rather than reporting everything configured: an environment
    carrying both GOOGLE_CLIENT_ID and GENERIC_CLIENT_ID runs the Google branch, so it must report
    Google. Presence is judged the way the callback judges it, so a client id set to the empty
    string still selects that branch here.
    """
    if os.getenv("GOOGLE_CLIENT_ID") is not None:
        return ActiveSSOProvider.google
    if os.getenv("MICROSOFT_CLIENT_ID") is not None:
        return ActiveSSOProvider.microsoft
    if os.getenv("GENERIC_CLIENT_ID") is not None:
        return ActiveSSOProvider.generic
    if SAMLAuthHandler.is_saml_configured():
        return ActiveSSOProvider.saml
    return ActiveSSOProvider.none


def id_jag_assertion_capture_gap() -> str | None:
    """Why ID-JAG cannot work under the active SSO provider, phrased for an operator reading a log,
    or ``None`` when that provider does capture an assertion."""
    provider = active_sso_provider()
    match provider:
        case ActiveSSOProvider.generic:
            return None
        case ActiveSSOProvider.none:
            return (
                "no SSO provider is configured, so no IdP identity assertion is ever captured and "
                f"ID-JAG credential resolution fails for every user. {_GENERIC_OIDC_REMEDY}"
            )
        case ActiveSSOProvider.google | ActiveSSOProvider.microsoft | ActiveSSOProvider.saml:
            return (
                f"the active SSO provider ({provider.value}) has no identity-assertion capture path, so no "
                "IdP id_token is ever stored and ID-JAG credential resolution fails for every user no matter "
                f"how often they sign in. {_GENERIC_OIDC_REMEDY}"
            )
        case _:
            assert_never(provider)


def id_jag_assertion_capture_gap_at_startup() -> str | None:
    """Config load runs before SSO settings stored in the database are reconciled into the process
    environment, so an unresolved provider at that point is not yet a gap; the SSO callback reports it
    once a login happens."""
    if active_sso_provider() is ActiveSSOProvider.none:
        return None
    return id_jag_assertion_capture_gap()

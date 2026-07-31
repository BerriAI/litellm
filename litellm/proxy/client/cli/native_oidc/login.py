"""Orchestration for `lite login --flow browser|device|auto`.

The CLI talks to the identity provider directly as a public native client. The
proxy contributes only the trust anchor (issuer, client id, scopes); it never
brokers the exchange and never sees the authorization code or refresh token.
"""

from typing import Any  # noqa: TID251  # unvalidated JSON payload

import click

from .browser_flow import run_browser_flow
from .credentials import (
    build_native_credential,
    save_credential,
    verify_token_with_litellm,
)
from .device_flow import run_device_flow
from .errors import NativeOIDCError
from .metadata import (
    NativeOIDCMetadata,
    ProviderMetadata,
    fetch_native_oidc_metadata,
    fetch_provider_metadata,
)

FLOW_AUTO = "auto"
FLOW_BROWSER = "browser"
FLOW_DEVICE = "device"
FLOW_PROXY = "proxy"

NATIVE_FLOW_CHOICES = (FLOW_AUTO, FLOW_BROWSER, FLOW_DEVICE, FLOW_PROXY)


def select_flow(provider: ProviderMetadata, requested: str, *, open_browser: bool) -> str:
    """Resolve the requested flow against what the provider actually supports.

    An explicit `--flow` is never silently downgraded: if the provider cannot
    support it, the specific reason is raised.
    """
    if requested == FLOW_BROWSER:
        provider.assert_browser_flow_supported()
        return FLOW_BROWSER
    if requested == FLOW_DEVICE:
        provider.assert_device_flow_supported()
        return FLOW_DEVICE

    # auto: on a headless machine the device flow is the only usable option.
    if not open_browser and provider.supports_device_flow():
        return FLOW_DEVICE
    if provider.supports_browser_flow():
        return FLOW_BROWSER
    if provider.supports_device_flow():
        return FLOW_DEVICE
    # Neither works -- surface why the primary flow is unavailable.
    provider.assert_browser_flow_supported()
    raise NativeOIDCError("the identity provider supports no usable login flow")


def run_native_login(
    base_url: str,
    *,
    flow: str = FLOW_AUTO,
    open_browser: bool = True,
    echo=click.echo,
) -> dict[str, Any]:
    """Log in against the proxy's advertised identity provider.

    Raises NativeOIDCUnavailable when the proxy offers no native OIDC -- the
    only condition under which the caller may fall back to proxy-mediated SSO.
    """
    metadata: NativeOIDCMetadata = fetch_native_oidc_metadata(base_url)
    provider = fetch_provider_metadata(metadata.issuer)

    chosen = select_flow(provider, flow, open_browser=open_browser)
    echo(f"Signing in via {metadata.issuer} ({chosen} flow)")

    if chosen == FLOW_BROWSER:
        token = run_browser_flow(metadata, provider, open_browser=open_browser, echo=echo)
    else:
        token = run_device_flow(metadata, provider, open_browser=open_browser, echo=echo)

    # The identity provider issuing a token does not mean LiteLLM accepts it.
    # Fail here rather than storing a credential that breaks on first use.
    verify_token_with_litellm(base_url, token.access_token)

    credential = build_native_credential(base_url=base_url, metadata=metadata, token=token)
    save_credential(credential)
    return credential

"""Authorization Code with PKCE S256 for native applications (RFC 8252)."""

import webbrowser
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import click

from litellm.litellm_core_utils.native_oidc_validation import format_scopes

from .callback import LoopbackCallbackListener
from .errors import NativeOIDCError
from .http_client import post_form
from .metadata import NativeOIDCMetadata, ProviderMetadata
from .pkce import PkceChallenge, generate_pkce_challenge
from .tokens import TokenResponse, describe_token_error, parse_token_response

DEFAULT_LOGIN_TIMEOUT_SECONDS = 300.0


def build_authorization_url(
    authorization_endpoint: str,
    *,
    client_id: str,
    redirect_uri: str,
    scopes,
    challenge: PkceChallenge,
) -> str:
    """Build the authorization request URL.

    Any query already present on the authorization endpoint is preserved -- some
    providers legitimately publish endpoints carrying parameters.
    """
    parsed = urlsplit(authorization_endpoint)
    params = parse_qsl(parsed.query, keep_blank_values=True)
    params += [
        ("response_type", "code"),
        ("client_id", client_id),
        ("redirect_uri", redirect_uri),
        ("scope", format_scopes(scopes)),
        ("state", challenge.state),
        ("code_challenge", challenge.code_challenge),
        ("code_challenge_method", challenge.code_challenge_method),
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(params), ""))


def exchange_code_for_token(
    token_endpoint: str,
    *,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str,
) -> TokenResponse:
    """Redeem an authorization code at the token endpoint as a public client."""
    response = post_form(
        token_endpoint,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            # The verifier, never the challenge. No client_secret and no HTTP
            # Basic client authentication: this is a public native client.
            "code_verifier": code_verifier,
        },
    )
    if response.status_code != 200 or response.payload is None:
        raise NativeOIDCError(describe_token_error(response.status_code, response.payload))
    return parse_token_response(response.payload)


def run_browser_flow(
    metadata: NativeOIDCMetadata,
    provider: ProviderMetadata,
    *,
    open_browser: bool = True,
    timeout: float = DEFAULT_LOGIN_TIMEOUT_SECONDS,
    echo=click.echo,
) -> TokenResponse:
    """Run Authorization Code + PKCE S256 against a loopback redirect URI."""
    provider.assert_browser_flow_supported()
    authorization_endpoint = provider.require_authorization_endpoint()
    token_endpoint = provider.require_token_endpoint()

    challenge = generate_pkce_challenge()

    # Bind before opening the browser so the redirect can never race the listener.
    with LoopbackCallbackListener(expected_state=challenge.state) as listener:
        authorization_url = build_authorization_url(
            authorization_endpoint,
            client_id=metadata.client_id,
            redirect_uri=listener.redirect_uri,
            scopes=metadata.scopes,
            challenge=challenge,
        )

        echo("Open this URL to sign in:")
        echo(f"  {authorization_url}")
        if open_browser:
            _try_open_browser(authorization_url, echo=echo)
        echo("Waiting for the browser login to complete...")

        code = listener.wait_for_code(timeout=timeout)

        return exchange_code_for_token(
            token_endpoint,
            code=code,
            redirect_uri=listener.redirect_uri,
            client_id=metadata.client_id,
            code_verifier=challenge.code_verifier,
        )


def _try_open_browser(url: str, *, echo=click.echo) -> Optional[bool]:
    """Best-effort browser launch; failure is never fatal.

    The URL has already been printed, so the user can always continue manually.
    """
    try:
        if webbrowser.open(url):
            return True
    except Exception:  # noqa: BLE001 - any browser backend failure is non-fatal
        pass
    echo("Could not open a browser automatically; open the URL above manually.")
    return False

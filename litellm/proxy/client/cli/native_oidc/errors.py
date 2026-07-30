"""Error types for the native OIDC CLI login flows.

Every message that reaches a user must be safe to print: no access tokens,
refresh tokens, authorization codes, device codes, PKCE verifiers, state values
or raw provider response bodies.
"""


class NativeOIDCError(Exception):
    """A native OIDC login failed.

    Raising this is always a hard failure. Once a proxy advertises native OIDC,
    the CLI must not silently downgrade to the proxy-mediated SSO flow.
    """


class NativeOIDCUnavailable(NativeOIDCError):
    """The proxy does not offer native OIDC.

    The *only* condition that permits `--flow auto` to fall back to the legacy
    proxy-mediated SSO flow: either the discovery route does not exist (older
    proxy) or it returned a valid document with no `native_oidc` object.
    """


class NativeOIDCAuthRejected(NativeOIDCError):
    """LiteLLM refused the identity-provider access token (HTTP 401/403)."""

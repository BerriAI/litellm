"""LiteLLM discovery metadata and OpenID provider metadata.

Two different trust levels are handled here:

- The small `native_oidc` object is a LiteLLM-owned closed contract, so unknown
  fields are rejected (mirroring `extra="forbid"` on the proxy response model).
- The provider configuration document is third-party and open by design, so
  unknown standard/extension fields are ignored. Only the fields the CLI
  actually needs are decoded.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any  # noqa: TID251  # unvalidated JSON payload

from litellm.litellm_core_utils.native_oidc_validation import (
    derive_provider_configuration_url,
    is_trusted_metadata_origin,
    validate_endpoint_url,
    validate_issuer,
    validate_scope_tokens,
)

from .errors import NativeOIDCError, NativeOIDCUnavailable
from .http_client import get_json, get_json_response

LITELLM_DISCOVERY_PATH = "/.well-known/litellm-ui-config"

NATIVE_OIDC_ALLOWED_KEYS = frozenset({"issuer", "client_id", "scopes"})

DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


def build_litellm_discovery_url(base_url: str) -> str:
    """Build the discovery URL relative to the configured proxy base URL.

    Plain concatenation rather than `urljoin`, which would discard a configured
    path prefix (`https://host/litellm` -> `https://host/.well-known/...`).
    """
    return base_url.rstrip("/") + LITELLM_DISCOVERY_PATH


@dataclass(frozen=True)
class NativeOIDCMetadata:
    """The public issuer/client/scopes trust anchor advertised by the proxy."""

    issuer: str
    client_id: str
    scopes: tuple[str, ...]


def parse_native_oidc_metadata(raw: object) -> NativeOIDCMetadata:
    """Strictly validate the LiteLLM-owned `native_oidc` object."""
    if not isinstance(raw, dict):
        raise NativeOIDCError("advertised native_oidc metadata is not an object")

    unknown = set(raw) - NATIVE_OIDC_ALLOWED_KEYS
    if unknown:
        raise NativeOIDCError(
            "advertised native_oidc metadata contains unsupported field(s): " + ", ".join(sorted(unknown))
        )

    issuer = raw.get("issuer")
    client_id = raw.get("client_id")
    scopes = raw.get("scopes")

    if not isinstance(issuer, str):
        raise NativeOIDCError("advertised native_oidc issuer is missing or not a string")
    try:
        validate_issuer(issuer)
    except ValueError as error:
        raise NativeOIDCError(f"advertised native_oidc issuer {error}") from error

    if not isinstance(client_id, str) or not client_id.strip():
        raise NativeOIDCError("advertised native_oidc client_id is missing or blank")

    if not isinstance(scopes, list):
        raise NativeOIDCError("advertised native_oidc scopes is missing or not a list")
    try:
        validated_scopes = validate_scope_tokens(scopes)
    except ValueError as error:
        raise NativeOIDCError(f"advertised native_oidc scopes {error}") from error

    return NativeOIDCMetadata(issuer=issuer, client_id=client_id, scopes=validated_scopes)


def fetch_native_oidc_metadata(base_url: str) -> NativeOIDCMetadata:
    """Fetch the proxy's public native OIDC metadata.

    Raises NativeOIDCUnavailable -- the only fallback-permitting outcome -- when
    the proxy is too old to serve the route or advertises no `native_oidc`.
    Every other problem is a hard NativeOIDCError.
    """
    if not is_trusted_metadata_origin(base_url):
        raise NativeOIDCError(
            "native OIDC login requires an HTTPS proxy URL (or a numeric loopback "
            "address for local development); refusing to bootstrap identity "
            "provider metadata over plaintext HTTP. Use --flow proxy to force the "
            "legacy proxy-mediated login."
        )

    url = build_litellm_discovery_url(base_url)
    response = get_json_response(url)

    if response.status_code in (404, 405):
        raise NativeOIDCUnavailable(f"{url} returned HTTP {response.status_code}; proxy predates native OIDC discovery")
    if response.status_code != 200:
        raise NativeOIDCError(f"{url} returned HTTP {response.status_code}")
    if response.payload is None:
        raise NativeOIDCError(f"{url} did not return a JSON object")

    # Unknown unrelated top-level discovery fields are ignored on purpose.
    if response.payload.get("native_oidc") is None:
        raise NativeOIDCUnavailable("proxy does not advertise native OIDC metadata")

    return parse_native_oidc_metadata(response.payload["native_oidc"])


def _optional_string_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise NativeOIDCError(f"provider metadata {key} is not a list of strings")
    return tuple(value)


def _optional_endpoint(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NativeOIDCError(f"provider metadata {key} is not a string")
    return value


@dataclass(frozen=True)
class ProviderMetadata:
    """The subset of the OpenID provider configuration the CLI needs.

    Endpoints are validated lazily, when a flow actually needs one, so that a
    malformed endpoint the CLI never uses cannot break an otherwise fine login.
    """

    issuer: str
    authorization_endpoint: str | None
    token_endpoint: str | None
    device_authorization_endpoint: str | None
    response_types_supported: tuple[str, ...] | None
    grant_types_supported: tuple[str, ...] | None
    code_challenge_methods_supported: tuple[str, ...] | None
    token_endpoint_auth_methods_supported: tuple[str, ...] | None

    def _require_endpoint(self, value: str | None, name: str) -> str:
        if value is None:
            raise NativeOIDCError(f"provider does not advertise a {name}")
        try:
            return validate_endpoint_url(value)
        except ValueError as error:
            raise NativeOIDCError(f"provider {name} {error}") from error

    def require_authorization_endpoint(self) -> str:
        return self._require_endpoint(self.authorization_endpoint, "authorization_endpoint")

    def require_token_endpoint(self) -> str:
        return self._require_endpoint(self.token_endpoint, "token_endpoint")

    def require_device_authorization_endpoint(self) -> str:
        return self._require_endpoint(self.device_authorization_endpoint, "device_authorization_endpoint")

    def supports_browser_flow(self) -> bool:
        """True when Authorization Code + PKCE S256 is usable with this provider."""
        if self.authorization_endpoint is None or self.token_endpoint is None:
            return False
        if self.code_challenge_methods_supported is not None and "S256" not in self.code_challenge_methods_supported:
            return False
        if self.response_types_supported is not None and "code" not in self.response_types_supported:
            return False
        if self.grant_types_supported is not None and "authorization_code" not in self.grant_types_supported:
            return False
        return self.supports_public_client()

    def supports_device_flow(self) -> bool:
        if self.device_authorization_endpoint is None or self.token_endpoint is None:
            return False
        if self.grant_types_supported is not None and DEVICE_CODE_GRANT_TYPE not in self.grant_types_supported:
            return False
        return self.supports_public_client()

    def supports_public_client(self) -> bool:
        """True when the token endpoint accepts an unauthenticated public client.

        Absent metadata is permitted: RFC 8414 defaults
        `token_endpoint_auth_methods_supported` to `client_secret_basic`, but
        real native-client deployments routinely omit the field, and the token
        request itself will fail loudly if `none` is genuinely unsupported.
        """
        if self.token_endpoint_auth_methods_supported is None:
            return True
        return "none" in self.token_endpoint_auth_methods_supported

    def assert_browser_flow_supported(self) -> None:
        """Raise a specific reason why browser login cannot be used."""
        if self.authorization_endpoint is None:
            raise NativeOIDCError("provider does not advertise an authorization_endpoint")
        if self.token_endpoint is None:
            raise NativeOIDCError("provider does not advertise a token_endpoint")
        if self.code_challenge_methods_supported is not None and "S256" not in self.code_challenge_methods_supported:
            raise NativeOIDCError("provider does not support PKCE S256; browser login is unsupported")
        if self.response_types_supported is not None and "code" not in self.response_types_supported:
            raise NativeOIDCError("provider does not support the 'code' response type")
        if self.grant_types_supported is not None and "authorization_code" not in self.grant_types_supported:
            raise NativeOIDCError("provider does not support the authorization_code grant")
        if not self.supports_public_client():
            raise NativeOIDCError(
                "provider token endpoint does not accept public clients "
                "(token_endpoint_auth_methods_supported has no 'none')"
            )

    def assert_device_flow_supported(self) -> None:
        if self.device_authorization_endpoint is None:
            raise NativeOIDCError("provider does not advertise a device_authorization_endpoint")
        if self.token_endpoint is None:
            raise NativeOIDCError("provider does not advertise a token_endpoint")
        if self.grant_types_supported is not None and DEVICE_CODE_GRANT_TYPE not in self.grant_types_supported:
            raise NativeOIDCError("provider does not support the device_code grant")
        if not self.supports_public_client():
            raise NativeOIDCError(
                "provider token endpoint does not accept public clients "
                "(token_endpoint_auth_methods_supported has no 'none')"
            )


def parse_provider_metadata(raw: object, expected_issuer: str) -> ProviderMetadata:
    """Validate a provider configuration document against the advertised issuer.

    The issuer comparison is an exact string comparison against the value the
    proxy advertised -- never against a normalized or reconstructed form.
    """
    if not isinstance(raw, dict):
        raise NativeOIDCError("provider metadata is not a JSON object")

    issuer = raw.get("issuer")
    if not isinstance(issuer, str):
        raise NativeOIDCError("provider metadata issuer is missing or not a string")
    if issuer != expected_issuer:
        raise NativeOIDCError(
            "provider metadata issuer does not exactly match the issuer advertised by the LiteLLM proxy"
        )

    return ProviderMetadata(
        issuer=issuer,
        authorization_endpoint=_optional_endpoint(raw, "authorization_endpoint"),
        token_endpoint=_optional_endpoint(raw, "token_endpoint"),
        device_authorization_endpoint=_optional_endpoint(raw, "device_authorization_endpoint"),
        response_types_supported=_optional_string_tuple(raw, "response_types_supported"),
        grant_types_supported=_optional_string_tuple(raw, "grant_types_supported"),
        code_challenge_methods_supported=_optional_string_tuple(raw, "code_challenge_methods_supported"),
        token_endpoint_auth_methods_supported=_optional_string_tuple(raw, "token_endpoint_auth_methods_supported"),
    )


def fetch_provider_metadata(issuer: str) -> ProviderMetadata:
    """Discover and validate the provider configuration for `issuer`."""
    return parse_provider_metadata(get_json(derive_provider_configuration_url(issuer)), expected_issuer=issuer)


def format_scope_list(scopes: Sequence[str]) -> str:
    return " ".join(scopes)

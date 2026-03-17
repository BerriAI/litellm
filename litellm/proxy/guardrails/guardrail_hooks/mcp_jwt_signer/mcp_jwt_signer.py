"""
MCPJWTSigner — Built-in LiteLLM guardrail for zero trust MCP authentication.

Signs outbound MCP requests with a LiteLLM-issued RS256 JWT so that MCP servers
can trust a single signing authority (liteLLM) instead of every upstream IdP.

Full feature config (all params optional):

    guardrails:
      - guardrail_name: "mcp-jwt-signer"
        litellm_params:
          guardrail: mcp_jwt_signer
          mode: "pre_mcp_call"
          default_on: true
          issuer: "https://my-litellm.example.com"   # optional
          audience: "mcp"                             # optional
          ttl_seconds: 300                            # optional

          # FR-5: Inbound token verification
          access_token_discovery_uri: "https://login.example.com/.well-known/openid-configuration"
          access_token_introspection_endpoint: null   # for opaque tokens (future)

          # FR-12: End-user identity mapping (ordered; first non-empty wins)
          end_user_claim_sources: ["sub", "sso_id", "preferred_username", "email"]

          # FR-13: Claim operations (Kong parity)
          add_claims: {}           # add new claims if not already present
          set_claims: {}           # override/set claims
          remove_claims: []        # strip claims from output JWT

          # FR-14: Two-token model (access token + channel/agent token)
          channel_token_header: "X-Channel-Token"
          channel_token_discovery_uri: null  # OIDC discovery for channel token IdP
          channel_token_jwks_uri: null       # fallback JWKS URI for channel token IdP

          # FR-15: Incoming claim validation
          required_claims: []     # claims that MUST be present in incoming JWT
          optional_claims: []     # informational; no rejection if absent

          # FR-9: Debug headers
          debug_header: true      # emit x-litellm-mcp-debug on outbound requests

          # FR-10: Configurable scope (admin-defined fine-grained tool control)
          allowed_tools: []       # if non-empty, restricts scope to these tools only

MCP servers verify tokens via:
  GET /.well-known/openid-configuration  → { jwks_uri: ".../.well-known/jwks.json" }
  GET /.well-known/jwks.json             → RSA public key in JWKS format

Optionally set MCP_JWT_SIGNING_KEY env var (PEM string or file:///path) to use
your own RSA keypair. If unset, an RSA-2048 keypair is auto-generated at startup.
"""

import base64
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

# Module-level singleton for the JWKS discovery endpoint to access.
_mcp_jwt_signer_instance: Optional["MCPJWTSigner"] = None

# Default ordered list of sources to resolve end-user identity (FR-12).
_DEFAULT_END_USER_CLAIM_SOURCES: List[str] = [
    "sub",
    "preferred_username",
    "email",
    "user_id",
]

# UserAPIKeyAuth attribute names that are valid sources for end-user identity.
# Only these field names trigger step 3 (attribute lookup on the auth object)
# in _resolve_end_user_identity — prevents spurious matches on mock objects or
# subclasses that happen to have extra properties with the same name as a JWT claim.
_USER_API_KEY_IDENTITY_FIELDS: frozenset = frozenset(
    {
        "user_id",
        "end_user_id",
        "user_email",
        "org_id",
        "team_id",
    }
)


def get_mcp_jwt_signer() -> Optional["MCPJWTSigner"]:
    """Return the active MCPJWTSigner singleton, or None if not initialized."""
    return _mcp_jwt_signer_instance


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _load_private_key_from_env(env_var: str) -> RSAPrivateKey:
    """Load an RSA private key from an env var (PEM string or file:// path)."""
    key_material = os.environ.get(env_var, "")
    if not key_material:
        raise ValueError(
            f"MCPJWTSigner: environment variable '{env_var}' is set but empty."
        )
    if key_material.startswith("file://"):
        path = key_material[len("file://"):]
        with open(path, "rb") as f:
            key_bytes = f.read()
    else:
        key_bytes = key_material.encode("utf-8")
    return serialization.load_pem_private_key(key_bytes, password=None)  # type: ignore[return-value]


def _generate_rsa_key_pair() -> RSAPrivateKey:
    """Generate a new RSA-2048 private key."""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def _int_to_base64url(n: int) -> str:
    """Encode an integer as a base64url string (no padding)."""
    byte_length = (n.bit_length() + 7) // 8
    return (
        base64.urlsafe_b64encode(n.to_bytes(byte_length, byteorder="big"))
        .rstrip(b"=")
        .decode("ascii")
    )


def _compute_kid(public_key: Any) -> str:
    """Derive a key ID from the public key's DER encoding (SHA-256, first 16 hex chars)."""
    der_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# FR-5: OIDC discovery cache (NFR-2: avoid per-request fetches)
# ---------------------------------------------------------------------------


class _OIDCDiscoveryCache:
    """
    Cache for OIDC discovery documents and PyJWKClient instances.

    Avoids per-request OIDC discovery + JWKS fetches (NFR-2).
    Discovery docs are cached indefinitely (they rarely change).
    PyJWKClient handles its own JWKS refresh internally.
    """

    def __init__(self) -> None:
        self._discovery_docs: Dict[str, Dict] = {}
        self._jwks_clients: Dict[str, Any] = {}  # uri -> PyJWKClient

    async def _fetch_discovery_doc(self, discovery_uri: str) -> Dict:
        if discovery_uri in self._discovery_docs:
            return self._discovery_docs[discovery_uri]

        from litellm.llms.custom_httpx.http_handler import (
            get_async_httpx_client,
            httpxSpecialProvider,
        )

        http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.Oauth2Check
        )
        response = await http_client.get(discovery_uri)
        response.raise_for_status()
        doc: Dict = response.json()
        self._discovery_docs[discovery_uri] = doc
        return doc

    async def get_jwks_client(self, discovery_uri: str) -> Any:
        """Return a PyJWKClient for the given OIDC discovery URI."""
        from jwt import PyJWKClient  # type: ignore[attr-defined]

        if discovery_uri not in self._jwks_clients:
            doc = await self._fetch_discovery_doc(discovery_uri)
            jwks_uri = doc.get("jwks_uri")
            if not jwks_uri:
                raise ValueError(
                    f"MCPJWTSigner: OIDC discovery doc at '{discovery_uri}' "
                    "does not contain a 'jwks_uri' field."
                )
            self._jwks_clients[discovery_uri] = PyJWKClient(jwks_uri)
        return self._jwks_clients[discovery_uri]

    async def get_issuer(self, discovery_uri: str) -> Optional[str]:
        """Return the issuer from the OIDC discovery doc."""
        doc = await self._fetch_discovery_doc(discovery_uri)
        return doc.get("issuer")


# Module-level OIDC cache shared across signer instances.
_oidc_cache = _OIDCDiscoveryCache()


# ---------------------------------------------------------------------------
# FR-12: End-user identity resolution
# ---------------------------------------------------------------------------


def _resolve_end_user_identity(
    sources: List[str],
    jwt_claims: Optional[Dict],
    user_api_key_dict: UserAPIKeyAuth,
    raw_headers: Optional[Dict[str, str]],
) -> Optional[str]:
    """
    Resolve end-user identity from an ordered list of sources.

    For each source, tries (in order):
      1. The JWT claim with that name (from the incoming token)
      2. The request header with that name (case-insensitive)
      3. The UserAPIKeyAuth attribute with that name

    Returns the first non-empty string value found, or None.
    """
    normalized_headers: Dict[str, str] = {
        k.lower(): v for k, v in (raw_headers or {}).items()
    }

    for source in sources:
        # 1. JWT claim
        if jwt_claims:
            value = jwt_claims.get(source)
            if value:
                return str(value)

        # 2. Request header (case-insensitive)
        header_val = normalized_headers.get(source.lower())
        if header_val:
            return header_val

        # 3. UserAPIKeyAuth attribute — only for known identity-related fields.
        #    Using an explicit whitelist prevents spurious matches on MagicMock
        #    attributes or extra subclass properties that share a name with a
        #    JWT claim (e.g. mock.sub would return a truthy MagicMock).
        if source in _USER_API_KEY_IDENTITY_FIELDS:
            attr_val = getattr(user_api_key_dict, source, None)
            if attr_val:
                return str(attr_val)

    return None


# ---------------------------------------------------------------------------
# FR-15: Incoming claim validation
# ---------------------------------------------------------------------------


def _validate_required_claims(
    jwt_claims: Optional[Dict],
    required_claims: List[str],
) -> None:
    """
    Validate that all required_claims are present in the incoming JWT claims.

    Raises ValueError if any required claim is missing or if jwt_claims is
    None/empty but required_claims are configured (not a JWT auth request).
    """
    if not required_claims:
        return

    if not jwt_claims:
        raise ValueError(
            f"MCPJWTSigner: required_claims {required_claims} are configured but "
            "the incoming request has no JWT claims. Required claims are only "
            "satisfiable from JWT-authenticated requests (not virtual key auth)."
        )

    missing = [c for c in required_claims if c not in jwt_claims]
    if missing:
        raise ValueError(
            f"MCPJWTSigner: incoming JWT is missing required_claims: {missing}. "
            f"Present claims: {list(jwt_claims.keys())}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MCPJWTSigner(CustomGuardrail):
    """
    Built-in LiteLLM guardrail that signs outbound MCP requests with a
    LiteLLM-issued RS256 JWT, enabling zero trust MCP authentication.

    Features:
      - FR-1/FR-2: RS256 JWT signing
      - FR-3: Configurable issuer/audience
      - FR-5: Verify + re-sign using upstream JWT claims (access_token_discovery_uri)
      - FR-9: Debug headers (x-litellm-mcp-debug)
      - FR-10: Configurable fine-grained scope (allowed_tools)
      - FR-11: act claim (RFC 8693 delegation)
      - FR-12: Configurable end-user identity mapping (end_user_claim_sources)
      - FR-13: Claim operations (add_claims, set_claims, remove_claims)
      - FR-14: Two-token model (access + channel token, channel_token_header)
      - FR-15: Required/optional claim validation
    """

    ALGORITHM = "RS256"
    DEFAULT_TTL = 300
    DEFAULT_AUDIENCE = "mcp"
    SIGNING_KEY_ENV = "MCP_JWT_SIGNING_KEY"

    def __init__(
        self,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        # FR-5: inbound token verification
        access_token_discovery_uri: Optional[str] = None,
        access_token_introspection_endpoint: Optional[str] = None,
        # FR-12: end-user identity mapping
        end_user_claim_sources: Optional[List[str]] = None,
        # FR-13: claim operations
        add_claims: Optional[Dict[str, Any]] = None,
        set_claims: Optional[Dict[str, Any]] = None,
        remove_claims: Optional[List[str]] = None,
        # FR-14: two-token model
        channel_token_header: Optional[str] = None,
        channel_token_discovery_uri: Optional[str] = None,
        channel_token_jwks_uri: Optional[str] = None,
        # FR-15: claim validation
        required_claims: Optional[List[str]] = None,
        optional_claims: Optional[List[str]] = None,
        # FR-9: debug headers
        debug_header: bool = True,
        # FR-10: configurable scope
        allowed_tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Key setup
        key_material = os.environ.get(self.SIGNING_KEY_ENV)
        if key_material:
            self._private_key = _load_private_key_from_env(self.SIGNING_KEY_ENV)
            self._persistent_key: bool = True
            verbose_proxy_logger.info(
                "MCPJWTSigner: loaded RSA key from env var %s", self.SIGNING_KEY_ENV
            )
        else:
            self._private_key = _generate_rsa_key_pair()
            self._persistent_key = False
            verbose_proxy_logger.info(
                "MCPJWTSigner: auto-generated RSA-2048 keypair (set %s to use your own key)",
                self.SIGNING_KEY_ENV,
            )

        self._public_key = self._private_key.public_key()
        self._kid = _compute_kid(self._public_key)

        # Core claims
        self.issuer: str = (
            issuer
            or os.environ.get("MCP_JWT_ISSUER")
            or os.environ.get("LITELLM_EXTERNAL_URL")
            or "litellm"
        )
        self.audience: str = (
            audience
            or os.environ.get("MCP_JWT_AUDIENCE")
            or self.DEFAULT_AUDIENCE
        )
        resolved_ttl = int(
            ttl_seconds
            if ttl_seconds is not None
            else os.environ.get("MCP_JWT_TTL_SECONDS", str(self.DEFAULT_TTL))
        )
        if resolved_ttl <= 0:
            raise ValueError(
                f"MCPJWTSigner: ttl_seconds must be > 0, got {resolved_ttl}"
            )
        self.ttl_seconds: int = resolved_ttl

        # FR-5
        self.access_token_discovery_uri: Optional[str] = access_token_discovery_uri
        self.access_token_introspection_endpoint: Optional[str] = (
            access_token_introspection_endpoint
        )

        # FR-12
        self.end_user_claim_sources: List[str] = (
            end_user_claim_sources
            if end_user_claim_sources is not None
            else _DEFAULT_END_USER_CLAIM_SOURCES
        )

        # FR-13
        self.add_claims: Dict[str, Any] = add_claims or {}
        self.set_claims: Dict[str, Any] = set_claims or {}
        self.remove_claims: List[str] = remove_claims or []

        # FR-14
        self.channel_token_header: str = (
            channel_token_header or "X-Channel-Token"
        ).lower()
        self.channel_token_discovery_uri: Optional[str] = channel_token_discovery_uri
        self.channel_token_jwks_uri: Optional[str] = channel_token_jwks_uri

        # FR-15
        self.required_claims: List[str] = required_claims or []
        self.optional_claims: List[str] = optional_claims or []

        # FR-9
        self.debug_header: bool = debug_header

        # FR-10
        self.allowed_tools: List[str] = allowed_tools or []

        # Register singleton so the JWKS endpoint can access it.
        global _mcp_jwt_signer_instance
        if _mcp_jwt_signer_instance is not None:
            verbose_proxy_logger.warning(
                "MCPJWTSigner: replacing existing singleton — previously issued tokens "
                "signed with the old key will fail JWKS verification. "
                "Avoid configuring multiple mcp_jwt_signer guardrails."
            )
        _mcp_jwt_signer_instance = self

        verbose_proxy_logger.info(
            "MCPJWTSigner initialized: issuer=%s audience=%s ttl=%ds kid=%s "
            "access_token_discovery_uri=%s channel_token_header=%s",
            self.issuer,
            self.audience,
            self.ttl_seconds,
            self._kid,
            self.access_token_discovery_uri or "(none — M2M mode)",
            self.channel_token_header,
        )

    # ------------------------------------------------------------------
    # Public helpers (used by /.well-known/jwks.json endpoint)
    # ------------------------------------------------------------------

    @property
    def jwks_max_age(self) -> int:
        """
        Recommended Cache-Control max-age for the JWKS response (seconds).

        Use 1 hour for persistent keys (loaded from env var) — safe to cache long.
        Use 5 minutes for auto-generated keys — key rotates on every restart, so
        MCP servers must re-fetch quickly to avoid verifying with a stale key.
        """
        return 3600 if self._persistent_key else 300

    def get_jwks(self) -> Dict[str, Any]:
        """
        Return the JWKS (JSON Web Key Set) for the RSA public key.
        Used by GET /.well-known/jwks.json so MCP servers can verify tokens.
        """
        public_numbers = self._public_key.public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "alg": self.ALGORITHM,
                    "use": "sig",
                    "kid": self._kid,
                    "n": _int_to_base64url(public_numbers.n),
                    "e": _int_to_base64url(public_numbers.e),
                }
            ]
        }

    # ------------------------------------------------------------------
    # FR-14: Channel token verification
    # ------------------------------------------------------------------

    async def _verify_channel_token(self, channel_token: str) -> Dict[str, Any]:
        """
        Decode and verify the channel token JWT. Returns decoded claims.

        If channel_token_discovery_uri or channel_token_jwks_uri is configured,
        the signature is verified. Otherwise the token is decoded without
        signature verification (with a warning logged).
        """
        if self.channel_token_discovery_uri:
            jwks_client = await _oidc_cache.get_jwks_client(
                self.channel_token_discovery_uri
            )
            signing_key = jwks_client.get_signing_key_from_jwt(channel_token)
            return jwt.decode(
                channel_token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                options={"verify_aud": False},
            )

        if self.channel_token_jwks_uri:
            from jwt import PyJWKClient  # type: ignore[attr-defined]

            jwks_client = PyJWKClient(self.channel_token_jwks_uri)
            signing_key = jwks_client.get_signing_key_from_jwt(channel_token)
            return jwt.decode(
                channel_token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                options={"verify_aud": False},
            )

        verbose_proxy_logger.warning(
            "MCPJWTSigner: channel token present but no channel_token_discovery_uri "
            "or channel_token_jwks_uri configured — decoding without signature "
            "verification. Configure one to enable zero trust channel token auth."
        )
        return jwt.decode(channel_token, options={"verify_signature": False})

    # ------------------------------------------------------------------
    # Internal claim building
    # ------------------------------------------------------------------

    def _build_scope(self, tool_name: str) -> str:
        """
        Build the scope claim. FR-10: uses allowed_tools if configured;
        otherwise auto-generates least-privilege tool-scoped access.
        """
        if self.allowed_tools:
            # Admin-defined fine-grained scope: only allowed tools get scopes.
            scope_parts = []
            for allowed_tool in self.allowed_tools:
                sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", allowed_tool)
                scope_parts.append(f"mcp:tools/{sanitized}:call")
                scope_parts.append(f"mcp:tools/{sanitized}:list")
            # tools/list only granted when not in the middle of a specific tool call.
            if not tool_name:
                scope_parts.append("mcp:tools/list")
            return " ".join(sorted(set(scope_parts)))

        # Auto-generated least-privilege scope (original behaviour).
        if tool_name:
            return f"mcp:tools/call mcp:tools/{tool_name}:call"
        return "mcp:tools/call mcp:tools/list"

    def _build_claims(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        channel_token_claims: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Build JWT claims from the authenticated user context and MCP request data.
        Follows RFC 8693 (OAuth 2.0 Token Exchange) for sub/act semantics.

        When access_token_discovery_uri is configured (FR-5), upstream jwt_claims
        from the already-validated incoming token are used as the source of truth
        for end-user identity, rather than the LiteLLM virtual-key profile.
        """
        now = int(time.time())
        claims: Dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "nbf": now,
        }

        # FR-5: Use upstream jwt_claims when available (verify + re-sign mode).
        jwt_claims: Optional[Dict] = getattr(user_api_key_dict, "jwt_claims", None)

        # Raw request headers (for FR-12 header-based identity resolution).
        raw_headers: Optional[Dict[str, str]] = data.get("mcp_raw_headers")

        # FR-12: Resolve sub (end-user identity) from ordered sources.
        end_user = _resolve_end_user_identity(
            self.end_user_claim_sources,
            jwt_claims,
            user_api_key_dict,
            raw_headers,
        )
        if end_user:
            claims["sub"] = end_user
        else:
            token = getattr(user_api_key_dict, "token", None) or getattr(
                user_api_key_dict, "api_key", None
            )
            if token:
                claims["sub"] = "apikey:" + hashlib.sha256(
                    str(token).encode()
                ).hexdigest()[:16]
            else:
                claims["sub"] = "litellm-proxy"

        # Email: prefer jwt_claims, fall back to user profile.
        email = (jwt_claims or {}).get("email") or getattr(
            user_api_key_dict, "user_email", None
        )
        if email:
            claims["email"] = email

        # FR-14: Two-token model — act reflects the requester/agent identity.
        if channel_token_claims:
            # Channel token present: its sub (or client_id) is the actor.
            channel_sub = (
                channel_token_claims.get("sub")
                or channel_token_claims.get("client_id")
                or "unknown-agent"
            )
            act: Dict[str, Any] = {"sub": channel_sub}
            if channel_token_claims.get("client_id"):
                act["client_id"] = channel_token_claims["client_id"]
            claims["act"] = act
        else:
            # Fallback: team_id or org_id as the acting entity (RFC 8693).
            team_id = getattr(user_api_key_dict, "team_id", None)
            org_id = getattr(user_api_key_dict, "org_id", None)
            claims["act"] = {"sub": team_id or org_id or "litellm-proxy"}

        # FR-10: Scope claim — tool-level least-privilege access.
        raw_tool_name: str = data.get("mcp_tool_name", "")
        tool_name = (
            re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_tool_name) if raw_tool_name else ""
        )
        claims["scope"] = self._build_scope(tool_name)

        # FR-13: Claim operations — applied in Kong order: add → set → remove.
        for k, v in self.add_claims.items():
            if k not in claims:
                claims[k] = v
        claims.update(self.set_claims)
        for k in self.remove_claims:
            claims.pop(k, None)

        return claims

    # ------------------------------------------------------------------
    # Guardrail hook
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Signs a JWT and injects it as the outbound Authorization header for
        MCP tool calls. All other call types pass through unchanged.

        Also handles:
          - FR-5: Uses upstream jwt_claims for re-signing when available
          - FR-9: Emits x-litellm-mcp-debug header
          - FR-14: Reads and verifies channel token for two-token model
          - FR-15: Validates required_claims against incoming JWT claims
        """
        if call_type != "call_mcp_tool":
            return data

        jwt_claims: Optional[Dict] = getattr(user_api_key_dict, "jwt_claims", None)

        # FR-5: When access_token_discovery_uri is set, the guardrail operates in
        # "verify + re-sign" mode. Verification of the incoming JWT is already
        # performed by liteLLM's JWT auth handler (configured independently via
        # general_settings.litellm_jwtauth). The decoded claims land in
        # user_api_key_dict.jwt_claims and are used directly for re-signing.
        if self.access_token_discovery_uri and not jwt_claims:
            verbose_proxy_logger.debug(
                "MCPJWTSigner: access_token_discovery_uri is configured but incoming "
                "request has no JWT claims (virtual key auth). Proceeding with "
                "M2M signing from user profile."
            )

        # FR-15: Validate required claims against the incoming token.
        try:
            _validate_required_claims(jwt_claims, self.required_claims)
        except ValueError as exc:
            raise exc  # Propagate to block the MCP call

        # FR-14: Resolve channel token for two-token model.
        channel_token_claims: Optional[Dict] = None
        raw_headers: Optional[Dict[str, str]] = data.get("mcp_raw_headers")
        if raw_headers:
            normalized_headers = {k.lower(): v for k, v in raw_headers.items()}
            channel_token_raw = normalized_headers.get(self.channel_token_header)
            if channel_token_raw:
                try:
                    channel_token_claims = await self._verify_channel_token(
                        channel_token_raw
                    )
                    verbose_proxy_logger.debug(
                        "MCPJWTSigner: channel token resolved — act.sub=%s",
                        channel_token_claims.get("sub") or channel_token_claims.get("client_id"),
                    )
                except Exception as exc:
                    verbose_proxy_logger.warning(
                        "MCPJWTSigner: channel token verification failed (%s). "
                        "Falling back to single-token mode.",
                        exc,
                    )

        claims = self._build_claims(user_api_key_dict, data, channel_token_claims)

        signed_token = jwt.encode(
            claims,
            self._private_key,
            algorithm=self.ALGORITHM,
            headers={"kid": self._kid},
        )

        # Merge into existing extra_headers rather than replacing — a prior guardrail
        # in the chain may have already injected headers (e.g. tracing, correlation IDs).
        # MCPJWTSigner sets Authorization last so its JWT takes precedence.
        existing_headers: Dict[str, str] = data.get("extra_headers") or {}
        outbound_headers: Dict[str, str] = {
            **existing_headers,
            "Authorization": f"Bearer {signed_token}",
        }

        # FR-9: Debug header — tells downstream what auth resolution was used.
        if self.debug_header:
            debug_info: Dict[str, Any] = {
                "signer": "mcp_jwt_signer",
                "kid": self._kid,
                "issuer": self.issuer,
                "sub": claims.get("sub"),
                "act": claims.get("act"),
                "mode": "re-sign" if jwt_claims else "sign",
                "channel_token": channel_token_claims is not None,
            }
            outbound_headers["x-litellm-mcp-debug"] = json.dumps(
                debug_info, separators=(",", ":")
            )

        data["extra_headers"] = outbound_headers

        verbose_proxy_logger.debug(
            "MCPJWTSigner: signed JWT sub=%s act=%s tool=%s exp=%d mode=%s",
            claims.get("sub"),
            claims.get("act", {}).get("sub"),
            data.get("mcp_tool_name"),
            claims["exp"],
            "re-sign" if jwt_claims else "sign",
        )

        return data

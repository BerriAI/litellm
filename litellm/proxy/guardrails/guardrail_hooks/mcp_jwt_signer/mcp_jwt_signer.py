"""
MCPJWTSigner — Built-in LiteLLM guardrail for zero trust MCP authentication.

Signs outbound MCP requests with a LiteLLM-issued RS256 JWT so that MCP servers
can trust a single signing authority (liteLLM) instead of every upstream IdP.

Usage in config.yaml:

    guardrails:
      - guardrail_name: "mcp-jwt-signer"
        litellm_params:
          guardrail: mcp_jwt_signer
          mode: "pre_mcp_call"
          default_on: true

          # Core signing config
          issuer: "https://my-litellm.example.com"   # optional
          audience: "mcp"                             # optional
          ttl_seconds: 300                            # optional

          # FR-5: Verify + re-sign — validate incoming Bearer token before signing
          access_token_discovery_uri: "https://idp.example.com/.well-known/openid-configuration"
          token_introspection_endpoint: "https://idp.example.com/introspect"  # opaque tokens
          verify_issuer: "https://idp.example.com"   # expected iss in incoming JWT
          verify_audience: "api://my-app"             # expected aud in incoming JWT

          # FR-12: End-user identity mapping — ordered resolution chain
          # Supported: token:<claim>, litellm:user_id, litellm:email,
          #            litellm:end_user_id, litellm:team_id
          end_user_claim_sources:
            - "token:sub"
            - "token:email"
            - "litellm:user_id"

          # FR-13: Claim operations
          add_claims:             # add if key not already present in the JWT
            deployment_id: "prod-001"
          set_claims:             # always set (overrides computed value)
            env: "production"
          remove_claims:          # remove from final JWT
            - "nbf"

          # FR-14: Two-token model — issue a second JWT for the MCP transport channel
          channel_token_audience: "bedrock-gateway"
          channel_token_ttl: 60

          # FR-15: Incoming claim validation — enforce required IdP claims
          required_claims:
            - "sub"
            - "email"
          optional_claims:        # pass through from jwt_claims into outbound JWT
            - "groups"
            - "roles"

          # FR-9: Debug headers
          debug_headers: false    # emit x-litellm-mcp-debug header when true

          # FR-10: Configurable scopes — explicit list replaces auto-generation
          allowed_scopes:
            - "mcp:tools/call"
            - "mcp:tools/list"

MCP servers verify tokens via:
  GET /.well-known/openid-configuration  → { jwks_uri: ".../.well-known/jwks.json" }
  GET /.well-known/jwks.json             → RSA public key in JWKS format

Optionally set MCP_JWT_SIGNING_KEY env var (PEM string or file:///path) to use
your own RSA keypair. If unset, an RSA-2048 keypair is auto-generated at startup.
"""

import base64
import hashlib
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

# Simple in-memory JWKS cache: keyed by JWKS URI → (keys_list, fetched_at).
_jwks_cache: Dict[str, tuple] = {}
_JWKS_CACHE_TTL = 3600  # 1 hour


def get_mcp_jwt_signer() -> Optional["MCPJWTSigner"]:
    """Return the active MCPJWTSigner singleton, or None if not initialized."""
    return _mcp_jwt_signer_instance


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


async def _fetch_jwks(jwks_uri: str) -> List[Dict[str, Any]]:
    """
    Fetch and cache a JWKS from the given URI.

    Results are cached for _JWKS_CACHE_TTL seconds to avoid hammering the IdP.
    """
    now = time.time()
    cached = _jwks_cache.get(jwks_uri)
    if cached is not None:
        keys, fetched_at = cached
        if now - fetched_at < _JWKS_CACHE_TTL:
            return keys  # type: ignore[return-value]

    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        httpxSpecialProvider,
    )

    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    resp = await client.get(jwks_uri, headers={"Accept": "application/json"})
    resp.raise_for_status()
    keys = resp.json().get("keys", [])
    _jwks_cache[jwks_uri] = (keys, now)
    return keys  # type: ignore[return-value]


async def _fetch_oidc_discovery(discovery_uri: str) -> Dict[str, Any]:
    """Fetch an OIDC discovery document and return its parsed JSON."""
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,
        httpxSpecialProvider,
    )

    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
    resp = await client.get(discovery_uri, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


class MCPJWTSigner(CustomGuardrail):
    """
    Built-in LiteLLM guardrail that signs outbound MCP requests with a
    LiteLLM-issued RS256 JWT, enabling zero trust authentication.

    MCP servers verify tokens using liteLLM's OIDC discovery endpoint and
    JWKS endpoint rather than trusting each upstream IdP directly.

    The signed JWT carries:
      - iss: LiteLLM issuer identifier
      - aud: MCP audience (configurable)
      - sub: End-user identity (resolved via end_user_claim_sources, RFC 8693)
      - act: Actor/agent identity (team_id or org_id, RFC 8693 delegation)
      - scope: Tool-level access scopes (configurable via allowed_scopes)
      - iat, exp, nbf: Standard timing claims

    Feature set:
      FR-5:  Verify + re-sign (access_token_discovery_uri, token_introspection_endpoint)
      FR-9:  Debug headers (debug_headers)
      FR-10: Configurable scopes (allowed_scopes)
      FR-12: Configurable end-user identity mapping (end_user_claim_sources)
      FR-13: Claim operations (add_claims, set_claims, remove_claims)
      FR-14: Two-token model (channel_token_audience, channel_token_ttl)
      FR-15: Incoming claim validation (required_claims, optional_claims)
    """

    ALGORITHM = "RS256"
    DEFAULT_TTL = 300
    DEFAULT_AUDIENCE = "mcp"
    SIGNING_KEY_ENV = "MCP_JWT_SIGNING_KEY"

    def __init__(
        self,
        # Core signing config
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        # FR-5: Verify + re-sign
        access_token_discovery_uri: Optional[str] = None,
        token_introspection_endpoint: Optional[str] = None,
        verify_issuer: Optional[str] = None,
        verify_audience: Optional[str] = None,
        # FR-12: End-user identity mapping
        end_user_claim_sources: Optional[List[str]] = None,
        # FR-13: Claim operations
        add_claims: Optional[Dict[str, Any]] = None,
        set_claims: Optional[Dict[str, Any]] = None,
        remove_claims: Optional[List[str]] = None,
        # FR-14: Two-token model
        channel_token_audience: Optional[str] = None,
        channel_token_ttl: Optional[int] = None,
        # FR-15: Incoming claim validation
        required_claims: Optional[List[str]] = None,
        optional_claims: Optional[List[str]] = None,
        # FR-9: Debug headers
        debug_headers: bool = False,
        # FR-10: Configurable scopes
        allowed_scopes: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- Signing key setup ---
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

        # --- Core config ---
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

        # --- FR-5: Verify + re-sign ---
        self.access_token_discovery_uri: Optional[str] = access_token_discovery_uri
        self.token_introspection_endpoint: Optional[str] = token_introspection_endpoint
        self.verify_issuer: Optional[str] = verify_issuer
        self.verify_audience: Optional[str] = verify_audience
        # Cached OIDC discovery document (fetched lazily on first use)
        self._oidc_discovery_doc: Optional[Dict[str, Any]] = None

        # --- FR-12: End-user identity mapping ---
        # Default chain: try incoming JWT sub, fall back to litellm user_id
        self.end_user_claim_sources: List[str] = end_user_claim_sources or [
            "token:sub",
            "litellm:user_id",
        ]

        # --- FR-13: Claim operations ---
        self.add_claims: Dict[str, Any] = add_claims or {}
        self.set_claims: Dict[str, Any] = set_claims or {}
        self.remove_claims: List[str] = remove_claims or []

        # --- FR-14: Two-token model ---
        self.channel_token_audience: Optional[str] = channel_token_audience
        self.channel_token_ttl: int = (
            channel_token_ttl if channel_token_ttl is not None else self.ttl_seconds
        )

        # --- FR-15: Incoming claim validation ---
        self.required_claims: List[str] = required_claims or []
        self.optional_claims: List[str] = optional_claims or []

        # --- FR-9: Debug headers ---
        self.debug_headers: bool = debug_headers

        # --- FR-10: Configurable scopes ---
        self.allowed_scopes: Optional[List[str]] = allowed_scopes

        # Register singleton for JWKS/OIDC discovery endpoints.
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
            "verify=%s channel_token=%s debug=%s",
            self.issuer,
            self.audience,
            self.ttl_seconds,
            self._kid,
            bool(self.access_token_discovery_uri),
            bool(self.channel_token_audience),
            self.debug_headers,
        )

    # ------------------------------------------------------------------
    # Public helpers (used by /.well-known/jwks.json endpoint)
    # ------------------------------------------------------------------

    @property
    def jwks_max_age(self) -> int:
        """
        Recommended Cache-Control max-age for the JWKS response (seconds).

        1 hour for persistent keys; 5 minutes for auto-generated keys so MCP
        servers re-fetch quickly after a proxy restart.
        """
        return 3600 if self._persistent_key else 300

    def get_jwks(self) -> Dict[str, Any]:
        """
        Return the JWKS for the RSA public key.
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
    # FR-5: Verify + re-sign helpers
    # ------------------------------------------------------------------

    async def _get_oidc_discovery(self) -> Dict[str, Any]:
        """Lazily fetch and cache the OIDC discovery document.

        Only caches when the doc contains a 'jwks_uri' so that a transient or
        malformed response (missing the key) doesn't permanently disable JWT
        verification until proxy restart.
        """
        if self._oidc_discovery_doc is None and self.access_token_discovery_uri:
            doc = await _fetch_oidc_discovery(self.access_token_discovery_uri)
            if "jwks_uri" in doc:
                self._oidc_discovery_doc = doc
            else:
                return doc
        return self._oidc_discovery_doc or {}

    async def _verify_incoming_jwt(self, raw_token: str) -> Dict[str, Any]:
        """
        Verify an incoming Bearer JWT against the configured IdP's JWKS.

        Returns the verified payload claims dict.
        Raises jwt.PyJWTError (or subclass) if verification fails.
        """
        discovery = await self._get_oidc_discovery()
        jwks_uri = discovery.get("jwks_uri")
        if not jwks_uri:
            raise ValueError(
                "MCPJWTSigner: access_token_discovery_uri discovery document "
                f"at {self.access_token_discovery_uri!r} has no 'jwks_uri'."
            )

        jwks_keys = await _fetch_jwks(jwks_uri)

        unverified_header = jwt.get_unverified_header(raw_token)
        kid = unverified_header.get("kid")
        alg = unverified_header.get("alg", "RS256")

        # Build a JWKS object and pick the matching key.
        # PyJWT's PyJWKSet handles key-type parsing and kid matching correctly.
        from jwt import PyJWKSet

        try:
            jwks_set = PyJWKSet.from_dict({"keys": jwks_keys})
        except Exception as exc:
            raise jwt.exceptions.PyJWKSetError(  # type: ignore[attr-defined]
                f"Failed to parse JWKS from {jwks_uri!r}: {exc}"
            ) from exc

        signing_jwk = None
        for jwk_obj in jwks_set.keys:
            if not kid or jwk_obj.key_id == kid:
                signing_jwk = jwk_obj
                break

        if signing_jwk is None:
            raise jwt.exceptions.PyJWKSetError(  # type: ignore[attr-defined]
                f"No JWKS key matching kid={kid!r} at {jwks_uri!r}"
            )

        decode_options: Dict[str, Any] = {"verify_exp": True}
        decode_kwargs: Dict[str, Any] = {
            "algorithms": [alg],
            "options": decode_options,
        }
        if self.verify_audience:
            decode_kwargs["audience"] = self.verify_audience
        else:
            decode_options["verify_aud"] = False

        if self.verify_issuer:
            decode_kwargs["issuer"] = self.verify_issuer

        payload: Dict[str, Any] = jwt.decode(
            raw_token, signing_jwk.key, **decode_kwargs
        )
        return payload

    async def _introspect_opaque_token(self, token: str) -> Dict[str, Any]:
        """
        Perform RFC 7662 token introspection for opaque (non-JWT) tokens.

        Returns the introspection response dict.  Raises on HTTP error or
        inactive token.
        """
        if not self.token_introspection_endpoint:
            raise ValueError(
                "MCPJWTSigner: token_introspection_endpoint is required for "
                "opaque token verification but is not configured."
            )

        from litellm.llms.custom_httpx.http_handler import (
            get_async_httpx_client,
            httpxSpecialProvider,
        )

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        resp = await client.post(
            self.token_introspection_endpoint,
            data={"token": token},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        result: Dict[str, Any] = resp.json()
        if not result.get("active", False):
            raise jwt.exceptions.ExpiredSignatureError(  # type: ignore[attr-defined]
                "MCPJWTSigner: incoming token is inactive (introspection returned active=false)"
            )
        return result

    # ------------------------------------------------------------------
    # FR-15: Incoming claim validation
    # ------------------------------------------------------------------

    def _validate_required_claims(
        self,
        jwt_claims: Optional[Dict[str, Any]],
    ) -> None:
        """
        Raise HTTP 403 if any required_claims are absent from the verified
        incoming token claims.
        """
        if not self.required_claims:
            return

        from fastapi import HTTPException

        missing = [c for c in self.required_claims if not (jwt_claims or {}).get(c)]
        if missing:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": (
                        f"MCPJWTSigner: incoming token is missing required claims: "
                        f"{missing}. Configure the IdP to include these claims."
                    )
                },
            )

    # ------------------------------------------------------------------
    # FR-12: End-user identity mapping
    # ------------------------------------------------------------------

    def _resolve_end_user_identity(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        jwt_claims: Optional[Dict[str, Any]],
    ) -> str:
        """
        Resolve the outbound JWT 'sub' using the ordered end_user_claim_sources list.

        Supported source prefixes:
          token:<claim>       — from verified incoming JWT / introspection claims
          litellm:user_id     — from UserAPIKeyAuth.user_id
          litellm:email       — from UserAPIKeyAuth.user_email
          litellm:end_user_id — from UserAPIKeyAuth.end_user_id
          litellm:team_id     — from UserAPIKeyAuth.team_id

        Falls back to a stable hash of the API token for service-account callers.
        """
        for source in self.end_user_claim_sources:
            value: Optional[str] = None

            if source.startswith("token:"):
                claim_name = source[len("token:"):]
                raw = (jwt_claims or {}).get(claim_name)
                value = str(raw) if raw else None

            elif source == "litellm:user_id":
                uid = getattr(user_api_key_dict, "user_id", None)
                value = str(uid) if uid else None

            elif source == "litellm:email":
                email = getattr(user_api_key_dict, "user_email", None)
                value = str(email) if email else None

            elif source == "litellm:end_user_id":
                eid = getattr(user_api_key_dict, "end_user_id", None)
                value = str(eid) if eid else None

            elif source == "litellm:team_id":
                tid = getattr(user_api_key_dict, "team_id", None)
                value = str(tid) if tid else None

            else:
                verbose_proxy_logger.warning(
                    "MCPJWTSigner: unknown end_user_claim_source %r — skipping", source
                )
                continue

            if value:
                return value

        # Final fallback for service accounts with no user identity
        token = getattr(user_api_key_dict, "token", None) or getattr(
            user_api_key_dict, "api_key", None
        )
        if token:
            return "apikey:" + hashlib.sha256(str(token).encode()).hexdigest()[:16]
        return "litellm-proxy"

    # ------------------------------------------------------------------
    # FR-10: Scope building
    # ------------------------------------------------------------------

    def _build_scope(self, raw_tool_name: str) -> str:
        """
        Build the JWT scope string.

        When allowed_scopes is configured: join them verbatim.
        Otherwise auto-generate minimal, least-privilege scopes:
          - Tool call   → mcp:tools/call  mcp:tools/<name>:call
          - No tool     → mcp:tools/call  mcp:tools/list

        NOTE: tools/list is intentionally NOT granted on tool-call JWTs to
        prevent callers from enumerating tools they didn't ask to use.
        """
        if self.allowed_scopes is not None:
            return " ".join(self.allowed_scopes)

        tool_name = (
            re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_tool_name) if raw_tool_name else ""
        )
        if tool_name:
            scopes = ["mcp:tools/call", f"mcp:tools/{tool_name}:call"]
        else:
            scopes = ["mcp:tools/call", "mcp:tools/list"]
        return " ".join(scopes)

    # ------------------------------------------------------------------
    # FR-13: Claim operations
    # ------------------------------------------------------------------

    def _apply_claim_operations(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """Apply add_claims, set_claims, and remove_claims to the claim dict."""
        # add_claims: insert only when key is absent
        for k, v in self.add_claims.items():
            if k not in claims:
                claims[k] = v

        # set_claims: always override (highest priority)
        claims = {**claims, **self.set_claims}

        # remove_claims: delete listed keys
        for k in self.remove_claims:
            claims.pop(k, None)

        return claims

    # ------------------------------------------------------------------
    # FR-15: optional_claims passthrough
    # ------------------------------------------------------------------

    def _passthrough_optional_claims(
        self,
        claims: Dict[str, Any],
        jwt_claims: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Forward optional_claims from verified incoming token into the outbound JWT."""
        if not self.optional_claims or not jwt_claims:
            return claims
        for claim in self.optional_claims:
            if claim in jwt_claims and claim not in claims:
                claims[claim] = jwt_claims[claim]
        return claims

    # ------------------------------------------------------------------
    # Core JWT builder
    # ------------------------------------------------------------------

    def _build_claims(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        jwt_claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build JWT claims for the outbound MCP access token.

        Args:
            user_api_key_dict: LiteLLM auth context for the current request.
            data: Pre-call hook data dict (contains mcp_tool_name etc.).
            jwt_claims: Verified incoming IdP claims (FR-5), or LiteLLM-decoded
                        jwt_claims if available.  None for pure API-key requests.
        """
        now = int(time.time())
        claims: Dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "nbf": now,
        }

        # sub — resolved via ordered claim sources (FR-12)
        claims["sub"] = self._resolve_end_user_identity(user_api_key_dict, jwt_claims)

        # email passthrough when available from LiteLLM context
        user_email = getattr(user_api_key_dict, "user_email", None)
        if user_email:
            claims["email"] = user_email

        # act — RFC 8693 delegation claim (team/org context)
        team_id = getattr(user_api_key_dict, "team_id", None)
        org_id = getattr(user_api_key_dict, "org_id", None)
        act_sub = team_id or org_id or "litellm-proxy"
        claims["act"] = {"sub": act_sub}

        # end_user_id when set separately from user_id
        end_user_id = getattr(user_api_key_dict, "end_user_id", None)
        if end_user_id:
            claims["end_user_id"] = end_user_id

        # scope (FR-10)
        raw_tool_name: str = data.get("mcp_tool_name", "")
        claims["scope"] = self._build_scope(raw_tool_name)

        # optional_claims passthrough (FR-15)
        claims = self._passthrough_optional_claims(claims, jwt_claims)

        # Claim operations — applied last so admin overrides take effect (FR-13)
        claims = self._apply_claim_operations(claims)

        return claims

    def _build_channel_token_claims(
        self,
        base_claims: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build claims for the channel token (FR-14 two-token model).

        Inherits sub/act/scope from the access token but uses a separate
        audience and TTL so the transport layer and resource layer receive
        purpose-bound credentials.
        """
        now = int(time.time())
        return {
            **base_claims,
            "aud": self.channel_token_audience,
            "iat": now,
            "exp": now + self.channel_token_ttl,
            "nbf": now,
        }

    # ------------------------------------------------------------------
    # FR-9: Debug header
    # ------------------------------------------------------------------

    @staticmethod
    def _build_debug_header(claims: Dict[str, Any], kid: str) -> str:
        """
        Build the x-litellm-mcp-debug header value.

        Format: v=1; kid=<kid>; sub=<sub>; iss=<iss>; exp=<exp>; scope=<scope>
        Scope is truncated to 80 chars for header safety.
        """
        sub = claims.get("sub", "")
        iss = claims.get("iss", "")
        exp = claims.get("exp", 0)
        scope = claims.get("scope", "")
        if len(scope) > 80:
            scope = scope[:77] + "..."
        return f"v=1; kid={kid}; sub={sub}; iss={iss}; exp={exp}; scope={scope}"

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
        Verifies the incoming token (when configured), validates required claims,
        then signs an outbound JWT and injects it as the Authorization header.

        All non-MCP call types pass through unchanged.
        """
        if call_type != "call_mcp_tool":
            return data

        # ------------------------------------------------------------------
        # FR-5: Verify incoming token before re-signing
        # ------------------------------------------------------------------
        jwt_claims: Optional[Dict[str, Any]] = None
        raw_token: Optional[str] = data.get("incoming_bearer_token")

        if self.access_token_discovery_uri and raw_token:
            # Three-dot pattern → JWT;  otherwise opaque.
            is_jwt = raw_token.count(".") == 2
            try:
                if is_jwt:
                    jwt_claims = await self._verify_incoming_jwt(raw_token)
                elif self.token_introspection_endpoint:
                    jwt_claims = await self._introspect_opaque_token(raw_token)
                else:
                    verbose_proxy_logger.warning(
                        "MCPJWTSigner: access_token_discovery_uri is set but the "
                        "incoming token appears to be opaque and no "
                        "token_introspection_endpoint is configured. "
                        "Proceeding without incoming token verification."
                    )
            except Exception as exc:
                verbose_proxy_logger.error(
                    "MCPJWTSigner: incoming token verification failed: %s", exc
                )
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": (
                            f"MCPJWTSigner: incoming token verification failed: {exc}"
                        )
                    },
                )
        elif not raw_token and self.access_token_discovery_uri:
            verbose_proxy_logger.debug(
                "MCPJWTSigner: access_token_discovery_uri configured but no Bearer "
                "token found in request (API-key auth request — skipping verification)."
            )

        # Fall back to LiteLLM-decoded JWT claims (available when proxy uses JWT auth).
        if jwt_claims is None:
            jwt_claims = getattr(user_api_key_dict, "jwt_claims", None)

        # ------------------------------------------------------------------
        # FR-15: Validate required claims
        # ------------------------------------------------------------------
        self._validate_required_claims(jwt_claims)

        # ------------------------------------------------------------------
        # Build outbound access token
        # ------------------------------------------------------------------
        claims = self._build_claims(user_api_key_dict, data, jwt_claims)

        signed_token = jwt.encode(
            claims,
            self._private_key,
            algorithm=self.ALGORITHM,
            headers={"kid": self._kid},
        )

        # Merge into existing extra_headers — a prior guardrail in the chain may
        # have already injected tracing headers or correlation IDs.
        existing_headers: Dict[str, str] = data.get("extra_headers") or {}
        new_headers: Dict[str, str] = {
            **existing_headers,
            "Authorization": f"Bearer {signed_token}",
        }

        # ------------------------------------------------------------------
        # FR-14: Two-token model — channel token
        # ------------------------------------------------------------------
        if self.channel_token_audience:
            channel_claims = self._build_channel_token_claims(claims)
            channel_token = jwt.encode(
                channel_claims,
                self._private_key,
                algorithm=self.ALGORITHM,
                headers={"kid": self._kid},
            )
            new_headers["x-mcp-channel-token"] = f"Bearer {channel_token}"

        # ------------------------------------------------------------------
        # FR-9: Debug header
        # ------------------------------------------------------------------
        if self.debug_headers:
            new_headers["x-litellm-mcp-debug"] = self._build_debug_header(
                claims, self._kid
            )

        data["extra_headers"] = new_headers

        verbose_proxy_logger.debug(
            "MCPJWTSigner: signed JWT sub=%s act=%s tool=%s exp=%d "
            "verified=%s channel=%s",
            claims.get("sub"),
            claims.get("act", {}).get("sub"),
            data.get("mcp_tool_name"),
            claims["exp"],
            jwt_claims is not None,
            bool(self.channel_token_audience),
        )

        return data

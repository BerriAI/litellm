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
          issuer: "https://my-litellm.example.com"   # optional
          audience: "mcp"                             # optional
          ttl_seconds: 300                            # optional

MCP servers verify tokens via:
  GET /.well-known/openid-configuration  → { jwks_uri: ".../.well-known/jwks.json" }
  GET /.well-known/jwks.json             → RSA public key in JWKS format

Optionally set MCP_JWT_SIGNING_KEY env var (PEM string or file:///path) to use
your own RSA keypair. If unset, an RSA-2048 keypair is auto-generated at startup.
"""

import base64
import hashlib
import os
import time
from typing import Any, Dict, Optional, Union

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
        path = key_material[len("file://") :]
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


class MCPJWTSigner(CustomGuardrail):
    """
    Built-in LiteLLM guardrail that signs outbound MCP requests with a
    LiteLLM-issued RS256 JWT, enabling zero trust authentication.

    MCP servers verify tokens using liteLLM's OIDC discovery endpoint and
    JWKS endpoint rather than trusting each upstream IdP directly.

    The signed JWT carries:
      - iss: LiteLLM issuer identifier
      - aud: MCP audience (configurable)
      - sub: End-user identity (from UserAPIKeyAuth.user_id, RFC 8693)
      - act: Actor/agent identity (team_id or org_id, RFC 8693 delegation)
      - scope: Tool-level access scopes
      - iat, exp, nbf: Timing claims
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

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

        # Register singleton so the JWKS endpoint can access it.
        global _mcp_jwt_signer_instance
        _mcp_jwt_signer_instance = self

        verbose_proxy_logger.info(
            "MCPJWTSigner initialized: issuer=%s audience=%s ttl=%ds kid=%s",
            self.issuer,
            self.audience,
            self.ttl_seconds,
            self._kid,
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_claims(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
    ) -> Dict[str, Any]:
        """
        Build JWT claims from the authenticated user context and MCP request data.
        Follows RFC 8693 (OAuth 2.0 Token Exchange) for sub/act semantics.
        """
        now = int(time.time())
        claims: Dict[str, Any] = {
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "nbf": now,
        }

        # sub: End-user identity (RFC 8693).
        # Falls back to a stable hash of the API token for service-account / anonymous
        # callers so strict JWT consumers (which require sub) always get a value.
        user_id = getattr(user_api_key_dict, "user_id", None)
        if user_id:
            claims["sub"] = user_id
        else:
            token = getattr(user_api_key_dict, "token", None) or getattr(
                user_api_key_dict, "api_key", None
            )
            if token:
                claims["sub"] = "apikey:" + hashlib.sha256(token.encode()).hexdigest()[:16]
            else:
                claims["sub"] = "litellm-proxy"

        user_email = getattr(user_api_key_dict, "user_email", None)
        if user_email:
            claims["email"] = user_email

        # act: Requester/agent identity (RFC 8693 delegation)
        team_id = getattr(user_api_key_dict, "team_id", None)
        org_id = getattr(user_api_key_dict, "org_id", None)
        act_sub = team_id or org_id or "litellm-proxy"
        claims["act"] = {"sub": act_sub}

        # end_user_id (if set separately from user_id)
        end_user_id = getattr(user_api_key_dict, "end_user_id", None)
        if end_user_id:
            claims["end_user_id"] = end_user_id

        # scope: tool-level access
        tool_name: str = data.get("mcp_tool_name", "")
        scopes = ["mcp:tools/call", "mcp:tools/list"]
        if tool_name:
            scopes.append(f"mcp:tools/{tool_name}:call")
        claims["scope"] = " ".join(scopes)

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
        """
        if call_type != "call_mcp_tool":
            return data

        claims = self._build_claims(user_api_key_dict, data)

        signed_token = jwt.encode(
            claims,
            self._private_key,
            algorithm=self.ALGORITHM,
        )

        data["extra_headers"] = {
            "Authorization": f"Bearer {signed_token}",
        }

        verbose_proxy_logger.debug(
            "MCPJWTSigner: signed JWT sub=%s act=%s tool=%s exp=%d",
            claims.get("sub"),
            claims.get("act", {}).get("sub"),
            data.get("mcp_tool_name"),
            claims["exp"],
        )

        return data

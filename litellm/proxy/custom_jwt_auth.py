"""
Custom JWT Authentication for LiteLLM Proxy

This module provides JWT validation for external authentication providers
(like Flask-Security) while maintaining compatibility with LiteLLM's 
cost tracking and logging features.

Usage:
    Configure in config.yaml:
    general_settings:
      custom_auth: custom_jwt_auth.jwt_auth
      jwt_settings:
        issuer: "https://your-auth-provider.com"
        audience: "litellm-proxy"
        public_key_url: "https://your-auth-provider.com/.well-known/jwks.json"
        user_claim_mappings:
          user_id: "sub"
          user_email: "email"
          user_role: "role"
          team_id: "team"
"""

import json
import time
from typing import Dict, Optional, Any
from urllib.parse import urljoin

import jwt
import httpx
from fastapi import Request, HTTPException
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles


class JWTConfig:
    """Configuration for JWT authentication"""
    
    def __init__(self, jwt_settings: Dict[str, Any]):
        self.issuer = jwt_settings.get("issuer")
        self.audience = jwt_settings.get("audience") 
        self.public_key_url = jwt_settings.get("public_key_url")
        self.user_claim_mappings = jwt_settings.get("user_claim_mappings", {})
        self.algorithm = jwt_settings.get("algorithm", "RS256")
        self.leeway = jwt_settings.get("leeway", 0)
        
        # Validation
        if not self.issuer:
            raise ValueError("JWT issuer is required")
        if not self.public_key_url:
            raise ValueError("JWT public_key_url is required")


class JWKSCache:
    """Simple cache for JWKS public keys with TTL"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached JWKS if not expired"""
        if url in self._cache:
            if time.time() - self._cache_time[url] < self.ttl_seconds:
                return self._cache[url]
            else:
                # Expired, remove from cache
                del self._cache[url]
                del self._cache_time[url]
        return None
    
    def set(self, url: str, jwks: Dict[str, Any]) -> None:
        """Cache JWKS with timestamp"""
        self._cache[url] = jwks
        self._cache_time[url] = time.time()


# Global JWKS cache instance
_jwks_cache = JWKSCache()


async def fetch_jwks(url: str) -> Dict[str, Any]:
    """Fetch JWKS from the given URL with caching"""
    # Check cache first
    cached_jwks = _jwks_cache.get(url)
    if cached_jwks:
        verbose_proxy_logger.debug(f"Using cached JWKS from {url}")
        return cached_jwks
    
    # Fetch from URL
    verbose_proxy_logger.debug(f"Fetching JWKS from {url}")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            jwks = response.json()
            
            # Cache the result
            _jwks_cache.set(url, jwks)
            return jwks
            
        except httpx.RequestError as e:
            verbose_proxy_logger.error(f"Failed to fetch JWKS from {url}: {e}")
            raise HTTPException(status_code=503, detail=f"Unable to fetch JWT public keys: {e}")
        except httpx.HTTPStatusError as e:
            verbose_proxy_logger.error(f"JWKS endpoint returned error {e.response.status_code}: {e}")
            raise HTTPException(status_code=503, detail=f"JWT public key endpoint error: {e.response.status_code}")


def jwk_to_public_key(jwk: Dict[str, Any]) -> Any:
    """Convert JWK to cryptography public key object"""
    if jwk.get("kty") != "RSA":
        raise ValueError(f"Unsupported key type: {jwk.get('kty')}")
    
    try:
        # Decode base64url-encoded values
        n = jwt.utils.base64url_decode(jwk["n"])
        e = jwt.utils.base64url_decode(jwk["e"])
        
        # Convert to integers
        n_int = int.from_bytes(n, byteorder="big")
        e_int = int.from_bytes(e, byteorder="big")
        
        # Create RSA public key
        public_numbers = rsa.RSAPublicNumbers(e_int, n_int)
        public_key = public_numbers.public_key()
        
        return public_key
        
    except (KeyError, ValueError) as e:
        verbose_proxy_logger.error(f"Failed to convert JWK to public key: {e}")
        raise ValueError(f"Invalid JWK format: {e}")


async def get_public_key(token_header: Dict[str, Any], config: JWTConfig) -> Any:
    """Get the public key for JWT validation"""
    kid = token_header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="JWT token missing 'kid' in header")
    
    # Fetch JWKS
    jwks = await fetch_jwks(config.public_key_url)
    
    # Find the matching key
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwk_to_public_key(key)
    
    raise HTTPException(status_code=401, detail=f"Unable to find public key with kid: {kid}")


def map_jwt_claims_to_user_auth(claims: Dict[str, Any], config: JWTConfig) -> UserAPIKeyAuth:
    """Map JWT claims to UserAPIKeyAuth object"""
    mappings = config.user_claim_mappings
    
    # Extract user information from claims
    user_id = claims.get(mappings.get("user_id", "sub"))
    user_email = claims.get(mappings.get("user_email", "email"))
    team_id = claims.get(mappings.get("team_id", "team"))
    
    # Map role to LiteLLM user role
    role_claim = claims.get(mappings.get("user_role", "role"))
    user_role = map_role_to_litellm_role(role_claim)
    
    # Create UserAPIKeyAuth object
    auth_obj = UserAPIKeyAuth(
        user_id=user_id,
        user_email=user_email,
        user_role=user_role,
        team_id=team_id,
        # Add any additional metadata
        metadata={
            "jwt_claims": claims,
            "auth_method": "jwt"
        }
    )
    
    verbose_proxy_logger.debug(f"Mapped JWT claims to user auth: user_id={user_id}, role={user_role}, team_id={team_id}")
    return auth_obj


def map_role_to_litellm_role(role_claim: Optional[str]) -> LitellmUserRoles:
    """Map external role claim to LiteLLM user role"""
    if not role_claim:
        return LitellmUserRoles.INTERNAL_USER
    
    # Role mapping - customize this based on your external provider
    role_mappings = {
        "admin": LitellmUserRoles.PROXY_ADMIN,
        "proxy_admin": LitellmUserRoles.PROXY_ADMIN,
        "user": LitellmUserRoles.INTERNAL_USER,
        "internal_user": LitellmUserRoles.INTERNAL_USER,
        "viewer": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        "internal_user_viewer": LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        "team": LitellmUserRoles.TEAM,
        "customer": LitellmUserRoles.CUSTOMER,
    }
    
    return role_mappings.get(role_claim.lower(), LitellmUserRoles.INTERNAL_USER)


async def validate_jwt_token(token: str, config: JWTConfig) -> Dict[str, Any]:
    """Validate JWT token and return claims"""
    try:
        # Decode header to get key ID
        unverified_header = jwt.get_unverified_header(token)
        
        # Get public key
        public_key = await get_public_key(unverified_header, config)
        
        # Validate token
        claims = jwt.decode(
            token,
            public_key,
            algorithms=[config.algorithm],
            issuer=config.issuer,
            audience=config.audience,
            leeway=config.leeway
        )
        
        verbose_proxy_logger.debug(f"Successfully validated JWT for user: {claims.get('sub')}")
        return claims
        
    except jwt.ExpiredSignatureError:
        verbose_proxy_logger.warning("JWT token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        verbose_proxy_logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        verbose_proxy_logger.error(f"JWT validation error: {e}")
        raise HTTPException(status_code=401, detail="Token validation failed")


async def jwt_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    """
    Main JWT authentication function for LiteLLM custom auth
    
    This function is called by LiteLLM's custom auth mechanism when
    a request comes in with a JWT token in the Authorization header.
    
    Args:
        request: FastAPI request object
        api_key: The JWT token from Authorization header (without 'Bearer ' prefix)
        
    Returns:
        UserAPIKeyAuth: Object containing user authentication information
        
    Raises:
        HTTPException: If token validation fails
    """
    # Get JWT configuration from proxy settings
    from litellm.proxy.proxy_server import general_settings
    
    jwt_settings = general_settings.get("jwt_settings")
    if not jwt_settings:
        verbose_proxy_logger.error("JWT settings not configured")
        raise HTTPException(status_code=500, detail="JWT authentication not properly configured")
    
    try:
        config = JWTConfig(jwt_settings)
    except ValueError as e:
        verbose_proxy_logger.error(f"Invalid JWT configuration: {e}")
        raise HTTPException(status_code=500, detail=f"JWT configuration error: {e}")
    
    # Check if the token looks like a JWT (has 3 parts separated by dots)
    if not api_key or api_key.count('.') != 2:
        verbose_proxy_logger.warning("Received non-JWT token in JWT auth handler")
        raise HTTPException(status_code=401, detail="Expected JWT token")
    
    # Validate the JWT token
    claims = await validate_jwt_token(api_key, config)
    
    # Map claims to UserAPIKeyAuth object
    user_auth = map_jwt_claims_to_user_auth(claims, config)
    
    verbose_proxy_logger.info(f"Successfully authenticated user {user_auth.user_id} via JWT")
    return user_auth 
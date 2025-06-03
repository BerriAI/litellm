"""
Unit tests for custom JWT authentication module
"""

import json
import pytest
import jwt
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import HTTPException, Request
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import time
import httpx

from litellm.proxy.custom_jwt_auth import (
    JWTConfig,
    JWKSCache,
    jwt_auth,
    validate_jwt_token,
    map_jwt_claims_to_user_auth,
    map_role_to_litellm_role,
    jwk_to_public_key,
    fetch_jwks,
)
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles


class TestJWTConfig:
    """Test JWT configuration validation"""
    
    def test_valid_config(self):
        """Test valid JWT configuration"""
        settings = {
            "issuer": "https://example.com",
            "audience": "litellm-proxy",
            "public_key_url": "https://example.com/.well-known/jwks.json",
            "user_claim_mappings": {"user_id": "sub"}
        }
        config = JWTConfig(settings)
        assert config.issuer == "https://example.com"
        assert config.audience == "litellm-proxy"
        assert config.algorithm == "RS256"  # default
    
    def test_missing_issuer(self):
        """Test configuration without required issuer"""
        settings = {
            "public_key_url": "https://example.com/.well-known/jwks.json"
        }
        with pytest.raises(ValueError, match="JWT issuer is required"):
            JWTConfig(settings)
    
    def test_missing_public_key_url(self):
        """Test configuration without required public key URL"""
        settings = {
            "issuer": "https://example.com"
        }
        with pytest.raises(ValueError, match="JWT public_key_url is required"):
            JWTConfig(settings)


class TestJWKSCache:
    """Test JWKS caching functionality"""
    
    def test_cache_get_set(self):
        """Test basic cache get/set functionality"""
        cache = JWKSCache(ttl_seconds=60)
        url = "https://example.com/jwks"
        jwks = {"keys": []}
        
        # Initially empty
        assert cache.get(url) is None
        
        # Set and get
        cache.set(url, jwks)
        assert cache.get(url) == jwks
    
    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        cache = JWKSCache(ttl_seconds=1)
        url = "https://example.com/jwks"
        jwks = {"keys": []}
        
        cache.set(url, jwks)
        assert cache.get(url) == jwks
        
        # Mock time passage
        with patch('time.time', return_value=time.time() + 2):
            assert cache.get(url) is None


class TestRoleMapping:
    """Test role mapping functionality"""
    
    def test_admin_role_mapping(self):
        """Test admin role mapping"""
        assert map_role_to_litellm_role("admin") == LitellmUserRoles.PROXY_ADMIN
        assert map_role_to_litellm_role("proxy_admin") == LitellmUserRoles.PROXY_ADMIN
    
    def test_user_role_mapping(self):
        """Test user role mapping"""
        assert map_role_to_litellm_role("user") == LitellmUserRoles.INTERNAL_USER
        assert map_role_to_litellm_role("internal_user") == LitellmUserRoles.INTERNAL_USER
    
    def test_viewer_role_mapping(self):
        """Test viewer role mapping"""
        assert map_role_to_litellm_role("viewer") == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        assert map_role_to_litellm_role("internal_user_viewer") == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
    
    def test_default_role_mapping(self):
        """Test default role mapping for unknown roles"""
        assert map_role_to_litellm_role("unknown") == LitellmUserRoles.INTERNAL_USER
        assert map_role_to_litellm_role(None) == LitellmUserRoles.INTERNAL_USER
        assert map_role_to_litellm_role("") == LitellmUserRoles.INTERNAL_USER


class TestClaimsMapping:
    """Test JWT claims to UserAPIKeyAuth mapping"""
    
    def test_basic_claims_mapping(self):
        """Test basic claims mapping"""
        config = JWTConfig({
            "issuer": "https://example.com",
            "public_key_url": "https://example.com/jwks",
            "user_claim_mappings": {
                "user_id": "sub",
                "user_email": "email",
                "user_role": "role",
                "team_id": "team"
            }
        })
        
        claims = {
            "sub": "user123",
            "email": "user@example.com",
            "role": "admin",
            "team": "engineering"
        }
        
        user_auth = map_jwt_claims_to_user_auth(claims, config)
        
        assert user_auth.user_id == "user123"
        assert user_auth.user_email == "user@example.com"
        assert user_auth.user_role == LitellmUserRoles.PROXY_ADMIN
        assert user_auth.team_id == "engineering"
        assert user_auth.metadata["auth_method"] == "jwt"
        assert user_auth.metadata["jwt_claims"] == claims
    
    def test_default_claim_mappings(self):
        """Test default claim mappings when not specified"""
        config = JWTConfig({
            "issuer": "https://example.com",
            "public_key_url": "https://example.com/jwks",
            "user_claim_mappings": {}
        })
        
        claims = {
            "sub": "user123",
            "email": "user@example.com",
            "role": "user"
        }
        
        user_auth = map_jwt_claims_to_user_auth(claims, config)
        
        assert user_auth.user_id == "user123"  # Default mapping: sub
        assert user_auth.user_email == "user@example.com"  # Default mapping: email
        assert user_auth.user_role == LitellmUserRoles.INTERNAL_USER


@pytest.fixture
def sample_rsa_key_pair():
    """Generate a sample RSA key pair for testing"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    
    # Convert to JWK format
    public_numbers = public_key.public_numbers()
    n = public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, 'big')
    e = public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, 'big')
    
    jwk = {
        "kty": "RSA",
        "kid": "test-key-1",
        "n": jwt.utils.base64url_encode(n).decode('ascii'),
        "e": jwt.utils.base64url_encode(e).decode('ascii'),
        "use": "sig",
        "alg": "RS256"
    }
    
    return private_key, public_key, jwk


class TestJWKConversion:
    """Test JWK to public key conversion"""
    
    def test_valid_jwk_conversion(self, sample_rsa_key_pair):
        """Test valid JWK to public key conversion"""
        private_key, expected_public_key, jwk = sample_rsa_key_pair
        
        converted_key = jwk_to_public_key(jwk)
        
        # Compare public key numbers
        expected_numbers = expected_public_key.public_numbers()
        converted_numbers = converted_key.public_numbers()
        
        assert expected_numbers.n == converted_numbers.n
        assert expected_numbers.e == converted_numbers.e
    
    def test_unsupported_key_type(self):
        """Test unsupported key type"""
        jwk = {
            "kty": "EC",  # Elliptic Curve, not RSA
            "kid": "test-key-1"
        }
        
        with pytest.raises(ValueError, match="Unsupported key type"):
            jwk_to_public_key(jwk)
    
    def test_invalid_jwk_format(self):
        """Test invalid JWK format"""
        jwk = {
            "kty": "RSA",
            "kid": "test-key-1"
            # Missing 'n' and 'e'
        }
        
        with pytest.raises(ValueError, match="Invalid JWK format"):
            jwk_to_public_key(jwk)


class TestJWTValidation:
    """Test JWT token validation"""
    
    @pytest.mark.asyncio
    async def test_valid_jwt_validation(self, sample_rsa_key_pair):
        """Test validation of a valid JWT token"""
        private_key, public_key, jwk = sample_rsa_key_pair
        
        # Create test config
        config = JWTConfig({
            "issuer": "https://example.com",
            "audience": "litellm-proxy",
            "public_key_url": "https://example.com/jwks",
            "algorithm": "RS256"
        })
        
        # Create test JWT
        claims = {
            "sub": "user123",
            "iss": "https://example.com",
            "aud": "litellm-proxy",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time())
        }
        
        token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})
        
        # Mock JWKS fetch
        with patch('litellm.proxy.custom_jwt_auth.fetch_jwks') as mock_fetch:
            mock_fetch.return_value = {"keys": [jwk]}
            
            result_claims = await validate_jwt_token(token, config)
            
            assert result_claims["sub"] == "user123"
            assert result_claims["iss"] == "https://example.com"
            assert result_claims["aud"] == "litellm-proxy"
    
    @pytest.mark.asyncio
    async def test_expired_jwt(self, sample_rsa_key_pair):
        """Test validation of an expired JWT token"""
        private_key, public_key, jwk = sample_rsa_key_pair
        
        config = JWTConfig({
            "issuer": "https://example.com",
            "audience": "litellm-proxy",
            "public_key_url": "https://example.com/jwks",
            "algorithm": "RS256"
        })
        
        # Create expired JWT
        claims = {
            "sub": "user123",
            "iss": "https://example.com",
            "aud": "litellm-proxy",
            "exp": int(time.time()) - 3600,  # Expired 1 hour ago
            "iat": int(time.time()) - 7200
        }
        
        token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})
        
        with patch('litellm.proxy.custom_jwt_auth.fetch_jwks') as mock_fetch:
            mock_fetch.return_value = {"keys": [jwk]}
            
            with pytest.raises(HTTPException) as exc_info:
                await validate_jwt_token(token, config)
            
            assert exc_info.value.status_code == 401
            assert "expired" in exc_info.value.detail.lower()


class TestMainJWTAuthFunction:
    """Test the main jwt_auth function"""
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self, sample_rsa_key_pair):
        """Test successful JWT authentication"""
        private_key, public_key, jwk = sample_rsa_key_pair
        
        # Mock general_settings
        jwt_settings = {
            "issuer": "https://example.com",
            "audience": "litellm-proxy",
            "public_key_url": "https://example.com/jwks",
            "user_claim_mappings": {
                "user_id": "sub",
                "user_email": "email",
                "user_role": "role"
            }
        }
        
        # Create test JWT
        claims = {
            "sub": "user123",
            "email": "user@example.com",
            "role": "admin",
            "iss": "https://example.com",
            "aud": "litellm-proxy",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time())
        }
        
        token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key-1"})
        
        # Mock request and dependencies
        request = Mock(spec=Request)
        
        with patch('litellm.proxy.custom_jwt_auth.fetch_jwks') as mock_fetch, \
             patch('litellm.proxy.proxy_server.general_settings', {"jwt_settings": jwt_settings}):
            
            mock_fetch.return_value = {"keys": [jwk]}
            
            result = await jwt_auth(request, token)
            
            assert isinstance(result, UserAPIKeyAuth)
            assert result.user_id == "user123"
            assert result.user_email == "user@example.com"
            assert result.user_role == LitellmUserRoles.PROXY_ADMIN
            assert result.metadata["auth_method"] == "jwt"
    
    @pytest.mark.asyncio
    async def test_missing_jwt_settings(self):
        """Test authentication failure when JWT settings are missing"""
        request = Mock(spec=Request)
        token = "invalid.jwt.token"
        
        with patch('litellm.proxy.proxy_server.general_settings', {}):
            with pytest.raises(HTTPException) as exc_info:
                await jwt_auth(request, token)
            
            assert exc_info.value.status_code == 500
            assert "not properly configured" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_invalid_token_format(self):
        """Test authentication failure with invalid token format"""
        request = Mock(spec=Request)
        token = "not-a-jwt-token"
        
        jwt_settings = {
            "issuer": "https://example.com",
            "public_key_url": "https://example.com/jwks"
        }
        
        with patch('litellm.proxy.proxy_server.general_settings', {"jwt_settings": jwt_settings}):
            with pytest.raises(HTTPException) as exc_info:
                await jwt_auth(request, token)
            
            assert exc_info.value.status_code == 401
            assert "Expected JWT token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_fetch_jwks_success():
    """Test successful JWKS fetching"""
    url = "https://example.com/jwks"
    expected_jwks = {"keys": [{"kid": "test", "kty": "RSA"}]}
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = Mock()
        mock_response.json.return_value = expected_jwks
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_client.return_value.get = AsyncMock(return_value=mock_response)
        
        result = await fetch_jwks(url)
        
        assert result == expected_jwks
        mock_client.return_value.get.assert_called_once_with(url)


@pytest.mark.asyncio
async def test_fetch_jwks_http_error():
    """Test JWKS fetching with HTTP error"""
    url = "https://example.com/jwks"
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_client.return_value.get = AsyncMock(side_effect=httpx.RequestError("Connection error"))
        
        with pytest.raises(HTTPException) as exc_info:
            await fetch_jwks(url)
        
        assert exc_info.value.status_code == 503
        assert "Unable to fetch JWT public keys" in exc_info.value.detail 
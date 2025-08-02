import json
import os
import secrets
import time
from typing import Dict, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from litellm.proxy._types import LitellmUserRoles


def check_is_admin_only_access(ui_access_mode: Union[str, Dict]) -> bool:
    """Checks ui access mode is admin_only"""
    if isinstance(ui_access_mode, str):
        return ui_access_mode == "admin_only"
    else:
        return False


def has_admin_ui_access(user_role: str) -> bool:
    """
    Check if the user has admin access to the UI.

    Returns:
        bool: True if user is 'proxy_admin' or 'proxy_admin_view_only', False otherwise.
    """

    if (
        user_role != LitellmUserRoles.PROXY_ADMIN.value
        and user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
    ):
        return False
    return True


class OAuth2TokenManager:
    """Centralized OAuth2 token response management."""
    
    @staticmethod
    def create_token_response(
        token: str, 
        expires_in: int = 86400, 
        scope: Optional[str] = None
    ) -> dict:
        """Create OAuth2 token response according to RFC 6749."""
        response = {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": expires_in
        }
        if scope:
            response["scope"] = scope
        return response
    
    @staticmethod
    def create_error_response(error: str, description: Optional[str] = None) -> dict:
        """Create OAuth2 error response according to RFC 6749."""
        response = {"error": error}
        if description:
            response["error_description"] = description
        return response
    
    @staticmethod
    async def generate_oauth_session_key(
        user_id: str,
        user_email: Optional[str] = None,
        user_role: Optional[str] = None,
        team_id: Optional[str] = None,
        client_type: str = "oauth_external_app",
        expires_in: int = 86400,
        scopes: Optional[list] = None,
        redirect_uri: Optional[str] = None
    ) -> str:
        """Generate a new session-specific API key for OAuth flows."""
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            generate_key_helper_fn,
        )
        
        # Create session metadata
        session_metadata = {
            "oauth_flow": True,
            "client_type": client_type,
            "created_at": time.time(),
            "session_type": "oauth_external_app",
            "expires_at": time.time() + expires_in
        }
        
        if scopes:
            session_metadata["scopes"] = scopes
        if redirect_uri:
            session_metadata["redirect_uri"] = redirect_uri
        
        # Generate session-specific key alias
        session_id = secrets.token_urlsafe(8)
        key_alias = f"oauth_session_{session_id}_{int(time.time())}"
        
        # Calculate duration string for expires_in
        duration_hours = expires_in // 3600
        duration = f"{duration_hours}h" if duration_hours > 0 else "24h"
        
        # Generate the session key
        response = await generate_key_helper_fn(
            request_type="key",
            duration=duration,
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            team_id=team_id,
            key_alias=key_alias,
            metadata=session_metadata,
            table_name="key"
        )
        
        return response["token"]


class OAuth2StateManager:
    """Secure OAuth2 state parameter management."""
    
    STATE_EXPIRY_SECONDS = 300  # 5 minutes
    
    @staticmethod
    def generate_secure_state(flow_type: str, redirect_uri: Optional[str] = None) -> str:
        """Generate a secure, timestamped OAuth2 state parameter."""
        state_data = {
            "flow": flow_type,
            "timestamp": int(time.time()),
            "nonce": secrets.token_urlsafe(16)
        }
        if redirect_uri:
            state_data["redirect_uri"] = redirect_uri
        return f"oauth:{json.dumps(state_data)}"
    
    @staticmethod
    def validate_state(state: str) -> bool:
        """Validate OAuth2 state parameter for security and expiry."""
        try:
            if not state.startswith("oauth:"):
                return False
            
            state_data = json.loads(state[6:])
            
            # Check required fields
            if not all(key in state_data for key in ["flow", "timestamp", "nonce"]):
                return False
            
            # Check expiry
            state_age = time.time() - state_data["timestamp"]
            if state_age > OAuth2StateManager.STATE_EXPIRY_SECONDS:
                return False
            
            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    @staticmethod
    def extract_flow_type(state: str) -> Optional[str]:
        """Extract flow type from valid state parameter."""
        try:
            if not state.startswith("oauth:"):
                return None
            
            state_data = json.loads(state[6:])
            return state_data.get("flow")
        except (json.JSONDecodeError, KeyError):
            return None
    
    @staticmethod
    def extract_redirect_uri(state: str) -> Optional[str]:
        """Extract redirect_uri from valid state parameter."""
        try:
            if not state.startswith("oauth:"):
                return None
            
            state_data = json.loads(state[6:])
            return state_data.get("redirect_uri")
        except (json.JSONDecodeError, KeyError):
            return None


class OAuth2URLManager:
    """OAuth2 URL construction and parsing utilities."""
    
    @staticmethod
    def build_redirect_url(base_url: str, route: str, params: dict) -> str:
        """Build OAuth2 redirect URL with parameters."""
        parsed_url = urlparse(base_url)
        new_path = parsed_url.path.rstrip('/') + route
        
        return urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            new_path,
            parsed_url.params,
            urlencode(params, doseq=True),
            parsed_url.fragment
        ))
    
    @staticmethod
    def parse_oauth_callback(url: str) -> dict:
        """Parse OAuth2 callback URL and extract parameters."""
        parsed_url = urlparse(url)
        return parse_qs(parsed_url.query)
    
    @staticmethod
    def modify_url_for_oauth_flow(request_url: str) -> str:
        """Modify URL to redirect to OAuth flow (remove response_type, add oauth_flow, preserve redirect_uri)."""
        parsed_url = urlparse(request_url)
        query_params = parse_qs(parsed_url.query)
        
        # Remove response_type from query params
        query_params.pop('response_type', None)
        
        # Add OAuth flow indicator
        query_params['oauth_flow'] = ['true']
        
        # Preserve redirect_uri if present (needed for VSCode extension compatibility)
        # redirect_uri is automatically preserved since we're not removing it
        
        # Rebuild URL with /sso/login path
        new_path = parsed_url.path.replace("/sso/key/generate", "/sso/login")
        
        return urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            new_path,
            parsed_url.params,
            urlencode(query_params, doseq=True),
            parsed_url.fragment
        ))
    
    @staticmethod
    def build_callback_redirect_url(redirect_uri: str, access_token: str, **kwargs) -> str:
        """Build callback redirect URL with access_token for VSCode extension compatibility."""
        parsed_url = urlparse(redirect_uri)
        query_params = parse_qs(parsed_url.query)
        
        # Add access_token to query parameters
        query_params['access_token'] = [access_token]
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                query_params[key] = [str(value)]
        
        return urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            urlencode(query_params, doseq=True),
            parsed_url.fragment
        ))
    
    @staticmethod
    def validate_redirect_uri(redirect_uri: str, allowed_schemes: Optional[list] = None) -> bool:
        """Validate redirect_uri for security purposes."""
        if not redirect_uri:
            return False
            
        try:
            parsed = urlparse(redirect_uri)
            
            # Default allowed schemes for IDE extensions and common OAuth
            if allowed_schemes is None:
                # Start with default schemes
                allowed_schemes = [
                    # Web protocols
                    'http', 'https',
                    # VSCode family
                    'vscode', 'vscode-insiders',
                    # Other popular IDEs/editors
                    'cursor',           # Cursor editor
                    'vscodium',         # VSCodium (open source VSCode)
                    'code-oss',         # VSCode OSS
                    'fleet',            # JetBrains Fleet
                    'zed',              # Zed editor
                    'sublime',          # Sublime Text
                    'atom',             # Atom editor
                    'brackets',         # Adobe Brackets
                    'nova',             # Nova editor
                    'coderunner',       # CodeRunner
                    'textmate',         # TextMate
                    # Development tools
                    'github-desktop',   # GitHub Desktop
                    'sourcetree',       # Sourcetree
                    'tower',            # Tower Git
                    'fork',             # Fork Git
                ]
                
                # Add custom schemes from environment variable
                custom_schemes = os.getenv("OAUTH_ALLOWED_REDIRECT_SCHEMES", "")
                if custom_schemes:
                    additional_schemes = [scheme.strip() for scheme in custom_schemes.split(",") if scheme.strip()]
                    allowed_schemes.extend(additional_schemes)
            
            # Check if scheme is allowed
            if parsed.scheme not in allowed_schemes:
                return False
            
            # For IDE/editor schemes, ensure it's a proper extension callback
            ide_schemes = ['vscode', 'vscode-insiders', 'cursor', 'vscodium', 'code-oss', 
                          'fleet', 'zed', 'sublime', 'atom', 'brackets', 'nova']
            if parsed.scheme in ide_schemes:
                # IDE URIs should have format: scheme://extension-id/callback-path
                return len(parsed.netloc) > 0 and len(parsed.path) > 0
            
            # For HTTP(S), ensure it's a valid URL
            if parsed.scheme in ['http', 'https']:
                return bool(parsed.netloc)
            
            # For other development tools, basic validation
            return bool(parsed.netloc or parsed.path)
            
        except Exception:
            return False


class OAuth2CORSHandler:
    """Centralized CORS header management for OAuth2 responses."""
    
    @staticmethod
    def get_oauth_cors_headers() -> dict:
        """Get CORS headers for OAuth2 responses."""
        return {
            "Access-Control-Allow-Origin": os.getenv("OAUTH_ALLOWED_ORIGINS", "*"),
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Cache-Control": "no-store",
            "Pragma": "no-cache"
        }

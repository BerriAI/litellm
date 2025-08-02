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


class OAuth2StateManager:
    """Secure OAuth2 state parameter management."""
    
    STATE_EXPIRY_SECONDS = 300  # 5 minutes
    
    @staticmethod
    def generate_secure_state(flow_type: str) -> str:
        """Generate a secure, timestamped OAuth2 state parameter."""
        state_data = {
            "flow": flow_type,
            "timestamp": int(time.time()),
            "nonce": secrets.token_urlsafe(16)
        }
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
        """Modify URL to redirect to OAuth flow (remove response_type, add oauth_flow)."""
        parsed_url = urlparse(request_url)
        query_params = parse_qs(parsed_url.query)
        
        # Remove response_type from query params
        query_params.pop('response_type', None)
        
        # Add OAuth flow indicator
        query_params['oauth_flow'] = ['true']
        
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

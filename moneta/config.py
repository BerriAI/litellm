"""
Configuration management for Lago billing integration.

This module handles loading and validation of environment variables required
for the Lago billing integration with LiteLLM.
"""

import os
from typing import Optional


class LagoConfig:
    """Configuration manager for Lago billing integration"""
    
    def __init__(self):
        """Load and validate configuration from environment variables"""
        # Required configuration
        self.api_base = os.getenv("LAGO_API_BASE")
        self.api_key = os.getenv("LAGO_API_KEY")
        self.publisher_id = os.getenv("LAGO_PUBLISHER_ID")
        
        # Optional configuration with defaults
        self.timeout = int(os.getenv("LAGO_TIMEOUT", "5"))
        self.fallback_allow = os.getenv("LAGO_FALLBACK_ALLOW", "true").lower() == "true"
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that required configuration is present"""
        missing = []
        
        if not self.api_base:
            missing.append("LAGO_API_BASE")
        if not self.api_key:
            missing.append("LAGO_API_KEY")
        if not self.publisher_id:
            missing.append("LAGO_PUBLISHER_ID")
        
        if missing:
            raise ValueError(
                f"Missing required Lago configuration: {', '.join(missing)}. "
                f"Please set these environment variables."
            )
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(self.api_base and self.api_key and self.publisher_id)
    
    def get_auth_headers(self) -> dict:
        """Get authorization headers for Lago API calls"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_entitlement_url(self) -> str:
        """Get the entitlement authorization endpoint URL"""
        return f"{self.api_base}/v1/entitlement/authorize"
    
    def get_events_url(self) -> str:
        """Get the usage events endpoint URL"""
        return f"{self.api_base}/api/v1/events"
    
    def __str__(self) -> str:
        """String representation for debugging (without sensitive data)"""
        return (
            f"LagoConfig(api_base={self.api_base}, "
            f"publisher_id={self.publisher_id}, "
            f"timeout={self.timeout}, "
            f"fallback_allow={self.fallback_allow})"
        )

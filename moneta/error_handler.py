"""
Error handling for Lago billing integration.

This module provides simplified error handling with sensible defaults
for the Lago billing integration, following the 5-scenario error matrix
from the technical design.
"""

from typing import Optional
import logging

# Set up logger for error handling
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Minimal error handling with sensible defaults"""
    
    @staticmethod
    def handle_authorization_error(error: Exception, fallback_allow: bool) -> bool:
        """
        Handle authorization error with configurable fallback.
        
        Args:
            error: The exception that occurred during authorization
            fallback_allow: Whether to allow requests when authorization fails
            
        Returns:
            Boolean indicating whether to allow the request
        """
        logger.warning(f"Authorization error: {error}")
        print(f"Authorization error: {error}")
        return fallback_allow
    
    @staticmethod
    def handle_usage_error(error: Exception) -> None:
        """
        Handle usage reporting error.
        
        Args:
            error: The exception that occurred during usage reporting
        """
        logger.error(f"Usage reporting error: {error}")
        print(f"Usage reporting error: {error}")
        # Just log and continue - don't block the main request flow
    
    @staticmethod
    def handle_missing_customer_id(call_id: Optional[str] = None) -> None:
        """
        Handle missing customer ID scenario.
        
        Args:
            call_id: The call ID if available
        """
        message = f"Missing customer ID for call {call_id}" if call_id else "Missing customer ID"
        logger.warning(message)
        print(f"Warning: {message}, allowing request")
    
    @staticmethod
    def handle_storage_error(error: Exception, operation: str) -> None:
        """
        Handle storage-related errors.
        
        Args:
            error: The exception that occurred
            operation: Description of the storage operation that failed
        """
        logger.warning(f"Storage error during {operation}: {error}")
        print(f"Storage error during {operation}: {error}")
        # Continue execution - storage errors shouldn't block requests
    
    @staticmethod
    def handle_config_error(error: Exception) -> None:
        """
        Handle configuration errors.
        
        Args:
            error: The configuration error that occurred
        """
        logger.error(f"Configuration error: {error}")
        print(f"Configuration error: {error}")
        # Configuration errors are critical and should be raised
        raise error


class ErrorScenarios:
    """
    Error scenario constants matching the 5-scenario error handling matrix
    from the technical design document.
    """
    
    AUTHORIZATION_TIMEOUT = "authorization_timeout_error"
    AUTHORIZATION_DENIED = "authorization_denied"
    MISSING_CUSTOMER_ID = "missing_customer_id"
    USAGE_REPORTING_FAILURE = "usage_reporting_failure"
    STORAGE_CLEANUP_FAILURE = "storage_cleanup_failure"

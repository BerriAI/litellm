"""
Moneta Lago Billing Integration for LiteLLM

This package provides streamlined integration between LiteLLM and Lago billing system,
enabling pre-call entitlement checking and post-call usage reporting for accurate
billing and usage tracking.

Key Components:
- LagoLogger: Main integration class extending LiteLLM's CustomLogger
- CallDataStore: Thread-safe storage for call metadata
- LagoConfig: Configuration management for environment variables
- ErrorHandler: Simplified error handling with sensible defaults

Usage:
    from litellm.moneta import LagoLogger
    import litellm
    
    # Initialize and register the logger
    lago_logger = LagoLogger()
    litellm.callbacks = [lago_logger]

Environment Variables Required:
    LAGO_API_BASE: Base URL for Lago API (e.g., https://your-lago-instance.com)
    LAGO_API_KEY: API key for Lago authentication
    LAGO_PUBLISHER_ID: Publisher ID for entitlement checks

Optional Environment Variables:
    LAGO_TIMEOUT: Request timeout in seconds (default: 5)
    LAGO_FALLBACK_ALLOW: Allow requests on authorization errors (default: true)
"""

from .lago_logger import LagoLogger
from .call_data_store import CallDataStore
from .config import LagoConfig
from .error_handler import ErrorHandler, ErrorScenarios
from .monitoring import MonitoringService

# Package metadata
__version__ = "1.0.0"
__author__ = "Moneta Team"
__description__ = "Streamlined Lago billing integration for LiteLLM"

# Main exports
__all__ = [
    "LagoLogger",
    "CallDataStore",
    "LagoConfig",
    "ErrorHandler",
    "ErrorScenarios",
    "MonitoringService"
]

# Convenience function for quick setup
def create_lago_logger() -> LagoLogger:
    """
    Create and return a configured LagoLogger instance.
    
    This is a convenience function that handles the initialization
    and basic error checking for the LagoLogger.
    
    Returns:
        LagoLogger: Configured logger instance ready for use
        
    Raises:
        ValueError: If required configuration is missing
    """
    try:
        return LagoLogger()
    except ValueError as e:
        print(f"Failed to create LagoLogger: {e}")
        print("Please ensure LAGO_API_BASE, LAGO_API_KEY, and LAGO_PUBLISHER_ID are set")
        raise


# Quick setup function for LiteLLM integration
def setup_lago_billing() -> LagoLogger:
    """
    Set up Lago billing integration with LiteLLM.
    
    This function creates a LagoLogger instance and provides guidance
    on registering it with LiteLLM callbacks.
    
    Returns:
        LagoLogger: Configured logger instance
        
    Example:
        import litellm
        from litellm.moneta import setup_lago_billing
        
        lago_logger = setup_lago_billing()
        litellm.callbacks = [lago_logger]
    """
    logger = create_lago_logger()
    
    print("LagoLogger created successfully!")
    print("To complete setup, register with LiteLLM:")
    print("  import litellm")
    print("  litellm.callbacks = [lago_logger]")
    
    return logger

"""Test that loggers respect LITELLM_LOG environment variable"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


def test_logger_level_respects_env_var(monkeypatch):
    """
    Test that loggers are initialized with the correct level based on LITELLM_LOG env var.
    This verifies the fix for issue #9815 where loggers ignored the LITELLM_LOG setting.
    """
    # Test different log levels
    test_cases = [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ]
    
    for env_level, expected_numeric_level in test_cases:
        # Set the environment variable
        monkeypatch.setenv("LITELLM_LOG", env_level)
        
        # Re-import the logging module to pick up the new env var
        import importlib
        import litellm._logging
        importlib.reload(litellm._logging)
        
        # Check that all loggers have the correct level
        assert litellm._logging.verbose_logger.level == expected_numeric_level, \
            f"verbose_logger level should be {expected_numeric_level} for LITELLM_LOG={env_level}"
        
        assert litellm._logging.verbose_proxy_logger.level == expected_numeric_level, \
            f"verbose_proxy_logger level should be {expected_numeric_level} for LITELLM_LOG={env_level}"
        
        assert litellm._logging.verbose_router_logger.level == expected_numeric_level, \
            f"verbose_router_logger level should be {expected_numeric_level} for LITELLM_LOG={env_level}"


def test_handler_and_logger_levels_match(monkeypatch):
    """
    Test that both handler and logger levels are set to the same value.
    This ensures consistent behavior across the logging system.
    """
    monkeypatch.setenv("LITELLM_LOG", "WARNING")
    
    # Re-import to apply settings
    import importlib
    import litellm._logging
    importlib.reload(litellm._logging)
    
    # Get the handler level (first handler should be our configured one)
    handler_level = litellm._logging.handler.level
    
    # All loggers should have the same level as the handler
    assert litellm._logging.verbose_logger.level == handler_level
    assert litellm._logging.verbose_proxy_logger.level == handler_level
    assert litellm._logging.verbose_router_logger.level == handler_level
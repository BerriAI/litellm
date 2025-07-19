import sys
import pytest
import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type


def test_exception_debug_info_goes_to_stderr(caplog):
    """Test that debug info from exception mapping is logged as warnings."""
    # Save original state
    original_suppress_debug_info = litellm.suppress_debug_info
    
    try:
        # Ensure debug info is not suppressed
        litellm.suppress_debug_info = False
        
        # Create a mock exception
        original_exception = Exception("Test exception")
        
        # Call the function that should log warnings
        try:
            exception_type(
                model="test-model",
                original_exception=original_exception,
                custom_llm_provider="openai"
            )
        except:
            pass  # We expect this to raise an exception
        
        # Check that the expected messages were logged as warnings
        log_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
        
        # Check that the expected messages are in the log
        assert any("Give Feedback / Get Help" in msg for msg in log_messages), "Expected message not found in logs"
        assert any("https://github.com/BerriAI/litellm/issues/new" in msg for msg in log_messages)
        assert any("LiteLLM.Info: If you need to debug this error" in msg for msg in log_messages)
        
    finally:
        # Restore original state
        litellm.suppress_debug_info = original_suppress_debug_info


def test_exception_debug_info_suppressed(caplog):
    """Test that debug info is not logged when suppress_debug_info is True."""
    # Save original state
    original_suppress_debug_info = litellm.suppress_debug_info
    
    try:
        # Suppress debug info
        litellm.suppress_debug_info = True
        
        # Create a mock exception
        original_exception = Exception("Test exception")
        
        # Call the function
        try:
            exception_type(
                model="test-model",
                original_exception=original_exception,
                custom_llm_provider="openai"
            )
        except:
            pass  # We expect this to raise an exception
        
        # Check that no warning messages were logged
        log_messages = [record.message for record in caplog.records if record.levelname == "WARNING"]
        
        # Check that debug info was not logged when suppressed
        assert not any("Give Feedback / Get Help" in msg for msg in log_messages), "Debug info should not be logged when suppressed"
        
    finally:
        # Restore original state
        litellm.suppress_debug_info = original_suppress_debug_info
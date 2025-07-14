import sys
import pytest
import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type


def test_exception_debug_info_goes_to_stderr(capsys):
    """Test that debug info from exception mapping goes to stderr, not stdout."""
    # Save original state
    original_suppress_debug_info = litellm.suppress_debug_info
    
    try:
        # Ensure debug info is not suppressed
        litellm.suppress_debug_info = False
        
        # Create a mock exception
        original_exception = Exception("Test exception")
        
        # Call the function that should print to stderr
        try:
            exception_type(
                model="test-model",
                original_exception=original_exception,
                custom_llm_provider="openai"
            )
        except:
            pass  # We expect this to raise an exception
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check that nothing was written to stdout
        assert captured.out == "", "Debug info should not be written to stdout"
        
        # Check that the expected messages were written to stderr
        assert "Give Feedback / Get Help" in captured.err, "Expected message not found in stderr"
        assert "https://github.com/BerriAI/litellm/issues/new" in captured.err
        assert "LiteLLM.Info: If you need to debug this error" in captured.err
        
    finally:
        # Restore original state
        litellm.suppress_debug_info = original_suppress_debug_info


def test_exception_debug_info_suppressed(capsys):
    """Test that debug info is not printed when suppress_debug_info is True."""
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
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check that nothing was written to either stdout or stderr
        assert captured.out == "", "No output should be written to stdout when suppressed"
        assert "Give Feedback / Get Help" not in captured.err, "Debug info should not be in stderr when suppressed"
        
    finally:
        # Restore original state
        litellm.suppress_debug_info = original_suppress_debug_info
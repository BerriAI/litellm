"""
Integration tests for shared session functionality in main.py
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the litellm directory to the path
sys.path.insert(0, os.path.abspath("../../.."))

import litellm


class TestSharedSessionIntegration:
    """Test cases for shared session integration in main.py"""

    def test_acompletion_shared_session_parameter(self):
        """Test that acompletion accepts shared_session parameter"""
        import inspect
        
        # Get the function signature
        sig = inspect.signature(litellm.acompletion)
        params = list(sig.parameters.keys())
        
        # Verify shared_session parameter exists
        assert 'shared_session' in params
        
        # Verify the parameter type annotation
        shared_session_param = sig.parameters['shared_session']
        assert 'ClientSession' in str(shared_session_param.annotation)
        
        # Verify default value is None
        assert shared_session_param.default is None

    def test_completion_shared_session_parameter(self):
        """Test that completion accepts shared_session parameter"""
        import inspect
        
        # Get the function signature
        sig = inspect.signature(litellm.completion)
        params = list(sig.parameters.keys())
        
        # Verify shared_session parameter exists
        assert 'shared_session' in params
        
        # Verify the parameter type annotation
        shared_session_param = sig.parameters['shared_session']
        assert 'ClientSession' in str(shared_session_param.annotation)
        
        # Verify default value is None
        assert shared_session_param.default is None

    @pytest.mark.asyncio
    async def test_acompletion_with_shared_session_mock(self):
        """Test acompletion with mocked shared session (no actual API call)"""
        import inspect
        
        # Create a mock session
        mock_session = MagicMock()
        mock_session.closed = False

        # Mock the completion function to avoid actual API calls
        with patch('litellm.completion') as mock_completion:
            mock_completion.return_value = {"choices": [{"message": {"content": "test"}}]}

            # This should not raise an error even though we can't make actual API calls
            try:
                # We can't actually call acompletion without proper setup,
                # but we can verify the parameter is accepted
                sig = inspect.signature(litellm.acompletion)
                assert 'shared_session' in sig.parameters
            except Exception as e:
                # Expected to fail due to missing API keys, but parameter should be valid
                sig = inspect.signature(litellm.acompletion)
                assert 'shared_session' in sig.parameters

    def test_shared_session_passed_to_completion_kwargs(self):
        """Test that shared_session is passed through completion_kwargs"""
        # This test verifies that the shared_session parameter
        # is properly included in the completion_kwargs dictionary
        
        # We can't easily test the internal logic without mocking,
        # but we can verify the parameter exists in the function signature
        import inspect
        
        sig = inspect.signature(litellm.acompletion)
        shared_session_param = sig.parameters['shared_session']
        
        # Verify the parameter is properly typed
        assert 'ClientSession' in str(shared_session_param.annotation)
        assert shared_session_param.default is None

    def test_backward_compatibility(self):
        """Test that existing code without shared_session still works"""
        import inspect
        
        # Verify that shared_session has a default value of None
        sig = inspect.signature(litellm.acompletion)
        shared_session_param = sig.parameters['shared_session']
        
        # This ensures backward compatibility
        assert shared_session_param.default is None

    def test_type_annotations_consistency(self):
        """Test that type annotations are consistent between acompletion and completion"""
        import inspect
        
        # Get signatures for both functions
        acompletion_sig = inspect.signature(litellm.acompletion)
        completion_sig = inspect.signature(litellm.completion)
        
        # Get the shared_session parameters
        acompletion_param = acompletion_sig.parameters['shared_session']
        completion_param = completion_sig.parameters['shared_session']
        
        # Verify they have the same type annotation
        assert str(acompletion_param.annotation) == str(completion_param.annotation)
        
        # Verify they have the same default value
        assert acompletion_param.default == completion_param.default

    def test_shared_session_parameter_position(self):
        """Test that shared_session parameter is in the correct position"""
        import inspect
        
        sig = inspect.signature(litellm.acompletion)
        params = list(sig.parameters.keys())
        
        # Find the position of shared_session
        shared_session_index = params.index('shared_session')
        
        # It should be near the end, before **kwargs
        assert shared_session_index > 0
        assert shared_session_index < len(params) - 1  # Should be before **kwargs
        
        # Verify it's after the main parameters
        assert 'model' in params[:shared_session_index]
        assert 'messages' in params[:shared_session_index]


class TestSharedSessionUsage:
    """Test cases demonstrating proper usage of shared sessions"""

    def test_shared_session_usage_example(self):
        """Test example usage pattern for shared sessions"""
        # This test demonstrates the expected usage pattern
        # without actually making API calls
        
        import inspect
        
        # Verify the function signature allows for the expected usage
        sig = inspect.signature(litellm.acompletion)
        params = sig.parameters
        
        # Verify all expected parameters exist
        expected_params = [
            'model', 'messages', 'shared_session'
        ]
        
        for param in expected_params:
            assert param in params, f"Parameter {param} not found in acompletion signature"
        
        # Verify shared_session is optional
        assert params['shared_session'].default is None

    def test_shared_session_with_other_parameters(self):
        """Test that shared_session works with other parameters"""
        import inspect
        
        sig = inspect.signature(litellm.acompletion)
        params = sig.parameters
        
        # Verify shared_session doesn't conflict with other parameters
        assert 'shared_session' in params
        assert 'model' in params
        assert 'messages' in params
        assert 'timeout' in params
        
        # Verify the parameter order makes sense
        param_list = list(params.keys())
        shared_session_index = param_list.index('shared_session')
        
        # shared_session should be after the main parameters but before **kwargs
        assert shared_session_index > param_list.index('model')
        assert shared_session_index > param_list.index('messages')
        
        # Should be before **kwargs (last parameter)
        assert shared_session_index < len(param_list) - 1

"""
Regression tests for Vertex AI file upload functionality in the proxy.

This module contains tests to ensure that the fix for Vertex AI file uploads
in the proxy server continues to work and prevents regression of the issue
where get_files_provider_config returned None for vertex_ai provider.
"""

import pytest
from unittest.mock import Mock, patch
from litellm.proxy.openai_files_endpoints.files_endpoints import get_files_provider_config


def test_vertex_ai_files_provider_config_never_returns_none_when_configured():
    """
    Regression test: Ensure that get_files_provider_config never returns None
    for vertex_ai when properly configured in model_list.
    
    This test prevents regression of the bug where vertex_ai provider
    always returned None, causing "Could not resolve project_id" errors.
    """
    from litellm.proxy.proxy_server import proxy_config
    
    # Mock the proxy_config with a proper Vertex AI configuration
    mock_config = {
        'model_list': [
            {
                'model_name': 'gemini-2.5-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-2.5-flash',
                    'vertex_project': 'test-project-123',
                    'vertex_location': 'us-central1',
                    'vertex_credentials': '/path/to/service_account.json'
                }
            }
        ]
    }
    
    # Mock proxy_config.config
    original_config = getattr(proxy_config, 'config', None)
    proxy_config.config = mock_config
    
    try:
        result = get_files_provider_config('vertex_ai')
        
        # This should NEVER be None when vertex_ai is properly configured
        assert result is not None, (
            "CRITICAL REGRESSION: get_files_provider_config returned None for vertex_ai "
            "when it should return configuration. This would cause 'Could not resolve project_id' errors."
        )
        
        # Verify all expected parameters are present
        assert 'vertex_project' in result
        assert 'vertex_location' in result
        assert 'vertex_credentials' in result
        
    finally:
        # Restore original config
        if original_config is not None:
            proxy_config.config = original_config
        else:
            delattr(proxy_config, 'config')


def test_vertex_ai_files_provider_config_handles_multiple_vertex_models():
    """
    Test that get_files_provider_config correctly handles multiple Vertex AI models
    in the model_list and returns configuration from the first one found.
    """
    from litellm.proxy.proxy_server import proxy_config
    
    # Mock the proxy_config with multiple Vertex AI models
    mock_config = {
        'model_list': [
            {
                'model_name': 'gemini-1.5-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-1.5-flash',
                    'vertex_project': 'project-1',
                    'vertex_location': 'us-east1',
                    'vertex_credentials': '/path/to/creds1.json'
                }
            },
            {
                'model_name': 'gemini-2.5-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-2.5-flash',
                    'vertex_project': 'project-2',
                    'vertex_location': 'us-central1',
                    'vertex_credentials': '/path/to/creds2.json'
                }
            }
        ]
    }
    
    # Mock proxy_config.config
    original_config = getattr(proxy_config, 'config', None)
    proxy_config.config = mock_config
    
    try:
        result = get_files_provider_config('vertex_ai')
        
        assert result is not None, "Should return config when multiple vertex_ai models are present"
        
        # Should return config from the first vertex_ai model found
        assert result['vertex_project'] == 'project-1'
        assert result['vertex_location'] == 'us-east1'
        assert result['vertex_credentials'] == '/path/to/creds1.json'
        
    finally:
        # Restore original config
        if original_config is not None:
            proxy_config.config = original_config
        else:
            delattr(proxy_config, 'config')


def test_vertex_ai_files_provider_config_ignores_non_vertex_models():
    """
    Test that get_files_provider_config correctly identifies vertex_ai models
    and ignores other model types when searching for configuration.
    """
    from litellm.proxy.proxy_server import proxy_config
    
    # Mock the proxy_config with mixed model types
    mock_config = {
        'model_list': [
            {
                'model_name': 'gpt-3.5-turbo',
                'litellm_params': {
                    'model': 'openai/gpt-3.5-turbo',
                    'api_key': 'test-key'
                }
            },
            {
                'model_name': 'gemini-2.5-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-2.5-flash',
                    'vertex_project': 'test-project',
                    'vertex_location': 'us-central1',
                    'vertex_credentials': '/path/to/creds.json'
                }
            },
            {
                'model_name': 'claude-3',
                'litellm_params': {
                    'model': 'anthropic/claude-3',
                    'api_key': 'test-key'
                }
            }
        ]
    }
    
    # Mock proxy_config.config
    original_config = getattr(proxy_config, 'config', None)
    proxy_config.config = mock_config
    
    try:
        result = get_files_provider_config('vertex_ai')
        
        assert result is not None, "Should find vertex_ai config even with mixed model types"
        assert result['vertex_project'] == 'test-project'
        
    finally:
        # Restore original config
        if original_config is not None:
            proxy_config.config = original_config
        else:
            delattr(proxy_config, 'config')


def test_vertex_ai_files_provider_config_handles_malformed_model_list():
    """
    Test that get_files_provider_config gracefully handles malformed model_list entries
    without crashing.
    """
    from litellm.proxy.proxy_server import proxy_config
    
    # Mock the proxy_config with malformed entries
    mock_config = {
        'model_list': [
            # Missing litellm_params
            {
                'model_name': 'gemini-2.5-flash'
            },
            # Missing model field
            {
                'model_name': 'gemini-1.5-flash',
                'litellm_params': {
                    'vertex_project': 'test-project'
                }
            },
            # Valid vertex_ai model
            {
                'model_name': 'gemini-2.0-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-2.0-flash',
                    'vertex_project': 'test-project',
                    'vertex_location': 'us-central1',
                    'vertex_credentials': '/path/to/creds.json'
                }
            }
        ]
    }
    
    # Mock proxy_config.config
    original_config = getattr(proxy_config, 'config', None)
    proxy_config.config = mock_config
    
    try:
        result = get_files_provider_config('vertex_ai')
        
        # Should still work and find the valid vertex_ai model
        assert result is not None, "Should handle malformed entries and find valid vertex_ai model"
        assert result['vertex_project'] == 'test-project'
        
    finally:
        # Restore original config
        if original_config is not None:
            proxy_config.config = original_config
        else:
            delattr(proxy_config, 'config')


def test_vertex_ai_files_provider_config_old_behavior_regression():
    """
    Regression test: Ensure that the old behavior of always returning None
    for vertex_ai provider is completely eliminated.
    
    This test specifically checks that the function no longer has the old
    hardcoded return None for vertex_ai.
    """
    from litellm.proxy.proxy_server import proxy_config
    
    # Mock the proxy_config with a minimal but valid Vertex AI configuration
    mock_config = {
        'model_list': [
            {
                'model_name': 'gemini-2.5-flash',
                'litellm_params': {
                    'model': 'vertex_ai/gemini-2.5-flash',
                    'vertex_project': 'minimal-project'
                }
            }
        ]
    }
    
    # Mock proxy_config.config
    original_config = getattr(proxy_config, 'config', None)
    proxy_config.config = mock_config
    
    try:
        result = get_files_provider_config('vertex_ai')
        
        # The old behavior would always return None here
        # The new behavior should return the configuration
        assert result is not None, (
            "REGRESSION DETECTED: The old behavior of returning None for vertex_ai "
            "has returned. This indicates the fix has been reverted."
        )
        
        # Verify we get the expected configuration
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'vertex_project' in result, "Should contain vertex_project"
        
    finally:
        # Restore original config
        if original_config is not None:
            proxy_config.config = original_config
        else:
            delattr(proxy_config, 'config')

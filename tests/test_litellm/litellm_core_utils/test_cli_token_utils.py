"""
Unit tests for CLI token utilities
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from litellm.litellm_core_utils.cli_token_utils import get_litellm_gateway_api_key


class TestCLITokenUtils:
    """Test CLI token utility functions"""

    def test_get_litellm_gateway_api_key_success(self):
        """Test getting CLI API key when token file exists and is valid"""
        token_data = {
            'key': 'sk-test-cli-key-123',
            'user_id': 'test-user',
            'user_email': 'test@example.com',
            'timestamp': 1234567890
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(token_data))), \
             patch('litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path', return_value='/test/.litellm/token.json'):
            
            result = get_litellm_gateway_api_key()
            
            assert result == 'sk-test-cli-key-123'

    def test_get_litellm_gateway_api_key_no_file(self):
        """Test getting CLI API key when token file doesn't exist"""
        with patch('os.path.exists', return_value=False), \
             patch('litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path', return_value='/test/.litellm/token.json'):
            
            result = get_litellm_gateway_api_key()
            
            assert result is None

    def test_get_litellm_gateway_api_key_invalid_json(self):
        """Test getting CLI API key when token file has invalid JSON"""
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='invalid json')), \
             patch('litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path', return_value='/test/.litellm/token.json'):
            
            result = get_litellm_gateway_api_key()
            
            assert result is None

    def test_get_litellm_gateway_api_key_no_key_field(self):
        """Test getting CLI API key when token file exists but has no key field"""
        token_data = {
            'user_id': 'test-user',
            'user_email': 'test@example.com'
            # Missing 'key' field
        }
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(token_data))), \
             patch('litellm.litellm_core_utils.cli_token_utils.get_cli_token_file_path', return_value='/test/.litellm/token.json'):
            
            result = get_litellm_gateway_api_key()
            
            assert result is None

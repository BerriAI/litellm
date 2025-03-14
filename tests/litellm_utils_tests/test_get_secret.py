import json
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.proxy._types import KeyManagementSystem
from litellm.secret_managers.main import get_secret


class MockSecretClient:
    def get_secret(self, secret_name):
        return Mock(value="mocked_secret_value")


@pytest.mark.asyncio
async def test_azure_kms():
    """
    Basic asserts that the value from get secret is from Azure Key Vault when Key Management System is Azure Key Vault
    """
    with patch("litellm.secret_manager_client", new=MockSecretClient()):
        litellm._key_management_system = KeyManagementSystem.AZURE_KEY_VAULT
        secret = get_secret(secret_name="ishaan-test-key")
        assert secret == "mocked_secret_value"


def test_file_based_secret_valid():
    """Test reading a secret from a valid file"""
    with tempfile.NamedTemporaryFile(mode="w+") as tmpfile:
        tmpfile.write("test-secret-123")
        tmpfile.flush()
        secret = get_secret(f"file/{tmpfile.name}")
        assert secret == "test-secret-123"


def test_file_based_secret_missing_file():
    """Test error handling when file is missing"""
    with pytest.raises(ValueError) as excinfo:
        get_secret("file/nonexistent/file.txt")
    assert "Secret file not found at path" in str(excinfo.value)

def test_file_based_secret_with_tilde_expansion():
    """Test home directory expansion with ~/ in file paths"""
    from pathlib import Path
    import tempfile

    # Create test file in home directory
    home_dir = Path.home()
    test_content = "tilde-secret-123"
    
    try:
        # Create temp file in home dir
        with tempfile.NamedTemporaryFile(mode="w+", dir=home_dir, delete=False) as tmpfile:
            tmpfile.write(test_content)
            tmpfile.flush()
            file_name = Path(tmpfile.name).name
            
            # Test tilde expansion
            secret = get_secret(f"file/~/{file_name}")
            assert secret == test_content
    finally:
        # Cleanup
        temp_path = Path(tmpfile.name)
        if temp_path.exists():
            temp_path.unlink()

def test_file_based_secret_permission_denied():
    """Test handling of files with insufficient permissions"""
    import stat
    
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpfile:
        tmpfile.write("secret-content")
        tmpfile.flush()
        tmpfile_path = tmpfile.name
    
    try:
        # Remove read permissions
        os.chmod(tmpfile_path, 0o000)  # No permissions
        
        with pytest.raises(PermissionError) as excinfo:
            get_secret(f"file/{tmpfile_path}")
            
        assert "Permission denied accessing file" in str(excinfo.value)
    finally:
        # Restore permissions to allow cleanup
        os.chmod(tmpfile_path, 0o600)  # Owner read/write
        os.unlink(tmpfile_path)

def test_file_based_secret_empty_file():
    """Test handling of empty files"""
    with tempfile.NamedTemporaryFile(mode="w+") as tmpfile:
        # File is empty by default
        secret = get_secret(f"file/{tmpfile.name}")
        assert secret == ""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.auth import (
    clear_token,
    get_stored_api_key,
    get_token_file_path,
    load_token,
    login,
    logout,
    save_token,
    whoami,
)


class TestTokenUtilities:
    """Test token file utility functions"""

    def test_get_token_file_path(self):
        """Test getting token file path"""
        with patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_home.return_value = Path('/home/user')
            
            result = get_token_file_path()
            
            assert result == '/home/user/.litellm/token.json'
            mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_get_token_file_path_creates_directory(self):
        """Test that get_token_file_path creates the config directory"""
        with patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_home.return_value = Path('/home/user')
            
            get_token_file_path()
            
            mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_save_token(self):
        """Test saving token data to file"""
        token_data = {
            'key': 'test-key',
            'user_id': 'test-user',
            'timestamp': 1234567890
        }
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.chmod') as mock_chmod:
            
            mock_path.return_value = '/test/path/token.json'
            
            save_token(token_data)
            
            mock_file.assert_called_once_with('/test/path/token.json', 'w')
            mock_file().write.assert_called()
            mock_chmod.assert_called_once_with('/test/path/token.json', 0o600)
            
            # Verify JSON content was written correctly
            written_content = ''.join(call[0][0] for call in mock_file().write.call_args_list)
            parsed_content = json.loads(written_content)
            assert parsed_content == token_data

    def test_load_token_success(self):
        """Test loading token data from file successfully"""
        token_data = {
            'key': 'test-key',
            'user_id': 'test-user',
            'timestamp': 1234567890
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(token_data))), \
             patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=True):
            
            mock_path.return_value = '/test/path/token.json'
            
            result = load_token()
            
            assert result == token_data

    def test_load_token_file_not_exists(self):
        """Test loading token when file doesn't exist"""
        with patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=False):
            
            mock_path.return_value = '/test/path/token.json'
            
            result = load_token()
            
            assert result is None

    def test_load_token_json_decode_error(self):
        """Test loading token with invalid JSON"""
        with patch('builtins.open', mock_open(read_data='invalid json')), \
             patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=True):
            
            mock_path.return_value = '/test/path/token.json'
            
            result = load_token()
            
            assert result is None

    def test_load_token_io_error(self):
        """Test loading token with IO error"""
        with patch('builtins.open', side_effect=IOError("Permission denied")), \
             patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=True):
            
            mock_path.return_value = '/test/path/token.json'
            
            result = load_token()
            
            assert result is None

    def test_clear_token_file_exists(self):
        """Test clearing token when file exists"""
        with patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            
            mock_path.return_value = '/test/path/token.json'
            
            clear_token()
            
            mock_remove.assert_called_once_with('/test/path/token.json')

    def test_clear_token_file_not_exists(self):
        """Test clearing token when file doesn't exist"""
        with patch('litellm.proxy.client.cli.commands.auth.get_token_file_path') as mock_path, \
             patch('os.path.exists', return_value=False), \
             patch('os.remove') as mock_remove:
            
            mock_path.return_value = '/test/path/token.json'
            
            clear_token()
            
            mock_remove.assert_not_called()

    def test_get_stored_api_key_success(self):
        """Test getting stored API key successfully"""
        token_data = {
            'key': 'test-api-key-123',
            'user_id': 'test-user'
        }
        
        with patch('litellm.litellm_core_utils.cli_token_utils.load_cli_token', return_value=token_data):
            result = get_stored_api_key()
            assert result == 'test-api-key-123'

    def test_get_stored_api_key_no_token(self):
        """Test getting stored API key when no token exists"""
        with patch('litellm.litellm_core_utils.cli_token_utils.load_cli_token', return_value=None):
            result = get_stored_api_key()
            assert result is None

    def test_get_stored_api_key_no_key_field(self):
        """Test getting stored API key when token has no key field"""
        token_data = {
            'user_id': 'test-user'
        }
        
        with patch('litellm.litellm_core_utils.cli_token_utils.load_cli_token', return_value=token_data):
            result = get_stored_api_key()
            assert result is None


class TestLoginCommand:
    """Test login CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_login_success(self):
        """Test successful login flow with single team (JWT generated immediately)"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock the requests for successful authentication with single team
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt",
            "user_id": "test-user-123",
            "team_id": "team-1",
            "teams": ["team-1"]
        }
        
        with patch('webbrowser.open') as mock_browser, \
             patch('requests.get', return_value=mock_response) as mock_get, \
             patch('litellm.proxy.client.cli.commands.auth.save_token') as mock_save, \
             patch('litellm.proxy.client.cli.interface.show_commands') as mock_show_commands, \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "✅ Login successful!" in result.output
            assert "Automatically assigned to team: team-1" in result.output
            
            # Verify browser was opened with correct URL
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args
            assert "sk-test-uuid-123" in call_args
            
            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data['key'] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.jwt"
            assert saved_data['user_id'] == 'test-user-123'
            
            # Verify commands were shown
            mock_show_commands.assert_called_once()

    def test_login_timeout(self):
        """Test login timeout scenario"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock response that never returns ready status
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "pending"}
        
        with patch('webbrowser.open'), \
             patch('requests.get', return_value=mock_response), \
             patch('time.sleep') as mock_sleep, \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            # Mock time.sleep to avoid actual delays in tests
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication timed out" in result.output

    def test_login_http_error(self):
        """Test login with HTTP error"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock response with HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch('webbrowser.open'), \
             patch('requests.get', return_value=mock_response), \
             patch('time.sleep'), \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication timed out" in result.output

    def test_login_request_exception(self):
        """Test login with request exception"""
        import requests
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        with patch('webbrowser.open'), \
             patch('requests.get', side_effect=requests.RequestException("Connection failed")), \
             patch('time.sleep'), \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication timed out" in result.output

    def test_login_keyboard_interrupt(self):
        """Test login cancelled by user"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        with patch('webbrowser.open'), \
             patch('requests.get', side_effect=KeyboardInterrupt), \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication cancelled by user" in result.output

    def test_login_no_api_key_in_response(self):
        """Test login when response doesn't contain API key"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock response without API key
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready"
            # Missing 'key' field
        }
        
        with patch('webbrowser.open'), \
             patch('requests.get', return_value=mock_response), \
             patch('time.sleep'), \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication timed out" in result.output

    def test_login_general_exception(self):
        """Test login with general exception (not requests exception)"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        with patch('webbrowser.open'), \
             patch('requests.get', side_effect=ValueError("Invalid value")), \
             patch('litellm._uuid.uuid.uuid4', return_value='test-uuid-123'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "❌ Authentication failed: Invalid value" in result.output


class TestLogoutCommand:
    """Test logout CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_logout_success(self):
        """Test successful logout"""
        with patch('litellm.proxy.client.cli.commands.auth.clear_token') as mock_clear:
            result = self.runner.invoke(logout)
            
            assert result.exit_code == 0
            assert "✅ Logged out successfully" in result.output
            mock_clear.assert_called_once()


class TestWhoamiCommand:
    """Test whoami CLI command"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_whoami_authenticated(self):
        """Test whoami when user is authenticated"""
        token_data = {
            'user_email': 'test@example.com',
            'user_id': 'test-user-123',
            'user_role': 'admin',
            'timestamp': time.time() - 3600  # 1 hour ago
        }
        
        with patch('litellm.proxy.client.cli.commands.auth.load_token', return_value=token_data):
            result = self.runner.invoke(whoami)
            
            assert result.exit_code == 0
            assert "✅ Authenticated" in result.output
            assert "test@example.com" in result.output
            assert "test-user-123" in result.output
            assert "admin" in result.output
            assert "Token age: 1.0 hours" in result.output

    def test_whoami_not_authenticated(self):
        """Test whoami when user is not authenticated"""
        with patch('litellm.proxy.client.cli.commands.auth.load_token', return_value=None):
            result = self.runner.invoke(whoami)
            
            assert result.exit_code == 0
            assert "❌ Not authenticated" in result.output
            assert "Run 'litellm-proxy login'" in result.output

    def test_whoami_old_token(self):
        """Test whoami with old token showing warning"""
        token_data = {
            'user_email': 'test@example.com',
            'user_id': 'test-user-123',
            'user_role': 'admin',
            'timestamp': time.time() - (25 * 3600)  # 25 hours ago
        }
        
        with patch('litellm.proxy.client.cli.commands.auth.load_token', return_value=token_data):
            result = self.runner.invoke(whoami)
            
            assert result.exit_code == 0
            assert "✅ Authenticated" in result.output
            assert "⚠️ Warning: Token is more than 24 hours old" in result.output

    def test_whoami_missing_fields(self):
        """Test whoami with token missing some fields"""
        token_data = {
            'timestamp': time.time() - 3600
            # Missing user_email, user_id, user_role
        }
        
        with patch('litellm.proxy.client.cli.commands.auth.load_token', return_value=token_data):
            result = self.runner.invoke(whoami)
            
            assert result.exit_code == 0
            assert "✅ Authenticated" in result.output
            assert "Unknown" in result.output  # Should show "Unknown" for missing fields

    def test_whoami_no_timestamp(self):
        """Test whoami with token missing timestamp"""
        token_data = {
            'user_email': 'test@example.com',
            'user_id': 'test-user-123',
            'user_role': 'admin'
            # Missing timestamp
        }
        
        with patch('litellm.proxy.client.cli.commands.auth.load_token', return_value=token_data), \
             patch('time.time', return_value=1000):
            
            result = self.runner.invoke(whoami)
            
            assert result.exit_code == 0
            assert "✅ Authenticated" in result.output
            # Should calculate age based on timestamp=0
            assert "Token age:" in result.output


class TestCLIKeyRegenerationFlow:
    """Test the end-to-end CLI key regeneration flow from CLI perspective"""

    def setup_method(self):
        """Setup for each test"""
        self.runner = CliRunner()

    def test_login_with_team_selection_flow(self):
        """Test complete login flow when user has multiple teams - should prompt for selection"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock first response - requires team selection
        mock_first_response = Mock()
        mock_first_response.status_code = 200
        mock_first_response.json.return_value = {
            "status": "ready",
            "requires_team_selection": True,
            "user_id": "test-user-456",
            "teams": ["team-alpha", "team-beta", "team-gamma"]
        }
        
        # Mock second response after team selection - JWT with selected team
        mock_second_response = Mock()
        mock_second_response.status_code = 200
        mock_second_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.team-beta.jwt",
            "user_id": "test-user-456",
            "team_id": "team-beta",
            "teams": ["team-alpha", "team-beta", "team-gamma"]
        }
        
        # Simulate user selecting team #2 (team-beta)
        with patch('webbrowser.open') as mock_browser, \
             patch('requests.get', side_effect=[mock_first_response, mock_second_response]) as mock_get, \
             patch('litellm.proxy.client.cli.commands.auth.save_token') as mock_save, \
             patch('litellm.proxy.client.cli.interface.show_commands') as mock_show_commands, \
             patch('litellm._uuid.uuid.uuid4', return_value='session-uuid-456'), \
             patch('click.prompt', return_value='2'):  # User selects index 2
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "✅ Login successful!" in result.output
            assert "team-beta" in result.output
            
            # Verify browser was opened
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args
            
            # Verify two polling requests were made
            assert mock_get.call_count == 2
            
            # First poll should be without team_id
            first_poll_url = mock_get.call_args_list[0][0][0]
            assert "sk-session-uuid-456" in first_poll_url
            assert "team_id=" not in first_poll_url
            
            # Second poll should include team_id=team-beta
            second_poll_url = mock_get.call_args_list[1][0][0]
            assert "team_id=team-beta" in second_poll_url
            
            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data['key'] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.team-beta.jwt"
            assert saved_data['user_id'] == 'test-user-456'
            
            mock_show_commands.assert_called_once()

    def test_login_without_teams_flow(self):
        """Test complete login flow when user has no teams - JWT generated without team"""
        mock_context = Mock()
        mock_context.obj = {"base_url": "https://test.example.com"}
        
        # Mock response with no teams
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ready",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.no-team.jwt",
            "user_id": "test-user-solo",
            "team_id": None,
            "teams": []
        }
        
        with patch('webbrowser.open') as mock_browser, \
             patch('requests.get', return_value=mock_response), \
             patch('litellm.proxy.client.cli.commands.auth.save_token') as mock_save, \
             patch('litellm.proxy.client.cli.interface.show_commands'), \
             patch('litellm._uuid.uuid.uuid4', return_value='session-uuid-solo'):
            
            result = self.runner.invoke(login, obj=mock_context.obj)
            
            assert result.exit_code == 0
            assert "✅ Login successful!" in result.output
            
            # Verify browser was opened
            mock_browser.assert_called_once()
            call_args = mock_browser.call_args[0][0]
            assert "https://test.example.com/sso/key/generate" in call_args
            assert "source=litellm-cli" in call_args
            assert "key=sk-session-uuid-solo" in call_args
            
            # Verify JWT was saved
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data['key'] == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.no-team.jwt"
            assert saved_data['user_id'] == 'test-user-solo'

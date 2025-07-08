import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

from enterprise.litellm_enterprise.enterprise_callbacks.callback_controls import (
    EnterpriseCallbackControls,
)
from litellm.constants import X_LITELLM_DISABLE_CALLBACKS
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.s3_v2 import S3Logger


class TestEnterpriseCallbackControls:
    
    @pytest.fixture
    def mock_premium_user(self):
        """Fixture to mock premium user check as True"""
        with patch.object(EnterpriseCallbackControls, '_premium_user_check', return_value=True):
            yield
    
    @pytest.fixture 
    def mock_non_premium_user(self):
        """Fixture to mock premium user check as False"""
        with patch.object(EnterpriseCallbackControls, '_premium_user_check', return_value=False):
            yield

    @pytest.fixture
    def mock_request_headers(self):
        """Fixture to mock get_proxy_server_request_headers"""
        with patch('enterprise.litellm_enterprise.enterprise_callbacks.callback_controls.get_proxy_server_request_headers') as mock_headers:
            yield mock_headers

    def test_callback_disabled_langfuse_string(self, mock_premium_user, mock_request_headers):
        """Test that 'langfuse' string callback is disabled when specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params)
        assert result is True

    def test_callback_disabled_langfuse_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that LangfusePromptManagement CustomLogger instance is disabled when 'langfuse' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        langfuse_logger = LangfusePromptManagement()
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers(langfuse_logger, litellm_params)
        assert result is True

    def test_callback_disabled_s3_v2_string(self, mock_premium_user, mock_request_headers):
        """Test that 's3_v2' string callback is disabled when specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("s3_v2", litellm_params)
        assert result is True

    def test_callback_disabled_s3_v2_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that S3Logger CustomLogger instance is disabled when 's3_v2' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        # Mock S3Logger to avoid async initialization issues
        with patch('litellm.integrations.s3_v2.S3Logger.__init__', return_value=None):
            s3_logger = S3Logger()
            result = EnterpriseCallbackControls.is_callback_disabled_via_headers(s3_logger, litellm_params)
            assert result is True

    def test_callback_disabled_datadog_string(self, mock_premium_user, mock_request_headers):
        """Test that 'datadog' string callback is disabled when specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "datadog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("datadog", litellm_params)
        assert result is True

    def test_callback_disabled_datadog_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that DataDogLogger CustomLogger instance is disabled when 'datadog' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "datadog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        # Mock DataDogLogger to avoid async initialization issues
        with patch('litellm.integrations.datadog.datadog.DataDogLogger.__init__', return_value=None):
            datadog_logger = DataDogLogger()
            result = EnterpriseCallbackControls.is_callback_disabled_via_headers(datadog_logger, litellm_params)
            assert result is True

    def test_multiple_callbacks_disabled(self, mock_premium_user, mock_request_headers):
        """Test that multiple callbacks can be disabled with comma-separated list"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse,datadog,s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        # Test each callback is disabled
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("datadog", litellm_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("s3_v2", litellm_params) is True
        
        # Test non-disabled callback is not disabled
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("prometheus", litellm_params) is False

    def test_callback_not_disabled_when_not_in_list(self, mock_premium_user, mock_request_headers):
        """Test that callbacks not in the disabled list are not disabled"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("datadog", litellm_params)
        assert result is False

    def test_callback_not_disabled_when_no_header(self, mock_premium_user, mock_request_headers):
        """Test that callbacks are not disabled when the header is not present"""
        mock_request_headers.return_value = {}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params)
        assert result is False

    def test_callback_not_disabled_when_header_none(self, mock_premium_user, mock_request_headers):
        """Test that callbacks are not disabled when the header value is None"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: None}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params)
        assert result is False

    def test_non_premium_user_cannot_disable_callbacks(self, mock_non_premium_user, mock_request_headers):
        """Test that non-premium users cannot disable callbacks even with the header"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params)
        assert result is False

    def test_case_insensitive_callback_matching(self, mock_premium_user, mock_request_headers):
        """Test that callback matching is case insensitive"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "LANGFUSE,DataDog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        # Test lowercase callbacks are disabled
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("datadog", litellm_params) is True

    def test_whitespace_handling_in_disabled_callbacks(self, mock_premium_user, mock_request_headers):
        """Test that whitespace around callback names is handled correctly"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: " langfuse , datadog , s3_v2 "}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("datadog", litellm_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_via_headers("s3_v2", litellm_params) is True

    def test_custom_logger_not_in_registry(self, mock_premium_user, mock_request_headers):
        """Test that CustomLogger not in registry is not disabled"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "unknown_logger"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        # Create a mock CustomLogger that's not in the registry
        class UnknownLogger(CustomLogger):
            pass
        
        unknown_logger = UnknownLogger()
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers(unknown_logger, litellm_params)
        assert result is False

    def test_exception_handling(self, mock_premium_user, mock_request_headers):
        """Test that exceptions are handled gracefully and return False"""
        # Make get_proxy_server_request_headers raise an exception
        mock_request_headers.side_effect = Exception("Test exception")
        litellm_params = {"proxy_server_request": {"url": "test"}}
        
        result = EnterpriseCallbackControls.is_callback_disabled_via_headers("langfuse", litellm_params)
        assert result is False

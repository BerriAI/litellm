import unittest.mock as mock
from typing import cast
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
from litellm.types.utils import StandardCallbackDynamicParams


class TestEnterpriseCallbackControls:
    
    @pytest.fixture
    def mock_premium_user(self):
        """Fixture to mock premium user check as True"""
        with patch.object(EnterpriseCallbackControls, '_should_allow_dynamic_callback_disabling', return_value=True):
            yield
    
    @pytest.fixture 
    def mock_non_premium_user(self):
        """Fixture to mock premium user check as False"""
        with patch.object(EnterpriseCallbackControls, '_should_allow_dynamic_callback_disabling', return_value=False):
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
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is True

    def test_callback_disabled_langfuse_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that LangfusePromptManagement CustomLogger instance is disabled when 'langfuse' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        langfuse_logger = LangfusePromptManagement()
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically(langfuse_logger, litellm_params, standard_callback_dynamic_params)
        assert result is True

    def test_callback_disabled_s3_v2_string(self, mock_premium_user, mock_request_headers):
        """Test that 's3_v2' string callback is disabled when specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("s3_v2", litellm_params, standard_callback_dynamic_params)
        assert result is True

    def test_callback_disabled_s3_v2_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that S3Logger CustomLogger instance is disabled when 's3_v2' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Mock S3Logger to avoid async initialization issues
        with patch('litellm.integrations.s3_v2.S3Logger.__init__', return_value=None):
            s3_logger = S3Logger()
            result = EnterpriseCallbackControls.is_callback_disabled_dynamically(s3_logger, litellm_params, standard_callback_dynamic_params)
            assert result is True

    def test_callback_disabled_datadog_string(self, mock_premium_user, mock_request_headers):
        """Test that 'datadog' string callback is disabled when specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "datadog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params)
        assert result is True

    def test_callback_disabled_datadog_customlogger(self, mock_premium_user, mock_request_headers):
        """Test that DataDogLogger CustomLogger instance is disabled when 'datadog' specified in headers"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "datadog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Mock DataDogLogger to avoid async initialization issues
        with patch('litellm.integrations.datadog.datadog.DataDogLogger.__init__', return_value=None):
            datadog_logger = DataDogLogger()
            result = EnterpriseCallbackControls.is_callback_disabled_dynamically(datadog_logger, litellm_params, standard_callback_dynamic_params)
            assert result is True

    def test_multiple_callbacks_disabled(self, mock_premium_user, mock_request_headers):
        """Test that multiple callbacks can be disabled with comma-separated list"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse,datadog,s3_v2"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Test each callback is disabled
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("s3_v2", litellm_params, standard_callback_dynamic_params) is True
        
        # Test non-disabled callback is not disabled
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("prometheus", litellm_params, standard_callback_dynamic_params) is False

    def test_callback_not_disabled_when_not_in_list(self, mock_premium_user, mock_request_headers):
        """Test that callbacks not in the disabled list are not disabled"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_callback_not_disabled_when_no_header(self, mock_premium_user, mock_request_headers):
        """Test that callbacks are not disabled when the header is not present"""
        mock_request_headers.return_value = {}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_callback_not_disabled_when_header_none(self, mock_premium_user, mock_request_headers):
        """Test that callbacks are not disabled when the header value is None"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: None}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_non_premium_user_cannot_disable_callbacks(self, mock_non_premium_user, mock_request_headers):
        """Test that non-premium users cannot disable callbacks even with the header"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_case_insensitive_callback_matching(self, mock_premium_user, mock_request_headers):
        """Test that callback matching is case insensitive"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "LANGFUSE,DataDog"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Test lowercase callbacks are disabled
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params) is True

    def test_whitespace_handling_in_disabled_callbacks(self, mock_premium_user, mock_request_headers):
        """Test that whitespace around callback names is handled correctly"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: " langfuse , datadog , s3_v2 "}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("s3_v2", litellm_params, standard_callback_dynamic_params) is True

    def test_custom_logger_not_in_registry(self, mock_premium_user, mock_request_headers):
        """Test that CustomLogger not in registry is not disabled"""
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "unknown_logger"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Create a mock CustomLogger that's not in the registry
        class UnknownLogger(CustomLogger):
            pass
        
        unknown_logger = UnknownLogger()
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically(unknown_logger, litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_exception_handling(self, mock_premium_user, mock_request_headers):
        """Test that exceptions are handled gracefully and return False"""
        # Make get_proxy_server_request_headers raise an exception
        mock_request_headers.side_effect = Exception("Test exception")
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is False

    def test_callback_disabled_via_request_body_langfuse(self, mock_premium_user, mock_request_headers):
        """Test that callbacks can be disabled via request body litellm_disabled_callbacks"""
        mock_request_headers.return_value = {}  # No headers
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams(litellm_disabled_callbacks=["langfuse"])
        
        result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
        assert result is True

    def test_callback_disabled_via_request_body_multiple(self, mock_premium_user, mock_request_headers):
        """Test that multiple callbacks can be disabled via request body"""
        mock_request_headers.return_value = {}  # No headers
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams(litellm_disabled_callbacks=["langfuse", "datadog", "s3_v2"])
        
        # Test each callback is disabled
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("datadog", litellm_params, standard_callback_dynamic_params) is True
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("s3_v2", litellm_params, standard_callback_dynamic_params) is True
        
        # Test non-disabled callback is not disabled
        assert EnterpriseCallbackControls.is_callback_disabled_dynamically("prometheus", litellm_params, standard_callback_dynamic_params) is False

    def test_admin_can_disable_dynamic_callback_disabling(self, mock_request_headers):
        """
        Test that when admin sets allow_dynamic_callback_disabling to False,
        callbacks cannot be disabled dynamically even for premium users
        """
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Mock litellm.allow_dynamic_callback_disabling set to False
        with patch('litellm.allow_dynamic_callback_disabling', False):
            with patch('litellm.proxy.proxy_server.premium_user', True):
                result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
                assert result is False

    def test_admin_can_enable_dynamic_callback_disabling(self, mock_request_headers):
        """
        Test that when admin sets allow_dynamic_callback_disabling to True,
        callbacks can be disabled dynamically for premium users
        """
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # Mock litellm.allow_dynamic_callback_disabling set to True
        with patch('litellm.allow_dynamic_callback_disabling', True):
            with patch('litellm.proxy.proxy_server.premium_user', True):
                result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
                assert result is True

    def test_default_admin_setting_allows_dynamic_callback_disabling(self, mock_request_headers):
        """
        Test that when allow_dynamic_callback_disabling is not set,
        it defaults to True and allows dynamic callback disabling for premium users
        """
        mock_request_headers.return_value = {X_LITELLM_DISABLE_CALLBACKS: "langfuse"}
        litellm_params = {"proxy_server_request": {"url": "test"}}
        standard_callback_dynamic_params = StandardCallbackDynamicParams()
        
        # litellm.allow_dynamic_callback_disabling should default to True
        with patch('litellm.proxy.proxy_server.premium_user', True):
            result = EnterpriseCallbackControls.is_callback_disabled_dynamically("langfuse", litellm_params, standard_callback_dynamic_params)
            assert result is True

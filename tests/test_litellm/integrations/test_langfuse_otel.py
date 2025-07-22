import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litellm.types.integrations.langfuse_otel import LangfuseOtelConfig


class TestLangfuseOtelIntegration:
    
    def test_get_langfuse_otel_config_with_required_env_vars(self):
        """Test that config is created correctly with required environment variables."""
        # Clean environment of any Langfuse-related variables
        env_vars_to_clean = ['LANGFUSE_HOST', 'OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_HEADERS']
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key'
        }, clear=False):
            # Remove any existing Langfuse variables
            for var in env_vars_to_clean:
                if var in os.environ:
                    del os.environ[var]
                    
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert isinstance(config, LangfuseOtelConfig)
            assert config.protocol == "otlp_http"
            assert "Authorization=Basic" in config.otlp_auth_headers
            # Check that environment variables are set correctly (US default)
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://us.cloud.langfuse.com/api/public/otel"
            assert "Authorization=Basic" in os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    
    def test_get_langfuse_otel_config_missing_keys(self):
        """Test that ValueError is raised when required keys are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set"):
                LangfuseOtelLogger.get_langfuse_otel_config()
    
    def test_get_langfuse_otel_config_with_eu_host(self):
        """Test config with EU host."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://cloud.langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://cloud.langfuse.com/api/public/otel"
    
    def test_get_langfuse_otel_config_with_custom_host(self):
        """Test config with custom host."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://my-langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://my-langfuse.com/api/public/otel"
    
    def test_get_langfuse_otel_config_with_host_no_protocol(self):
        """Test config with custom host without protocol."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'my-langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://my-langfuse.com/api/public/otel"
    
    def test_set_langfuse_otel_attributes(self):
        """Test that set_langfuse_otel_attributes calls the Arize utils function."""
        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        mock_response = {"test": "response"}
        
        with patch('litellm.integrations.arize._utils.set_attributes') as mock_set_attributes:
            LangfuseOtelLogger.set_langfuse_otel_attributes(mock_span, mock_kwargs, mock_response)
            
            mock_set_attributes.assert_called_once_with(mock_span, mock_kwargs, mock_response)

    def test_set_langfuse_environment_attribute(self):
        """Test that Langfuse environment is set correctly when environment variable is present."""
        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        test_env = "staging"

        with patch.dict(os.environ, {'LANGFUSE_TRACING_ENVIRONMENT': test_env}):
            with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
                LangfuseOtelLogger._set_langfuse_specific_attributes(mock_span, mock_kwargs)
                
                mock_safe_set_attribute.assert_called_once_with(
                    span=mock_span,
                    key="langfuse.environment",
                    value=test_env
                )


if __name__ == "__main__":
    pytest.main([__file__]) 
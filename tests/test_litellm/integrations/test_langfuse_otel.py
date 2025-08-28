import json
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
                
                # safe_set_attribute(span, key, value) → positional args
                mock_safe_set_attribute.assert_called_once_with(
                    mock_span,
                    "langfuse.environment",
                    test_env
                )

    def test_extract_langfuse_metadata_basic(self):
        """Ensure metadata is correctly pulled from litellm_params."""
        metadata_in = {"generation_name": "my-gen", "custom": "data"}
        kwargs = {"litellm_params": {"metadata": metadata_in}}
        extracted = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        assert extracted == metadata_in

    def test_extract_langfuse_metadata_with_header_enrichment(self, monkeypatch):
        """_extract_langfuse_metadata should call LangFuseLogger.add_metadata_from_header when available."""
        import sys
        import types

        # Build a stub module + class on-the-fly
        stub_module = types.ModuleType("litellm.integrations.langfuse.langfuse")
        class StubLFLogger:
            @staticmethod
            def add_metadata_from_header(litellm_params, metadata):
                # Echo back existing metadata plus a marker
                return {**metadata, "enriched": True}
        stub_module.LangFuseLogger = StubLFLogger  # type: ignore

        # Register stub in sys.modules so import inside method succeeds
        sys.modules["litellm.integrations.langfuse.langfuse"] = stub_module  # type: ignore

        kwargs = {"litellm_params": {"metadata": {"foo": "bar"}}}
        extracted = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        assert extracted.get("foo") == "bar"
        assert extracted.get("enriched") is True

    def test_set_langfuse_specific_attributes_full_mapping(self):
        """Verify every supported metadata key maps to the correct OTEL attribute and complex types are JSON-serialised."""
        # Build a sample metadata payload covering all mappings
        metadata = {
            "generation_name": "gen-name",
            "generation_id": "gen-id",
            "parent_observation_id": "parent-id",
            "version": "v1",
            "mask_input": True,
            "mask_output": False,
            "trace_user_id": "user-123",
            "session_id": "sess-456",
            "tags": ["tagA", "tagB"],
            "trace_name": "trace-name",
            "trace_id": "trace-id",
            "trace_metadata": {"k": "v"},
            "trace_version": "t-ver",
            "trace_release": "rel-1",
            "existing_trace_id": "existing-id",
            "update_trace_keys": ["key1", "key2"],
            "debug_langfuse": True,
        }
        kwargs = {"litellm_params": {"metadata": metadata}}

        # Capture calls to safe_set_attribute
        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(MagicMock(), kwargs)

            # Build expected calls manually for clarity
            from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
            expected = {
                LangfuseSpanAttributes.GENERATION_NAME.value: "gen-name",
                LangfuseSpanAttributes.GENERATION_ID.value: "gen-id",
                LangfuseSpanAttributes.PARENT_OBSERVATION_ID.value: "parent-id",
                LangfuseSpanAttributes.GENERATION_VERSION.value: "v1",
                LangfuseSpanAttributes.MASK_INPUT.value: True,
                LangfuseSpanAttributes.MASK_OUTPUT.value: False,
                LangfuseSpanAttributes.TRACE_USER_ID.value: "user-123",
                LangfuseSpanAttributes.SESSION_ID.value: "sess-456",
                # Lists / dicts should be JSON strings
                LangfuseSpanAttributes.TAGS.value: json.dumps(["tagA", "tagB"]),
                LangfuseSpanAttributes.TRACE_NAME.value: "trace-name",
                LangfuseSpanAttributes.TRACE_ID.value: "trace-id",
                LangfuseSpanAttributes.TRACE_METADATA.value: json.dumps({"k": "v"}),
                LangfuseSpanAttributes.TRACE_VERSION.value: "t-ver",
                LangfuseSpanAttributes.TRACE_RELEASE.value: "rel-1",
                LangfuseSpanAttributes.EXISTING_TRACE_ID.value: "existing-id",
                LangfuseSpanAttributes.UPDATE_TRACE_KEYS.value: json.dumps(["key1", "key2"]),
                LangfuseSpanAttributes.DEBUG_LANGFUSE.value: True,
            }

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: call.args[2]  # (span, key, value)
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual == expected, "Mismatch between expected and actual OTEL attribute mapping."

    def test_construct_dynamic_otel_headers_with_langfuse_keys(self):
        """Test that construct_dynamic_otel_headers creates proper auth headers when langfuse keys are provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        # Create dynamic params with langfuse keys
        dynamic_params = StandardCallbackDynamicParams(
            langfuse_public_key="test_public_key",
            langfuse_secret_key="test_secret_key"
        )
        
        logger = LangfuseOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)
        
        # Should return a dict with otlp_auth_headers
        assert result is not None
        assert "Authorization" in result
        
        # The auth header should contain the basic auth format
        auth_header = result["Authorization"]
        assert auth_header.startswith("Basic ")
        
        # Verify the header format by decoding
        import base64

        # Extract the base64 part from "Authorization=Basic <base64>"
        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()
        
        assert decoded == "test_public_key:test_secret_key"

    def test_construct_dynamic_otel_headers_empty_params(self):
        """Test that construct_dynamic_otel_headers returns empty dict when no langfuse keys are provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        # Create dynamic params without langfuse keys
        dynamic_params = StandardCallbackDynamicParams()
        
        logger = LangfuseOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)
        
        # Should return an empty dict
        assert result == {}
    
    def test_get_langfuse_otel_config_with_otel_host_priority(self):
        """LANGFUSE_OTEL_HOST should take priority over LANGFUSE_HOST."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://should-not-be-used.com',
            'LANGFUSE_OTEL_HOST': 'https://otel-host.com'
        }, clear=False):
            _ = LangfuseOtelLogger.get_langfuse_otel_config()

            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://otel-host.com/api/public/otel"
    


if __name__ == "__main__":
    pytest.main([__file__]) 
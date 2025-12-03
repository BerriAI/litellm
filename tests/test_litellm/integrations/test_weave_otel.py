import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.weave.weave_otel import (
    _set_weave_specific_attributes,
    get_weave_otel_config,
)
from litellm.types.integrations.weave_otel import WeaveOtelConfig, WeaveSpanAttributes


def test_get_weave_otel_config():
    """Test config creation with required env vars and error cases for missing vars."""
    # Test successful config creation with required environment variables
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
        },
        clear=True,
    ):
        config = get_weave_otel_config()

        assert isinstance(config, WeaveOtelConfig)
        assert config.protocol == "otlp_http"
        assert config.project_id == "test-entity/test-project"
        assert config.otlp_auth_headers is not None
        assert "Authorization=" in config.otlp_auth_headers
        assert "project_id=test-entity/test-project" in config.otlp_auth_headers
        assert config.endpoint == "https://trace.wandb.ai/otel/v1/traces"
        
        # Verify environment variables were set
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://trace.wandb.ai/otel/v1/traces"
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == config.otlp_auth_headers

    # Test ValueError when WANDB_API_KEY is missing
    with patch.dict(os.environ, {"WANDB_PROJECT_ID": "test-entity/test-project"}, clear=True):
        with pytest.raises(ValueError, match="WANDB_API_KEY must be set"):
            get_weave_otel_config()

    # Test ValueError when WANDB_PROJECT_ID is missing
    with patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"}, clear=True):
        with pytest.raises(ValueError, match="WANDB_PROJECT_ID must be set"):
            get_weave_otel_config()


def test_get_weave_otel_config_with_custom_host():
    """Test config creation with custom WANDB_HOST."""
    # Test with host that already has https://
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "https://custom.wandb.io",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"
        assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://custom.wandb.io/otel/v1/traces"

    # Test with host without http:// or https://
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "custom.wandb.io",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"

    # Test with host with trailing slash
    with patch.dict(
        os.environ,
        {
            "WANDB_API_KEY": "test_api_key",
            "WANDB_PROJECT_ID": "test-entity/test-project",
            "WANDB_HOST": "https://custom.wandb.io/",
        },
        clear=True,
    ):
        config = get_weave_otel_config()
        assert config.endpoint == "https://custom.wandb.io/otel/v1/traces"





def test_set_weave_specific_attributes_display_name_from_metadata():
    """Test _set_weave_specific_attributes sets display_name from metadata."""
    mock_span = MagicMock()
    kwargs = {
        "metadata": {"display_name": "custom-display-name"},
        "model": "gpt-4",
    }
    
    with patch("litellm.integrations.weave.weave_otel.safe_set_attribute") as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)
        
        # Should set display_name from metadata
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.DISPLAY_NAME.value, "custom-display-name"
        )


def test_set_weave_specific_attributes_display_name_from_model():
    """Test _set_weave_specific_attributes sets display_name from model when not in metadata."""
    mock_span = MagicMock()
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "metadata": {},
    }
    
    with patch("litellm.integrations.weave.weave_otel.safe_set_attribute") as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)
        
        # Should set display_name from model
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.DISPLAY_NAME.value, "openai__gpt-4o-mini"
        )



def test_set_weave_specific_attributes_thread_id_and_is_turn():
    """Test _set_weave_specific_attributes sets thread_id and is_turn from session_id."""
    mock_span = MagicMock()
    kwargs = {
        "metadata": {"session_id": "session-123"},
    }
    
    with patch("litellm.integrations.weave.weave_otel.safe_set_attribute") as mock_safe_set:
        _set_weave_specific_attributes(mock_span, kwargs, None)
        
        # Should set thread_id and is_turn
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.THREAD_ID.value, "session-123"
        )
        mock_safe_set.assert_any_call(
            mock_span, WeaveSpanAttributes.IS_TURN.value, True
        )


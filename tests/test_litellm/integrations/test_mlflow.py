import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm


@pytest.mark.asyncio
async def test_mlflow_request_tags_functionality():
    """Test that request_tags are properly extracted and transformed into tags for MLflow traces."""
    
    # Mock MLflow client and dependencies
    mock_client = MagicMock()
    mock_span = MagicMock()
    mock_span.parent_id = None  # Simulate root trace
    mock_span.request_id = "test_trace_id"
    mock_client.start_trace.return_value = mock_span
    
    # Mock all MLflow-related imports to avoid requiring MLflow as a dependency
    mock_mlflow_tracking = MagicMock()
    mock_mlflow_tracking.MlflowClient = MagicMock(return_value=mock_client)
    
    mock_mlflow_entities = MagicMock()
    mock_mlflow_entities.SpanStatusCode.OK = "OK"
    mock_mlflow_entities.SpanStatusCode.ERROR = "ERROR"
    mock_mlflow_entities.SpanType.LLM = "LLM"
    
    mock_mlflow = MagicMock()
    mock_mlflow.get_current_active_span.return_value = None
    
    with patch.dict('sys.modules', {
        'mlflow': mock_mlflow,
        'mlflow.tracking': mock_mlflow_tracking,
        'mlflow.entities': mock_mlflow_entities,
        'mlflow.tracing.utils': MagicMock(),
    }):
        # Now we can safely import MlflowLogger
        from litellm.integrations.mlflow import MlflowLogger

        # Create MlflowLogger instance
        mlflow_logger = MlflowLogger()
        litellm.callbacks = [mlflow_logger]
        
        # Test completion with request_tags
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test message"}],
            mock_response="test response",
            metadata={
                "tags": ["tag1", "tag2", "production"]
            }
        )
        
        # Allow time for async processing
        await asyncio.sleep(1)
        
        # Verify start_trace was called with tags parameter
        assert mock_client.start_trace.called, "start_trace should have been called"
        
        # Get the call arguments
        call_args = mock_client.start_trace.call_args
        assert call_args is not None, "start_trace call args should not be None"
        
        # Check that tags parameter was included and properly transformed
        tags_param = call_args.kwargs.get('tags', {})
        expected_tags = {"tag1": "", "tag2": "", "production": ""}
        assert tags_param == expected_tags, f"Expected tags {expected_tags}, got {tags_param}"
        
        print("✅ Request tags properly transformed and passed to MLflow trace")

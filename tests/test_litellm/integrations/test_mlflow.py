import importlib
import os
import sys
from unittest.mock import MagicMock, patch
import time

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import litellm


def test_mlflow_request_tags_functionality():
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
    
    with patch.dict(
        'sys.modules',
        {
            'mlflow': mock_mlflow,
            'mlflow.tracking': mock_mlflow_tracking,
            'mlflow.entities': mock_mlflow_entities,
            'mlflow.tracing.utils': MagicMock(),
        },
    ):
        from litellm.integrations.mlflow import MlflowLogger

        mlflow_logger = MlflowLogger()
        litellm.callbacks = [mlflow_logger]

        litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test message"}],
            mock_response="test response",
            metadata={"tags": ["tag1", "tag2", "production"]},
        )

        time.sleep(2)

        assert mock_client.start_trace.called, "start_trace should have been called"

        call_args = mock_client.start_trace.call_args
        assert call_args is not None, "start_trace call args should not be None"

        tags_param = call_args.kwargs.get('tags', {})
        expected_tags = {"tag1": "", "tag2": "", "production": ""}
        assert tags_param == expected_tags, f"Expected tags {expected_tags}, got {tags_param}"



def test_mlflow_token_usage_attribute_structure():
    """Ensure token usage attributes are formatted with mlflow.chat.tokenUsage."""

    mock_mlflow_tracking = MagicMock()
    mock_mlflow_tracking.MlflowClient = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "mlflow": MagicMock(),
            "mlflow.tracking": mock_mlflow_tracking,
            "mlflow.tracing.utils": MagicMock(),
        },
    ):
        mlflow_module = importlib.import_module("litellm.integrations.mlflow")
        mlflow_logger = mlflow_module.MlflowLogger()

        attrs = mlflow_logger._extract_attributes(  # type: ignore
            {
                "litellm_call_id": "123",
                "call_type": "completion",
                "model": "gpt-3.5-turbo",
                "standard_logging_object": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            }
        )

        assert attrs["mlflow.chat.tokenUsage"] == {
            "input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 12,
        }

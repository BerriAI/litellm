import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm


@pytest.mark.asyncio
async def test_mlflow_logging_functionality():
    """Test that inputs, outputs and tags are properly logged in MLflow traces."""

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
        "sys.modules",
        {
            "mlflow": mock_mlflow,
            "mlflow.tracking": mock_mlflow_tracking,
            "mlflow.entities": mock_mlflow_entities,
            "mlflow.tracing.utils": MagicMock(),
        },
    ):
        # Now we can safely import MlflowLogger
        from litellm.integrations.mlflow import MlflowLogger

        # Create MlflowLogger instance
        mlflow_logger = MlflowLogger()
        litellm.callbacks = [mlflow_logger]

        # Test completion with request_tags and prediction parameter
        test_prediction = {"type": "content", "content": "This is a predicted output"}
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test message"}],
            prediction=test_prediction,
            mock_response="test response",
            metadata={"tags": ["tag1", "tag2", "production", "jobID:214590dsff09fds", "taskName:run_page_classification"]},
        )

        # Allow time for async processing
        await asyncio.sleep(1)

        # Verify start_trace was called with tags parameter
        assert mock_client.start_trace.called, "start_trace should have been called"

        # Get the call arguments
        call_args = mock_client.start_trace.call_args
        assert call_args is not None, "start_trace call args should not be None"

        # Check that tags parameter was included and properly transformed
        tags_param = call_args.kwargs.get("tags", {})
        expected_tags = {
            "tag1": "",
            "tag2": "",
            "production": "",
            "jobID": "214590dsff09fds",
            "taskName": "run_page_classification",
        }
        assert tags_param == expected_tags, f"Expected tags {expected_tags}, got {tags_param}"

        # Check that prediction parameter was included in inputs
        inputs_param = call_args.kwargs.get("inputs", {})
        assert "prediction" in inputs_param, "Prediction should be included in span inputs"
        assert inputs_param["prediction"] == test_prediction, (
            f"Expected prediction {test_prediction}, got {inputs_param['prediction']}"
        )


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
        from litellm.integrations.mlflow import MlflowLogger

        mlflow_logger = MlflowLogger()

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

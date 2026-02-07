
import pytest
from unittest.mock import MagicMock, patch
import os
from litellm.integrations.langfuse.langfuse import (
    log_provider_specific_information_as_span,
    LangFuseLogger
)

class TestLangfuseEnvironment:
    
    @patch.dict(os.environ, {"LANGFUSE_TRACING_ENVIRONMENT": "production"}, clear=True)
    def test_log_provider_specific_information_as_span_includes_environment(self):
        """
        Verify that log_provider_specific_information_as_span includes the 
        LANGFUSE_TRACING_ENVIRONMENT in the span tags.
        """
        trace = MagicMock()
        clean_metadata = {
            "hidden_params": {
                "vertex_ai_grounding_metadata": {"foo": "bar"}
            }
        }
        
        # Call the function
        log_provider_specific_information_as_span(trace, clean_metadata)
        
        # Check if trace.span was called
        assert trace.span.called
        
        # Verify call arguments
        # We expect tags=["environment:production"] to be passed
        # Currently, this test should FAIL because tags are not passed.
        
        _, kwargs = trace.span.call_args
        tags = kwargs.get("tags", [])
        
        # This assertions mirrors what we WANT to see
        # If the bug exists, tags will be empty or None, or missing this specific tag.
        assert "environment:production" in tags, f"Expected 'environment:production' in tags, got {tags}"

    @patch.dict(os.environ, {"LANGFUSE_TRACING_ENVIRONMENT": "staging"}, clear=True)
    def test_log_guardrail_information_as_span_includes_environment(self):
        """
        Verify that _log_guardrail_information_as_span includes the
        LANGFUSE_TRACING_ENVIRONMENT in the span tags.
        """
        trace = MagicMock()
        
        # Mock sys.modules to simulate langfuse installation
        langfuse_mock = MagicMock()
        langfuse_mock.version.__version__ = "2.0.0"
        
        with patch.dict("sys.modules", {
            "langfuse": langfuse_mock, 
            "langfuse.Langfuse": MagicMock(),
            "langfuse.version": langfuse_mock.version
        }):
            # Setup mock logger wrapper to access the method
            # We need to bypass __init__ logic that requires valid keys/host if we don't supply them
            # or just supply dummy values.
            logger = LangFuseLogger(langfuse_public_key="pk", langfuse_secret="sk")
            logger._is_langfuse_v2 = MagicMock(return_value=True)
            
            standard_logging_object = {
                "guardrail_information": [
                    {
                        "guardrail_name": "test_guard",
                        "guardrail_request": "input",
                        "guardrail_response": "output"
                    }
                ]
            }
            
            # Call the private method
            logger._log_guardrail_information_as_span(trace, standard_logging_object)
            
            # Check if trace.span was called
            assert trace.span.called
            
            # Verify call arguments
            _, kwargs = trace.span.call_args
            tags = kwargs.get("tags", [])
            
            assert "environment:staging" in tags, f"Expected 'environment:staging' in tags, got {tags}"

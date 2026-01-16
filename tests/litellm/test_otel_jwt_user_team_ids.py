"""
Tests for OTEL logging of JWT user_id and team_id

This test verifies that when using JWT authentication, the user_id and team_id
extracted from the JWT token are properly logged to OTEL spans as metadata attributes.

Related issue: https://github.com/BerriAI/litellm/issues/5484
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.types.utils import StandardLoggingPayload, StandardLoggingMetadata


class TestOtelJWTUserTeamIds:
    """Test class for OTEL JWT user_id and team_id logging."""

    @pytest.fixture
    def in_memory_exporter(self):
        """Create an in-memory span exporter for testing."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        yield exporter
        exporter.clear()

    @pytest.fixture
    def otel_logger(self, in_memory_exporter):
        """Create an OpenTelemetry logger with in-memory exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        # Create a new TracerProvider with the in-memory exporter
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

        config = OpenTelemetryConfig(exporter=in_memory_exporter)
        return OpenTelemetry(config=config, tracer_provider=provider)

    def test_metadata_contains_user_api_key_fields(self, otel_logger, in_memory_exporter):
        """
        Test that user_api_key_team_id and user_api_key_user_id are set on OTEL spans.

        This simulates the scenario where JWT auth extracts user_id and team_id
        and they are passed through the metadata to OTEL logging.
        """
        # Create a mock standard logging payload with JWT-derived user/team IDs
        jwt_user_id = "jwt-user-123"
        jwt_team_id = "jwt-team-456"

        metadata: StandardLoggingMetadata = {
            "user_api_key_hash": "hashed-jwt-abc123",
            "user_api_key_alias": None,
            "user_api_key_spend": 0.0,
            "user_api_key_max_budget": None,
            "user_api_key_budget_reset_at": None,
            "user_api_key_team_id": jwt_team_id,
            "user_api_key_user_id": jwt_user_id,
            "user_api_key_org_id": None,
            "user_api_key_team_alias": "test-team",
            "user_api_key_end_user_id": None,
            "user_api_key_request_route": "/v1/chat/completions",
            "user_api_key_user_email": "test@example.com",
            "user_api_key_auth_metadata": {},
            "spend_logs_metadata": None,
            "requester_ip_address": "127.0.0.1",
            "requester_metadata": None,
            "requester_custom_headers": {},
            "prompt_management_metadata": None,
            "mcp_tool_call_metadata": None,
            "vector_store_request_metadata": None,
            "applied_guardrails": None,
            "usage_object": None,
            "cold_storage_object_key": None,
        }

        standard_logging_payload: StandardLoggingPayload = {
            "id": "test-id-123",
            "trace_id": "trace-123",
            "call_type": "completion",
            "cache_hit": None,
            "stream": False,
            "status": "success",
            "status_fields": {"llm_api_status": "success", "guardrail_status": "not_run"},
            "custom_llm_provider": "openai",
            "saved_cache_cost": 0.0,
            "startTime": datetime.now().timestamp(),
            "endTime": datetime.now().timestamp(),
            "completionStartTime": datetime.now().timestamp(),
            "response_time": 0.5,
            "model": "gpt-3.5-turbo",
            "metadata": metadata,
            "cache_key": None,
            "response_cost": 0.001,
            "cost_breakdown": None,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "request_tags": [],
            "end_user": "",
            "api_base": "https://api.openai.com",
            "model_group": "gpt-3.5-turbo",
            "model_id": "model-123",
            "requester_ip_address": "127.0.0.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "response": {
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi there!"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-3.5-turbo",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 50,
                    "total_tokens": 100,
                },
            },
            "model_parameters": {},
            "hidden_params": {
                "model_id": "model-123",
                "cache_key": None,
                "api_base": None,
                "response_cost": None,
                "litellm_overhead_time_ms": None,
                "additional_headers": None,
                "batch_models": None,
                "litellm_model_name": None,
                "usage_object": None,
            },
            "model_map_information": {"model_map_key": "gpt-3.5-turbo", "model_map_value": None},
            "error_str": None,
            "error_information": {
                "error_code": "",
                "error_class": "",
                "llm_provider": "",
                "traceback": "",
                "error_message": "",
            },
            "response_cost_failure_debug_info": None,
            "guardrail_information": None,
            "standard_built_in_tools_params": {"web_search_options": None, "file_search": None},
        }

        # Create kwargs that would be passed to the OTEL logger
        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {
                "custom_llm_provider": "openai",
                "metadata": {},
            },
            "standard_logging_object": standard_logging_payload,
        }

        response_obj = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "total_tokens": 100,
            },
        }

        start_time = datetime.now()
        end_time = datetime.now()

        # Call the success handler
        otel_logger.log_success_event(kwargs, response_obj, start_time, end_time)

        # Get the finished spans
        spans = in_memory_exporter.get_finished_spans()

        # Should have at least one span
        assert len(spans) >= 1, f"Expected at least 1 span, got {len(spans)}"

        # Find the litellm_request span
        litellm_request_span = None
        for span in spans:
            if span.name == "litellm_request":
                litellm_request_span = span
                break

        assert litellm_request_span is not None, "litellm_request span not found"

        # Check that the JWT user_id and team_id are in the span attributes
        span_attributes = dict(litellm_request_span.attributes)
        print("Span attributes:", span_attributes)

        # Verify user_api_key_user_id is set
        assert (
            "metadata.user_api_key_user_id" in span_attributes
        ), f"metadata.user_api_key_user_id not found in span attributes. Available: {list(span_attributes.keys())}"
        assert (
            span_attributes["metadata.user_api_key_user_id"] == jwt_user_id
        ), f"Expected user_id '{jwt_user_id}', got '{span_attributes.get('metadata.user_api_key_user_id')}'"

        # Verify user_api_key_team_id is set
        assert (
            "metadata.user_api_key_team_id" in span_attributes
        ), f"metadata.user_api_key_team_id not found in span attributes. Available: {list(span_attributes.keys())}"
        assert (
            span_attributes["metadata.user_api_key_team_id"] == jwt_team_id
        ), f"Expected team_id '{jwt_team_id}', got '{span_attributes.get('metadata.user_api_key_team_id')}'"

        # Also verify other related metadata fields
        assert "metadata.user_api_key_hash" in span_attributes
        assert "metadata.user_api_key_team_alias" in span_attributes
        assert "metadata.user_api_key_user_email" in span_attributes

        # Clear exporter
        in_memory_exporter.clear()

    def test_metadata_with_none_user_team_ids(self, otel_logger, in_memory_exporter):
        """
        Test that None user_id and team_id are handled gracefully.

        When JWT auth doesn't provide user_id or team_id, they should be None
        and should still be logged (as empty string after safe_set_attribute conversion).
        """
        metadata: StandardLoggingMetadata = {
            "user_api_key_hash": "hashed-jwt-abc123",
            "user_api_key_alias": None,
            "user_api_key_spend": 0.0,
            "user_api_key_max_budget": None,
            "user_api_key_budget_reset_at": None,
            "user_api_key_team_id": None,  # No team_id from JWT
            "user_api_key_user_id": None,  # No user_id from JWT
            "user_api_key_org_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_end_user_id": None,
            "user_api_key_request_route": "/v1/chat/completions",
            "user_api_key_user_email": None,
            "user_api_key_auth_metadata": {},
            "spend_logs_metadata": None,
            "requester_ip_address": "127.0.0.1",
            "requester_metadata": None,
            "requester_custom_headers": {},
            "prompt_management_metadata": None,
            "mcp_tool_call_metadata": None,
            "vector_store_request_metadata": None,
            "applied_guardrails": None,
            "usage_object": None,
            "cold_storage_object_key": None,
        }

        standard_logging_payload: StandardLoggingPayload = {
            "id": "test-id-456",
            "trace_id": "trace-456",
            "call_type": "completion",
            "cache_hit": None,
            "stream": False,
            "status": "success",
            "status_fields": {"llm_api_status": "success", "guardrail_status": "not_run"},
            "custom_llm_provider": "openai",
            "saved_cache_cost": 0.0,
            "startTime": datetime.now().timestamp(),
            "endTime": datetime.now().timestamp(),
            "completionStartTime": datetime.now().timestamp(),
            "response_time": 0.5,
            "model": "gpt-3.5-turbo",
            "metadata": metadata,
            "cache_key": None,
            "response_cost": 0.001,
            "cost_breakdown": None,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "request_tags": [],
            "end_user": "",
            "api_base": "https://api.openai.com",
            "model_group": "gpt-3.5-turbo",
            "model_id": "model-456",
            "requester_ip_address": "127.0.0.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "response": {
                "id": "chatcmpl-456",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi!"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-3.5-turbo",
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 50,
                    "total_tokens": 100,
                },
            },
            "model_parameters": {},
            "hidden_params": {
                "model_id": "model-456",
                "cache_key": None,
                "api_base": None,
                "response_cost": None,
                "litellm_overhead_time_ms": None,
                "additional_headers": None,
                "batch_models": None,
                "litellm_model_name": None,
                "usage_object": None,
            },
            "model_map_information": {"model_map_key": "gpt-3.5-turbo", "model_map_value": None},
            "error_str": None,
            "error_information": {
                "error_code": "",
                "error_class": "",
                "llm_provider": "",
                "traceback": "",
                "error_message": "",
            },
            "response_cost_failure_debug_info": None,
            "guardrail_information": None,
            "standard_built_in_tools_params": {"web_search_options": None, "file_search": None},
        }

        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {},
            "litellm_params": {
                "custom_llm_provider": "openai",
                "metadata": {},
            },
            "standard_logging_object": standard_logging_payload,
        }

        response_obj = {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "total_tokens": 100,
            },
        }

        start_time = datetime.now()
        end_time = datetime.now()

        # Call the success handler - should not raise any errors
        otel_logger.log_success_event(kwargs, response_obj, start_time, end_time)

        # Get the finished spans
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) >= 1

        # Find the litellm_request span
        litellm_request_span = None
        for span in spans:
            if span.name == "litellm_request":
                litellm_request_span = span
                break

        assert litellm_request_span is not None

        span_attributes = dict(litellm_request_span.attributes)
        print("Span attributes with None values:", span_attributes)

        # None values should be converted to empty strings by safe_set_attribute
        assert "metadata.user_api_key_user_id" in span_attributes
        assert "metadata.user_api_key_team_id" in span_attributes
        # The cast_as_primitive_value_type converts None to ""
        assert span_attributes["metadata.user_api_key_user_id"] == ""
        assert span_attributes["metadata.user_api_key_team_id"] == ""

        in_memory_exporter.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

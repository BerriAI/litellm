import json
import os
import sys
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm.integrations.arize.arize import ArizeLogger
from litellm.integrations.opentelemetry import OpenTelemetryConfig


@pytest.mark.asyncio
async def test_arize_dynamic_params():
    """Test that the OpenTelemetry logger uses the correct dynamic headers for each Arize request."""
    
    # Create ArizeLogger instance
    arize_logger = ArizeLogger()
    
    # Capture the get_tracer_to_use_for_request calls
    tracer_calls = []
    original_get_tracer = arize_logger.get_tracer_to_use_for_request
    
    def mock_get_tracer_to_use_for_request(kwargs):
        # Capture the kwargs to see what dynamic headers are being used
        tracer_calls.append(kwargs)
        # Return the default tracer
        return arize_logger.tracer
    
    # Mock the get_tracer_to_use_for_request method
    arize_logger.get_tracer_to_use_for_request = mock_get_tracer_to_use_for_request
    
    # Set up callbacks
    litellm.callbacks = [arize_logger]
    
    # First request with team1 credentials
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from arize dynamic config"}],
        temperature=0.1,
        mock_response="test_response",
        arize_api_key="team1_key",
        arize_space_id="team1_space_id"
    )

    # Second request with team2 credentials
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from arize dynamic config"}],
        temperature=0.1,
        mock_response="test_response",
        arize_api_key="team2_key",
        arize_space_id="team2_space_id"
    )

    # Allow some time for async processing
    await asyncio.sleep(5)

    # Assertions
    print(f"Tracer calls: {len(tracer_calls)}")
    
    # We should have captured calls for both requests
    assert len(tracer_calls) >= 2, f"Expected at least 2 tracer calls, got {len(tracer_calls)}"
    
    # Check that we have the expected dynamic params in the kwargs
    team1_found = False
    team2_found = False

    print("args to tracer calls", tracer_calls)
    
    for call_kwargs in tracer_calls:
        dynamic_params = call_kwargs.get("standard_callback_dynamic_params", {})
        if dynamic_params.get("arize_api_key") == "team1_key":
            team1_found = True
            assert dynamic_params.get("arize_space_id") == "team1_space_id"
        elif dynamic_params.get("arize_api_key") == "team2_key":
            team2_found = True
            assert dynamic_params.get("arize_space_id") == "team2_space_id"
    
    # Verify both teams were found
    assert team1_found, "team1 dynamic params not found"
    assert team2_found, "team2 dynamic params not found"
    
    print("✅ All assertions passed - OpenTelemetry logger correctly received dynamic params")


@pytest.mark.asyncio
async def test_arize_dynamic_headers_in_grpc_requests():
    """Test that dynamic Arize params are passed as headers to the gRPC/HTTP exporter."""
    
    # Track all exporter calls and their headers
    exporter_headers = []
    
    def mock_otlp_http_exporter(*args, **kwargs):
        # Capture the headers passed to the HTTP exporter
        headers = kwargs.get('headers', {})
        exporter_headers.append(headers)
        
        # Return a mock exporter
        mock_exporter = MagicMock()
        mock_exporter.export = MagicMock(return_value=None)
        return mock_exporter
    
    # Patch the HTTP exporter (Arize uses HTTP by default)
    with patch('opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter', mock_otlp_http_exporter):
        
        # Create ArizeLogger with HTTP configuration
        config = OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint="https://otlp.arize.com/v1"
        )
        arize_logger = ArizeLogger(config=config)
        litellm.callbacks = [arize_logger]
        
        # Request 1: team1 dynamic params
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi from team1"}],
            mock_response="response1",
            arize_api_key="team1_api_key",
            arize_space_id="team1_space_id"
        )

        # Request 2: team2 dynamic params
        await litellm.acompletion(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "hi from team2"}],
            mock_response="response2",
            arize_api_key="team2_api_key",
            arize_space_id="team2_space_id"
        )

        # Allow time for async processing
        await asyncio.sleep(3)

        # Assertions
        print(f"Captured exporter headers: {exporter_headers}")
        
        # Should have multiple exporter calls (default + dynamic)
        assert len(exporter_headers) >= 2, f"Expected at least 2 exporter calls, got {len(exporter_headers)}"
        
        # Find team1 and team2 headers
        team1_found = False
        team2_found = False
        
        for headers in exporter_headers:
            if headers.get('api_key') == 'team1_api_key' and headers.get('arize-space-id') == 'team1_space_id':
                team1_found = True
                print(f"✅ Found team1 headers: {headers}")
            elif headers.get('api_key') == 'team2_api_key' and headers.get('arize-space-id') == 'team2_space_id':
                team2_found = True  
                print(f"✅ Found team2 headers: {headers}")
        
        # Verify both dynamic header sets were used
        assert team1_found, "team1 dynamic headers not found in exporter calls"
        assert team2_found, "team2 dynamic headers not found in exporter calls"
        
        print("✅ Test passed - Dynamic Arize params correctly passed to gRPC/HTTP exporter")



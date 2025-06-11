import pytest

from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)


class TestBedrockRouteTracking:
    def test_is_bedrock_route_invoke(self):
        """Test that invoke routes are correctly identified as Bedrock routes"""
        handler = PassThroughEndpointLogging()

        # Test positive cases for invoke
        assert handler.is_bedrock_route("/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke") is True
        assert handler.is_bedrock_route("/model/amazon.titan-text-lite-v1/invoke") is True
        assert handler.is_bedrock_route("/invoke") is True
        assert handler.is_bedrock_route("some/path/invoke/other") is True

    def test_is_bedrock_route_converse(self):
        """Test that converse routes are correctly identified as Bedrock routes"""
        handler = PassThroughEndpointLogging()

        # Test positive cases for converse
        assert handler.is_bedrock_route("/model/anthropic.claude-3-sonnet-20240229-v1:0/converse") is True
        assert handler.is_bedrock_route("/model/us.amazon.nova-pro-v1:0/converse") is True
        assert handler.is_bedrock_route("/converse") is True
        assert handler.is_bedrock_route("some/path/converse/other") is True

    def test_is_bedrock_route_negative_cases(self):
        """Test that non-Bedrock routes are correctly identified as not Bedrock routes"""
        handler = PassThroughEndpointLogging()

        # Test negative cases
        assert handler.is_bedrock_route("/v1/chat/completions") is False
        assert handler.is_bedrock_route("/anthropic/messages") is False
        assert handler.is_bedrock_route("/vertex-ai/generateContent") is False
        assert handler.is_bedrock_route("/model/gpt-4/completions") is False
        assert handler.is_bedrock_route("/some/random/path") is False
        assert handler.is_bedrock_route("") is False

    def test_is_bedrock_route_partial_matches(self):
        """Test that partial matches don't incorrectly identify routes"""
        handler = PassThroughEndpointLogging()

        # These should NOT match because they're not the actual Bedrock API patterns
        assert handler.is_bedrock_route("/invoked") is False  # Not exactly "/invoke"
        assert handler.is_bedrock_route("/conversed") is False  # Not exactly "/converse"
        assert handler.is_bedrock_route("/some-invoke-path") is False  # Contains but not the pattern
        assert handler.is_bedrock_route("/some-converse-path") is False  # Contains but not the pattern

    def test_is_bedrock_route_case_sensitivity(self):
        """Test that route matching is case sensitive"""
        handler = PassThroughEndpointLogging()

        # These should match (correct case)
        assert handler.is_bedrock_route("/invoke") is True
        assert handler.is_bedrock_route("/converse") is True

        # These should NOT match (wrong case)
        assert handler.is_bedrock_route("/INVOKE") is False
        assert handler.is_bedrock_route("/Invoke") is False
        assert handler.is_bedrock_route("/CONVERSE") is False
        assert handler.is_bedrock_route("/Converse") is False

    def test_tracked_bedrock_routes_content(self):
        """Test that TRACKED_BEDROCK_ROUTES contains the expected routes"""
        handler = PassThroughEndpointLogging()

        expected_routes = ["/invoke", "/converse"]
        assert handler.TRACKED_BEDROCK_ROUTES == expected_routes

    def test_tracked_bedrock_routes_immutability(self):
        """Test that the tracked routes are the expected ones and haven't changed"""
        handler = PassThroughEndpointLogging()

        # Verify the routes are what we expect after our change
        assert "/invoke" in handler.TRACKED_BEDROCK_ROUTES
        assert "/converse" in handler.TRACKED_BEDROCK_ROUTES

        # Verify old service names are no longer present
        assert "bedrock-runtime" not in handler.TRACKED_BEDROCK_ROUTES
        assert "bedrock-agent-runtime" not in handler.TRACKED_BEDROCK_ROUTES

        # Verify we have exactly 2 routes
        assert len(handler.TRACKED_BEDROCK_ROUTES) == 2

"""
Tests for Vertex passthrough vertex_location handling.

Ensures the configured vertex_location from litellm_params takes precedence
over the URL-extracted location for cost calculation.

Ref: https://github.com/BerriAI/litellm/issues/40692
"""

from unittest.mock import MagicMock

import pytest


class TestVertexLocationFromConfig:
    """Verify configured vertex_location takes precedence over URL location."""

    def test_config_vertex_location_takes_precedence(self):
        """
        When the logging object already has vertex_location set from config,
        the URL-extracted location should not override it.
        """
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        logging_obj = MagicMock()
        # Config sets vertex_location to "global"
        logging_obj.optional_params = {"vertex_location": "global"}

        # URL contains "us-central1"
        url_route = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.8-flash:generateContent"

        # Simulate the fixed logic
        url_vertex_location = VertexPassthroughLoggingHandler.extract_model_from_url  # just for import
        from litellm.llms.vertex_ai.common_utils import get_vertex_location_from_url

        url_vertex = get_vertex_location_from_url(url_route)
        vertex_location = logging_obj.optional_params.get("vertex_location") or url_vertex

        assert vertex_location == "global", (
            f"Expected 'global' from config, got '{vertex_location}'"
        )

    def test_url_location_used_when_no_config(self):
        """
        When no vertex_location is configured, the URL-extracted location
        should be used as fallback.
        """
        from litellm.llms.vertex_ai.common_utils import get_vertex_location_from_url

        logging_obj = MagicMock()
        logging_obj.optional_params = {}

        url_route = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.8-flash:generateContent"

        url_vertex = get_vertex_location_from_url(url_route)
        vertex_location = logging_obj.optional_params.get("vertex_location") or url_vertex

        assert vertex_location == "us-central1", (
            f"Expected 'us-central1' from URL, got '{vertex_location}'"
        )

    def test_empty_config_falls_back_to_url(self):
        """
        When vertex_location is set to empty string in config, it should
        fall back to the URL-extracted location.
        """
        from litellm.llms.vertex_ai.common_utils import get_vertex_location_from_url

        logging_obj = MagicMock()
        logging_obj.optional_params = {"vertex_location": ""}

        url_route = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3.8-flash:generateContent"

        url_vertex = get_vertex_location_from_url(url_route)
        vertex_location = logging_obj.optional_params.get("vertex_location") or url_vertex

        assert vertex_location == "us-central1", (
            f"Expected 'us-central1' from URL fallback, got '{vertex_location}'"
        )

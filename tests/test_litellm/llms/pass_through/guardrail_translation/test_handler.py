"""
Tests for the guardrail_translation_mappings registry.

Validates:
- allm_passthrough_route is registered in the mappings (regression: this was the bug)
"""

from litellm.llms.pass_through.guardrail_translation import (
    guardrail_translation_mappings,
)
from litellm.llms.pass_through.guardrail_translation.handler import (
    LlmPassthroughRouteHandler,
)
from litellm.types.utils import CallTypes


class TestRegistry:
    def test_allm_passthrough_route_registered(self):
        """Regression: missing this mapping was the root cause of the bug."""
        assert CallTypes.allm_passthrough_route in guardrail_translation_mappings

    def test_allm_passthrough_route_maps_to_llm_passthrough_route_handler(self):
        assert (
            guardrail_translation_mappings[CallTypes.allm_passthrough_route]
            is LlmPassthroughRouteHandler
        )

    def test_pass_through_still_registered(self):
        from litellm.llms.pass_through.guardrail_translation.handler import (
            PassThroughEndpointHandler,
        )

        assert (
            guardrail_translation_mappings[CallTypes.pass_through]
            is PassThroughEndpointHandler
        )


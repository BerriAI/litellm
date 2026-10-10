"""
Tests for route -> CallTypes resolution (api_route_to_call_types).

Regression coverage for the guardrail route table bugs:
- placeholder segments with a literal suffix ({model}:generateContent) never matched
- the /v1beta generateContent routes were missing from the table
- /llm_passthrough listed the sync call type first, resolving consumers that
  take call_types[0] to a handler-less type
"""

from litellm.litellm_core_utils.api_route_to_call_types import (
    get_call_types_for_route,
)
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes


class TestGenerateContentRouteResolution:
    def test_bare_generate_content_route_resolves(self):
        call_types = get_call_types_for_route("/models/gemini-2.5-flash:generateContent")
        assert call_types is not None
        assert list(call_types) == [CallTypes.agenerate_content, CallTypes.generate_content]

    def test_v1beta_generate_content_route_resolves(self):
        call_types = get_call_types_for_route("/v1beta/models/gemini-2.5-flash:generateContent")
        assert call_types is not None
        assert list(call_types) == [CallTypes.agenerate_content, CallTypes.generate_content]

    def test_bare_stream_generate_content_route_resolves(self):
        call_types = get_call_types_for_route("/models/gemini-2.5-flash:streamGenerateContent")
        assert call_types is not None
        assert list(call_types) == [
            CallTypes.agenerate_content_stream,
            CallTypes.generate_content_stream,
        ]

    def test_v1beta_stream_generate_content_route_resolves(self):
        call_types = get_call_types_for_route("/v1beta/models/gemini-2.5-flash:streamGenerateContent")
        assert call_types is not None
        assert list(call_types) == [
            CallTypes.agenerate_content_stream,
            CallTypes.generate_content_stream,
        ]

    def test_slash_containing_model_name_resolves(self):
        call_types = get_call_types_for_route("/v1beta/models/gemini/gemini-2.5-flash:generateContent")
        assert call_types is not None
        assert list(call_types) == [CallTypes.agenerate_content, CallTypes.generate_content]

    def test_empty_model_name_does_not_match(self):
        assert get_call_types_for_route("/models/:generateContent") is None

    def test_unrelated_model_action_does_not_match(self):
        assert get_call_types_for_route("/models/gemini-2.5-flash:countTokens") is None


class TestPassthroughRouteOrdering:
    def test_llm_passthrough_lists_async_call_type_first(self):
        call_types = get_call_types_for_route("/llm_passthrough")
        assert call_types is not None
        assert list(call_types) == [
            CallTypes.allm_passthrough_route,
            CallTypes.llm_passthrough_route,
        ]

    def test_v1_llm_passthrough_lists_async_call_type_first(self):
        call_types = get_call_types_for_route("/v1/llm_passthrough")
        assert call_types is not None
        assert list(call_types) == [
            CallTypes.allm_passthrough_route,
            CallTypes.llm_passthrough_route,
        ]


class TestFirstCallTypeHasTranslationHandler:
    def test_first_call_type_is_translatable_whenever_any_is(self):
        """
        Consumers (unified guardrail post-call and streaming resolution) take
        call_types[0]. A route whose first call type lacks a guardrail
        translation handler while a later one has it silently skips guardrail
        scanning, so the table must list a handler-backed call type first.
        """
        from litellm.llms import load_guardrail_translation_mappings

        mappings = load_guardrail_translation_mappings()
        misordered = {
            route: [call_type.value for call_type in call_types]
            for route, call_types in API_ROUTE_TO_CALL_TYPES.items()
            if call_types
            and call_types[0] not in mappings
            and any(call_type in mappings for call_type in call_types)
        }
        assert misordered == {}


class TestExistingRouteResolutionUnchanged:
    def test_exact_route_still_resolves(self):
        call_types = get_call_types_for_route("/chat/completions")
        assert call_types is not None
        assert CallTypes.acompletion in call_types

    def test_single_segment_placeholder_still_resolves(self):
        call_types = get_call_types_for_route("/a2a/my-agent/message/send")
        assert call_types is not None
        assert list(call_types) == [CallTypes.asend_message, CallTypes.send_message]

    def test_longer_route_does_not_collapse_into_bare_placeholder_pattern(self):
        call_types = get_call_types_for_route("/responses/resp_123/input_items")
        assert call_types is not None
        assert list(call_types) == [CallTypes.alist_input_items]

    def test_unknown_route_returns_none(self):
        assert get_call_types_for_route("/not/a/real/route") is None

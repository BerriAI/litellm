"""
Tests for reasoning effort capability fields and normalize_reasoning_effort_value.

Covers:
- Commit 1: get_model_info returns supports_minimal/supports_max fields
- Commit 2: Model registry entries have correct reasoning effort fields
- Commit 3: normalize_reasoning_effort_value degradation chains + adapter translation
"""

import json
import os
from typing import Any, Dict, Optional

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.utils import (
    normalize_reasoning_effort_value,
)
from litellm.router_utils.reasoning_effort_capability import (
    resolve_supported_reasoning_efforts,
)
from litellm.utils import get_model_info


def _load_model_registry() -> Dict[str, Any]:
    """Load the root model_prices_and_context_window.json."""
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../model_prices_and_context_window.json",
    )
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Commit 1: get_model_info returns supports_minimal and supports_max fields
# ---------------------------------------------------------------------------


class TestGetModelInfoReasoningEffortFields:
    """get_model_info should expose supports_minimal_reasoning_effort and
    supports_max_reasoning_effort from the model registry."""

    def test_opus_4_6_has_supports_minimal(self):
        info = get_model_info("claude-opus-4-6")
        assert "supports_minimal_reasoning_effort" in info

    def test_opus_4_6_has_supports_max(self):
        info = get_model_info("claude-opus-4-6")
        assert "supports_max_reasoning_effort" in info

    def test_opus_4_7_has_supports_minimal(self):
        info = get_model_info("claude-opus-4-7")
        assert "supports_minimal_reasoning_effort" in info

    def test_opus_4_7_has_supports_max(self):
        info = get_model_info("claude-opus-4-7")
        assert "supports_max_reasoning_effort" in info


# ---------------------------------------------------------------------------
# Commit 2: JSON registry has correct reasoning effort fields
# ---------------------------------------------------------------------------


class TestModelRegistryReasoningEffortFields:
    """Verify specific models have the expected reasoning effort capability
    values in the JSON registry file.

    Claude models intentionally OMIT ``supports_minimal_reasoning_effort``:
    ``minimal`` is not a real Anthropic effort level (the API accepts only
    low/medium/high/xhigh/max), so LiteLLM degrades ``minimal`` to ``low``
    regardless of the flag. These tests guard against the flag being
    re-added to the Claude fleet."""

    @pytest.fixture(autouse=True)
    def _load_registry(self):
        self.registry = _load_model_registry()

    def test_opus_4_7_supports_max(self):
        entry = self.registry["claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_6_supports_max(self):
        entry = self.registry["claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_7_omits_minimal(self):
        entry = self.registry["claude-opus-4-7"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_opus_4_6_omits_minimal(self):
        entry = self.registry["claude-opus-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_sonnet_4_6_omits_minimal(self):
        entry = self.registry["anthropic.claude-sonnet-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_bedrock_opus_4_7_supports_max(self):
        entry = self.registry["anthropic.claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_vertex_opus_4_7_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_vertex_opus_4_6_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_azure_ai_opus_4_6_omits_minimal(self):
        entry = self.registry["azure_ai/claude-opus-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_azure_ai_opus_4_7_supports_max(self):
        entry = self.registry["azure_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry


# ---------------------------------------------------------------------------
# Commit 3: normalize_reasoning_effort_value
# ---------------------------------------------------------------------------


class TestNormalizeReasoningEffortValue:
    """The degradation chains, driven against the bundled map rather than hand-built flag dicts.

    A synthetic ``{"supports_max_reasoning_effort": True}`` is not a deployment the capability
    resolver can answer for, since it never says the model reasons at all, so asserting against one
    pins a shape the proxy never sees. Every case below names a real entry and the levels it
    resolves to."""

    @pytest.mark.parametrize(
        "model, provider, effort, expected",
        [
            ("claude-opus-4-7", "anthropic", "max", "max"),
            ("gpt-5.5", "azure_ai", "max", "xhigh"),
            ("gpt-5-mini", "azure", "max", "high"),
            ("gpt-5.5", "azure_ai", "xhigh", "xhigh"),
            ("gpt-5-mini", "azure", "xhigh", "high"),
            ("gpt-5-mini", "azure", "minimal", "minimal"),
            ("gpt-5.5", "azure_ai", "minimal", "low"),
        ],
    )
    def test_a_tier_degrades_to_the_nearest_level_the_entry_accepts(
        self, local_model_cost_map, model, provider, effort, expected
    ):
        assert normalize_reasoning_effort_value(effort, model, provider) == expected

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
    def test_a_tier_outside_any_chain_passes_through(self, local_model_cost_map, effort):
        assert normalize_reasoning_effort_value(effort, "claude-opus-4-7", "anthropic") == effort

    @pytest.mark.parametrize("effort, expected", [("max", "high"), ("xhigh", "high"), ("minimal", "low")])
    def test_a_model_the_map_does_not_describe_keeps_the_floor(self, local_model_cost_map, effort, expected):
        assert normalize_reasoning_effort_value(effort, "totally-made-up-model-xyz", "openai") == expected


# ---------------------------------------------------------------------------
# Commit 3: Adapter translation — adaptive thinking + output_config.effort
# ---------------------------------------------------------------------------


class TestAdapterAdaptiveThinking:
    """Test that adaptive thinking type maps correctly through the adapters."""

    def test_messages_adapter_adaptive_returns_medium_default(self):
        """Adaptive thinking returns 'medium' as default reasoning_effort."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        adapter = LiteLLMAnthropicMessagesAdapter()
        result = adapter.translate_anthropic_thinking_to_reasoning_effort(
            {"type": "adaptive"}
        )
        assert result == "medium"

    def test_messages_adapter_adaptive_overridden_by_output_config(self):
        """For adaptive thinking, output_config.effort overrides reasoning_effort."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.types.llms.anthropic import AnthropicMessagesRequest

        adapter = LiteLLMAnthropicMessagesAdapter()
        request = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        openai_kwargs, _ = adapter.translate_anthropic_to_openai(request)
        # reasoning_effort should be set (either as string or dict with effort)
        re = openai_kwargs.get("reasoning_effort")
        if isinstance(re, dict):
            assert re["effort"] == "high"
        else:
            assert re == "high"

    def test_responses_adapter_adaptive_with_output_config(self):
        """Responses adapter: adaptive thinking + output_config.effort."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
        )
        assert result is not None
        assert result["effort"] == "xhigh"

    def test_responses_adapter_adaptive_default_medium(self):
        """Responses adapter: adaptive thinking without output_config defaults to medium."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(
            thinking={"type": "adaptive"},
        )
        assert result is not None
        assert result["effort"] == "medium"


class TestAdvertisedLevelsAreTheForwardedLevels:
    """The regression this file exists for: /model_group/info and this path answered the question
    "which levels does this deployment take" through two different readers, so the proxy advertised
    kimi-k3 max while /v1/messages quietly forwarded high. Both now resolve through one owner."""

    KIMI_K3_SPELLINGS = (
        ("kimi-k3", "moonshot"),
        ("kimi-k3", "fireworks_ai"),
        ("kimi-k3-us", "fireworks_ai"),
        ("FW-Kimi-K3", "azure_ai"),
    )

    @pytest.mark.parametrize("model, provider", KIMI_K3_SPELLINGS)
    def test_a_declared_level_is_forwarded_rather_than_degraded(self, local_model_cost_map, model, provider):
        assert normalize_reasoning_effort_value("max", model, provider) == "max"

    @pytest.mark.parametrize("model, provider", KIMI_K3_SPELLINGS)
    def test_a_level_the_entry_does_not_declare_still_degrades(self, local_model_cost_map, model, provider):
        """kimi-k3 declares low, high and max, so xhigh and minimal are absent from its set and keep
        falling through the chain rather than being waved past by the presence of a declaration."""
        assert normalize_reasoning_effort_value("xhigh", model, provider) == "high"
        assert normalize_reasoning_effort_value("minimal", model, provider) == "low"

    @pytest.mark.parametrize(
        "model, provider",
        [
            ("kimi-k3", "fireworks_ai"),
            ("gpt-5-mini", "azure"),
            ("gpt-5.5", "azure_ai"),
            ("gpt-5.5-pro", "azure"),
            ("claude-opus-4-7", "anthropic"),
        ],
    )
    def test_a_degraded_tier_is_always_a_level_the_deployment_accepts(self, local_model_cost_map, model, provider):
        """The invariant as a property rather than a table: whatever the three degradable tiers
        resolve to must itself be a level the deployment accepts, so no request can arrive at a
        level the model map says the model rejects. gpt-5.5-pro is the case that makes this bite,
        refusing ``low`` outright, which is the floor the ``minimal`` chain used to stop on."""
        model_info = get_model_info(model=model, custom_llm_provider=provider)
        supported = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)

        assert supported is not None
        for effort in ("minimal", "xhigh", "max"):
            assert normalize_reasoning_effort_value(effort, model, provider) in supported

    def test_the_wider_perplexity_entry_keeps_the_levels_it_declares(self, local_model_cost_map):
        """The entry describing that reseller declares a six-level set, and every one of them is
        forwarded, which is what the declared list exists to express."""
        assert normalize_reasoning_effort_value("xhigh", "perplexity/kimi-k3", "perplexity") == "xhigh"
        assert normalize_reasoning_effort_value("minimal", "perplexity/kimi-k3", "perplexity") == "minimal"

    def test_the_minimal_chain_clears_a_deployment_that_refuses_low(self, local_model_cost_map):
        """gpt-5.5-pro accepts medium, high and xhigh only, so the nearest level to ``minimal`` it
        will actually take is ``medium``."""
        assert normalize_reasoning_effort_value("minimal", "gpt-5.5-pro", "azure") == "medium"


@pytest.fixture
def declared_effort_entry(local_model_cost_map, request):
    """Register one synthetic entry whose declared levels are whatever the test asks for, so the
    disjoint and empty declarations can be exercised without waiting for a real model to ship one.
    An operator writing this key on a config.yaml model_info block produces exactly these shapes."""
    key = f"synthetic/{request.node.name}"
    litellm.model_cost[key] = {
        "litellm_provider": "synthetic",
        "mode": "chat",
        "supports_reasoning": True,
        "reasoning_effort_levels": list(request.param),
    }
    litellm.get_model_info.cache_clear()
    try:
        yield key.removeprefix("synthetic/")
    finally:
        litellm.model_cost.pop(key, None)
        litellm.get_model_info.cache_clear()


class TestADeclarationDisjointFromTheChain:
    """A declared set wins whole, so it can exclude the levels the per-level flags treat as always
    available. The fallback therefore has to be read off that set: assuming ``medium`` emitted a
    level an entry declaring only ``max`` had said it would not take."""

    @pytest.mark.parametrize("declared_effort_entry", [("max",)], indirect=True)
    @pytest.mark.parametrize("effort", ["minimal", "xhigh"])
    def test_a_chain_that_matches_nothing_still_lands_inside_the_declaration(self, declared_effort_entry, effort):
        assert normalize_reasoning_effort_value(effort, declared_effort_entry, "synthetic") == "max"

    @pytest.mark.parametrize("declared_effort_entry", [("none", "max")], indirect=True)
    def test_a_fallback_never_silently_turns_thinking_off(self, declared_effort_entry):
        """``none`` is an off switch, so it must never be chosen as the nearest accepted level for a
        caller who explicitly asked to think."""
        assert normalize_reasoning_effort_value("minimal", declared_effort_entry, "synthetic") == "max"

    @pytest.mark.parametrize("declared_effort_entry", [()], indirect=True)
    @pytest.mark.parametrize("effort, expected", [("max", "high"), ("xhigh", "high"), ("minimal", "low")])
    def test_a_deployment_accepting_no_tier_keeps_the_historical_floor(self, declared_effort_entry, effort, expected):
        """There is no correct level to send a deployment that accepts none, so this keeps exactly
        what every deployment got before the resolver was consulted. Dropping the parameter outright
        is the real answer and belongs with the callers that build the request."""
        assert normalize_reasoning_effort_value(effort, declared_effort_entry, "synthetic") == expected

import pytest

from litellm.router_utils.reasoning_effort_capability import (
    intersect_supported_reasoning_efforts,
    resolve_supported_reasoning_efforts,
)


class TestResolveSupportedReasoningEfforts:
    def test_no_metadata_resolves_to_unknown(self):
        assert resolve_supported_reasoning_efforts({}) is None

    def test_non_reasoning_model_supports_no_efforts(self):
        assert resolve_supported_reasoning_efforts({"supports_reasoning": False}) == ()
        assert resolve_supported_reasoning_efforts({"mode": "chat", "supports_reasoning": None}) == ()

    def test_unset_flag_on_an_entry_with_no_mode_resolves_to_unknown(self):
        # The router registers a deployment absent from the model map under a synthesized entry, and
        # get_model_info then answers with supports_reasoning None just as it does for a mapped
        # non-reasoning model. Only the missing mode tells them apart, and reading the synthesized
        # one as () would let one custom model empty every level its mapped siblings agree on.
        assert resolve_supported_reasoning_efforts({"supports_reasoning": None}) is None
        assert resolve_supported_reasoning_efforts({"mode": None, "supports_reasoning": None}) is None

    def test_reasoning_model_with_no_flags_gets_the_opt_out_levels_only(self):
        # The kimi shape: supports_reasoning true, zero effort flags. medium/high are unconditional,
        # none/minimal/low are opt-out so absence means supported, xhigh/max are opt-in so absence
        # means unsupported.
        assert resolve_supported_reasoning_efforts({"supports_reasoning": True}) == (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
        )

    def test_explicit_false_removes_an_opt_out_level(self):
        # The gpt-5.5-pro shape from the model map: only medium/high/xhigh are accepted upstream.
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "supports_none_reasoning_effort": False,
                "supports_minimal_reasoning_effort": False,
                "supports_low_reasoning_effort": False,
                "supports_xhigh_reasoning_effort": True,
            }
        )
        assert resolved == ("medium", "high", "xhigh")

    def test_explicit_true_adds_the_opt_in_levels(self):
        # The claude-opus shape: xhigh and max explicitly true, everything else absent.
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "supports_xhigh_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
            }
        )
        assert resolved == ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    def test_ultra_is_opt_in(self):
        without_flag = resolve_supported_reasoning_efforts({"supports_reasoning": True})
        with_flag = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "supports_ultra_reasoning_effort": True}
        )
        assert without_flag is not None and "ultra" not in without_flag
        assert with_flag is not None and with_flag[-1] == "ultra"

    def test_opt_in_flag_set_false_stays_excluded(self):
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "supports_xhigh_reasoning_effort": False}
        )
        assert resolved is not None
        assert "xhigh" not in resolved


class TestNoneLevelPolarity:
    def test_none_stays_opt_out_off_azure(self):
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "litellm_provider": "openai", "key": "gpt-5-mini"}
        )
        assert resolved is not None and "none" in resolved

    def test_azure_without_the_flag_does_not_advertise_none(self):
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "litellm_provider": "azure", "key": "azure/unmapped-deployment"}
        )
        assert resolved == ("minimal", "low", "medium", "high")

    def test_azure_with_the_flag_advertises_none(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "azure",
                "supports_none_reasoning_effort": True,
                "key": "azure/unmapped-deployment",
            }
        )
        assert resolved is not None and "none" in resolved

    @pytest.mark.parametrize(
        "model_key",
        ["azure/gpt-5", "azure/gpt-5-mini", "azure/gpt-5-nano", "azure/gpt-5.2", "azure/gpt-5.6"],
    )
    def test_azure_advertisement_matches_the_azure_request_gate(self, model_key):
        """AzureOpenAIGPT5Config raises UnsupportedParamsError on reasoning_effort='none' for models
        it does not flag, so advertising the level there would offer routing a 400."""
        from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config
        from litellm.utils import _get_model_info_helper

        model_info = dict(_get_model_info_helper(model=model_key.split("/", 1)[1], custom_llm_provider="azure"))
        resolved = resolve_supported_reasoning_efforts(model_info)

        assert resolved is not None
        gate_accepts_none = AzureOpenAIGPT5Config._supports_reasoning_effort_level(model_key, "none")
        assert ("none" in resolved) is gate_accepts_none


class TestIntersectSupportedReasoningEfforts:
    def test_unknown_never_narrows(self):
        assert intersect_supported_reasoning_efforts(["medium", "high"], None) == ("medium", "high")
        assert intersect_supported_reasoning_efforts(None, ["medium", "high"]) == ("medium", "high")
        assert intersect_supported_reasoning_efforts(None, None) is None

    def test_intersection_keeps_canonical_order(self):
        assert intersect_supported_reasoning_efforts(
            ["max", "high", "medium", "xhigh"], ["xhigh", "medium", "minimal"]
        ) == ("medium", "xhigh")

    def test_disjoint_sets_intersect_to_empty(self):
        assert intersect_supported_reasoning_efforts(["max"], ["minimal"]) == ()

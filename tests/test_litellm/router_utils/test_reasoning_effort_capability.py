from litellm.router_utils.reasoning_effort_capability import (
    intersect_supported_reasoning_efforts,
    resolve_supported_reasoning_efforts,
)


class TestResolveSupportedReasoningEfforts:
    def test_no_metadata_resolves_to_unknown(self):
        assert resolve_supported_reasoning_efforts({}) is None

    def test_non_reasoning_model_supports_no_efforts(self):
        assert resolve_supported_reasoning_efforts({"supports_reasoning": None}) == ()
        assert resolve_supported_reasoning_efforts({"supports_reasoning": False}) == ()

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

    def test_opt_in_flag_set_false_stays_excluded(self):
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "supports_xhigh_reasoning_effort": False}
        )
        assert resolved is not None
        assert "xhigh" not in resolved


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

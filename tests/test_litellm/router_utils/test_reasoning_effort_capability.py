import pytest

from litellm.router_utils.reasoning_effort_capability import (
    deployment_is_catalog_mapped,
    intersect_supported_reasoning_efforts,
    resolve_supported_reasoning_efforts,
)


class TestDeploymentIsCatalogMapped:
    def test_a_mode_the_catalog_supplied_marks_the_deployment_mapped(self):
        assert deployment_is_catalog_mapped({"mode": "chat"}, {}) is True

    def test_a_deployment_the_catalog_never_described_is_not_mapped(self):
        assert deployment_is_catalog_mapped(None, {}) is False
        assert deployment_is_catalog_mapped({"max_input_tokens": 200000}, {}) is False

    def test_a_mode_the_operator_wrote_does_not_make_the_deployment_mapped(self):
        # Every deployment is registered in the cost map under its own id, so an operator-written
        # mode reads back identically to one the catalog supplied and would otherwise let an
        # off-map deployment empty the levels its mapped siblings agree on.
        assert deployment_is_catalog_mapped({"mode": "chat"}, {"mode": "chat", "id": "abc"}) is False


class TestProvenanceSeparatesUnknownFromNonReasoning:
    def test_an_off_map_deployment_resolves_to_unknown(self):
        # get_model_info answers supports_reasoning None both for a deployment the map never
        # described and for a mapped non-reasoning model, so reading an unset flag as () would let
        # one custom deployment empty every level its mapped siblings agree on.
        assert resolve_supported_reasoning_efforts({}, deployment_is_mapped=False) is None
        assert resolve_supported_reasoning_efforts({"supports_reasoning": None}, deployment_is_mapped=False) is None

    def test_a_mapped_deployment_the_map_calls_non_reasoning_supports_no_efforts(self):
        assert resolve_supported_reasoning_efforts({}, deployment_is_mapped=True) == ()
        assert resolve_supported_reasoning_efforts({"supports_reasoning": None}, deployment_is_mapped=True) == ()

    def test_an_explicit_false_supports_no_efforts_off_the_map_too(self):
        # The operator's own escape hatch: saying so on an off-map deployment must still empty the
        # group, since nothing else can tell the resolver that model takes no effort level.
        assert resolve_supported_reasoning_efforts({"supports_reasoning": False}, deployment_is_mapped=False) == ()


class TestResolveSupportedReasoningEfforts:
    def test_a_reasoning_model_with_no_flags_at_all_resolves_to_unknown(self):
        # 689 of the map's 854 reasoning entries carry no effort flag, and the o-series, xai and
        # bedrock nova entries among them accept neither none nor minimal, so composing a set out of
        # the opt-out defaults alone would advertise levels those providers reject.
        assert resolve_supported_reasoning_efforts({"supports_reasoning": True}, deployment_is_mapped=True) is None

    def test_explicit_false_removes_an_opt_out_level(self):
        # The gpt-5.5-pro shape from the model map: only medium/high/xhigh are accepted upstream.
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "supports_none_reasoning_effort": False,
                "supports_minimal_reasoning_effort": False,
                "supports_low_reasoning_effort": False,
                "supports_xhigh_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("medium", "high", "xhigh")

    def test_explicit_true_adds_the_opt_in_levels(self):
        # The claude-opus shape: xhigh and max explicitly true, everything else absent.
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "supports_xhigh_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("none", "minimal", "low", "medium", "high", "xhigh", "max")

    def test_opt_in_flag_set_false_stays_excluded(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "supports_minimal_reasoning_effort": True,
                "supports_xhigh_reasoning_effort": False,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("none", "minimal", "low", "medium", "high")


class TestBareModelNameFallback:
    def test_a_prefixed_entry_inherits_the_flags_of_its_unprefixed_twin(self):
        """azure/gpt-5-mini carries no effort flag while gpt-5-mini carries three, and the request
        path resolves capability flags through that same twin (#20885). Reading only the prefixed
        entry would answer unknown for a model the map fully describes."""
        from litellm.utils import _get_model_info_helper

        model_info = dict(_get_model_info_helper(model="gpt-5-mini", custom_llm_provider="azure"))

        assert resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) == (
            "minimal",
            "low",
            "medium",
            "high",
        )

    def test_the_prefixed_entry_wins_over_its_twin_per_flag(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "azure",
                "key": "azure/gpt-5-mini",
                "supports_xhigh_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("minimal", "low", "medium", "high", "xhigh")


class TestNoneLevelPolarity:
    def test_none_stays_opt_out_off_azure(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "openai",
                "key": "openai/some-reasoner",
                "supports_max_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved is not None and "none" in resolved

    def test_none_stays_opt_out_on_an_azure_model_outside_the_gpt_5_family(self):
        """AzureOpenAIGPT5Config is selected by is_model_gpt_5_model, so an azure o-series or
        anthropic deployment never reaches the gate that refuses none and must keep the level."""
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "azure",
                "key": "azure/o3",
                "supports_max_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved is not None and "none" in resolved

    def test_azure_gpt_5_without_the_flag_does_not_advertise_none(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "azure",
                "key": "azure/gpt-5-turbo",
                "supports_minimal_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("minimal", "low", "medium", "high")

    def test_azure_gpt_5_with_the_flag_advertises_none(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "azure",
                "key": "azure/gpt-5-turbo",
                "supports_none_reasoning_effort": True,
            },
            deployment_is_mapped=True,
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
        resolved = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)

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

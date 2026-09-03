import pytest

import litellm
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

    def test_per_level_flag_without_supports_reasoning_treats_as_implicit_true(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_minimal_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("none", "minimal", "low", "medium", "high")

    def test_explicit_supports_reasoning_false_wins_over_per_level_flags(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": False,
                "supports_minimal_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ()


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


class TestDeclaredEffortList:
    """reasoning_effort_levels is what the catalog DECLARES per deployment;
    ModelGroupInfo.supported_reasoning_efforts is what a group COMPUTED. test_router.py pins that
    the computed one is never seeded from model_info, so the two names must stay apart."""

    def test_a_declared_list_answers_where_no_flag_could(self):
        """No flag can drop medium, so before this key the entry could only stay silent or
        over-advertise a level the model does not document."""
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "reasoning_effort_levels": ["low", "high", "max"]},
            deployment_is_mapped=True,
        )
        assert resolved == ("low", "high", "max")

    def test_a_declared_list_wins_whole_over_the_flags(self):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "reasoning_effort_levels": ["low", "high", "max"],
                "supports_none_reasoning_effort": True,
                "supports_minimal_reasoning_effort": True,
                "supports_xhigh_reasoning_effort": True,
                "supports_max_reasoning_effort": False,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("low", "high", "max")

    def test_a_declaration_is_reordered_into_the_advertisement_order(self):
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "reasoning_effort_levels": ["max", "low", "high"]},
            deployment_is_mapped=True,
        )
        assert resolved == ("low", "high", "max")

    def test_a_declared_empty_list_empties_the_group(self):
        assert (
            resolve_supported_reasoning_efforts(
                {"supports_reasoning": True, "reasoning_effort_levels": []},
                deployment_is_mapped=True,
            )
            == ()
        )

    @pytest.mark.parametrize("declared", [["low", "bogus"], ["bogus"], ["low", 7, None]])
    def test_an_unknown_level_is_dropped_rather_than_raised(self, declared):
        """A config.yaml model_info block bypasses the map's enum schema, and one mistyped level
        must not fail every sibling on the proxy."""
        resolved = resolve_supported_reasoning_efforts(
            {"supports_reasoning": True, "reasoning_effort_levels": declared},
            deployment_is_mapped=True,
        )
        assert resolved == tuple(effort for effort in ("low",) if effort in declared)

    @pytest.mark.parametrize("malformed", ["low,high,max", {"low": True}, 3, True])
    def test_a_malformed_declaration_falls_through_to_the_flags(self, malformed):
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "reasoning_effort_levels": malformed,
                "supports_max_reasoning_effort": True,
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("none", "minimal", "low", "medium", "high", "max")

    def test_a_model_the_map_calls_non_reasoning_ignores_its_declaration(self):
        assert (
            resolve_supported_reasoning_efforts(
                {"supports_reasoning": False, "reasoning_effort_levels": ["low", "high", "max"]},
                deployment_is_mapped=True,
            )
            == ()
        )

    def test_a_declaration_is_read_through_the_bare_twin(self, monkeypatch):
        monkeypatch.setitem(
            litellm.model_cost,
            "some-declared-reasoner",
            {"supports_reasoning": True, "reasoning_effort_levels": ["low", "max"]},
        )
        resolved = resolve_supported_reasoning_efforts(
            {
                "supports_reasoning": True,
                "litellm_provider": "openai",
                "key": "openai/some-declared-reasoner",
            },
            deployment_is_mapped=True,
        )
        assert resolved == ("low", "max")


KIMI_K3_PASSTHROUGH_KEYS = (
    "azure_ai/FW-Kimi-K3",
    "moonshot/kimi-k3",
    "together_ai/moonshotai/Kimi-K3",
    "fireworks_ai/kimi-k3",
    "fireworks_ai/kimi-k3-fast",
    "fireworks_ai/kimi-k3-us",
    "fireworks_ai/accounts/fireworks/models/kimi-k3",
    "fireworks_ai/accounts/fireworks/routers/kimi-k3-fast",
    "fireworks_ai/accounts/fireworks/routers/kimi-k3-us",
)
KIMI_K3_PERPLEXITY_KEY = "perplexity/perplexity/kimi-k3"


class TestKimiK3AdvertisesItsDocumentedLevels:
    @pytest.mark.parametrize("model_key", KIMI_K3_PASSTHROUGH_KEYS)
    def test_a_passthrough_entry_advertises_the_models_own_levels(self, local_model_cost_map, model_key):
        """platform.kimi.ai documents exactly low, high and max, and these providers forward the
        level unchanged. Undeclared, each entry resolves to unknown and the dashboard falls back to
        a capability-blind list that omits max."""
        entry = dict(litellm.model_cost[model_key], key=model_key)

        assert resolve_supported_reasoning_efforts(entry, deployment_is_mapped=True) == ("low", "high", "max")

    def test_the_perplexity_entry_advertises_the_wider_set_it_maps_down(self, local_model_cost_map):
        """Perplexity's Agent API takes a six-value enum and maps it down internally, so this
        deployment is legitimately wider than a passthrough. One blanket list could not say both."""
        entry = dict(litellm.model_cost[KIMI_K3_PERPLEXITY_KEY], key=KIMI_K3_PERPLEXITY_KEY)

        assert resolve_supported_reasoning_efforts(entry, deployment_is_mapped=True) == (
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )

    @pytest.mark.parametrize("model, provider", [("kimi-k3", "moonshot"), ("kimi-k3", "fireworks_ai")])
    def test_the_declaration_survives_model_info_hydration(self, local_model_cost_map, model, provider):
        """The hydration line is the load-bearing seam: without it the key the map carries never
        reaches the resolver and reads as absent everywhere downstream."""
        from litellm.utils import _get_model_info_helper

        model_info = dict(_get_model_info_helper(model=model, custom_llm_provider=provider))

        assert model_info["reasoning_effort_levels"] == ["low", "high", "max"]
        assert resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) == ("low", "high", "max")

    def test_a_kimi_k3_deployment_now_narrows_a_mixed_group(self, local_model_cost_map):
        """kimi used to contribute unknown, which never narrows, so the group advertised whatever
        its other deployments agreed on."""
        kimi = resolve_supported_reasoning_efforts(
            dict(litellm.model_cost["fireworks_ai/kimi-k3"], key="fireworks_ai/kimi-k3"),
            deployment_is_mapped=True,
        )

        assert intersect_supported_reasoning_efforts(("none", "minimal", "low", "medium", "high", "xhigh"), kimi) == (
            "low",
            "high",
        )

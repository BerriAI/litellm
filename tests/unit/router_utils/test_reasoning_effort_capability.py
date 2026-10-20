import json
import logging
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm._logging import JsonFormatter, verbose_logger
from litellm.router_utils.reasoning_effort_capability import (
    REASONING_EFFORT_STRENGTH_ORDER,
    RequestedReasoningEffort,
    deployment_is_catalog_mapped,
    intersect_supported_reasoning_efforts,
    nearest_declared_reasoning_effort,
    resolve_deployment_reasoning_effort,
    resolve_supported_reasoning_efforts,
    thinking_budget_effort,
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
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model="gpt-5-mini", custom_llm_provider="azure"))

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
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model=model_key.split("/", 1)[1], custom_llm_provider="azure"))
        resolved = resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True)

        assert resolved is not None
        gate_accepts_none = AzureOpenAIGPT5Config.supports_reasoning_effort_level(model_key, "none")
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
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model=model, custom_llm_provider=provider))

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


class TestGpt6AstraAdvertisesItsDocumentedLevels:
    def test_the_entry_advertises_low_through_max_without_none(self, local_model_cost_map):
        """OpenAI documents low, medium, high, xhigh and max for gpt-6-astra. Unlike gpt-5.6-sol it
        does not take none, so a group must not offer none and must offer max."""
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model="gpt-6-astra", custom_llm_provider="openai"))

        assert resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) == (
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )

    @pytest.mark.parametrize(
        "model,custom_llm_provider",
        [
            ("azure/gpt-6-astra", "azure"),
            ("azure/us/gpt-6-astra", "azure"),
            ("azure/eu/gpt-6-astra", "azure"),
            ("azure_ai/gpt-6-astra", "azure_ai"),
        ],
    )
    def test_an_azure_hosted_deployment_advertises_the_levels_both_its_routes_accept(
        self, local_model_cost_map, model, custom_llm_provider
    ):
        """Verified live on 2026-10-01 against a Foundry gpt-6-astra deployment (model
        gpt-6-astra-2026-09-03): none returns 400 unsupported_value on chat and on Responses, naming
        low, medium, high and xhigh as the levels it takes; Responses also honors max with reasoning
        tokens billed while chat refuses it, as OpenAI's chat does. The flag is route-blind, so the
        row keeps max off and the advertised set stops at the level every route accepts.
        Earlier evidence (2026-09-05, commit e79f3ec5205) had none accepted on chat; the deployment
        no longer behaves that way."""
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider))

        assert resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) == (
            "low",
            "medium",
            "high",
            "xhigh",
        )


class TestGpt6SolAndLunaAdvertiseNoneThroughMax:
    @pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
    def test_the_entry_advertises_none_through_max(self, local_model_cost_map, model):
        """OpenAI documents none, low, medium (default), high, xhigh and max for both. Unlike
        gpt-6-astra they take none."""
        from litellm.utils import get_model_info_helper

        model_info = dict(get_model_info_helper(model=model, custom_llm_provider="openai"))

        assert resolve_supported_reasoning_efforts(model_info, deployment_is_mapped=True) == (
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        )


class TestNearestDeclaredReasoningEffort:
    def test_a_declared_level_is_kept(self):
        assert nearest_declared_reasoning_effort("high", ("none", "high")) == "high"
        assert nearest_declared_reasoning_effort("none", ("none", "high")) == "none"

    def test_an_undeclared_level_rounds_up_to_the_next_declared_one(self):
        assert nearest_declared_reasoning_effort("medium", ("none", "high")) == "high"
        assert nearest_declared_reasoning_effort("minimal", ("low", "high", "max")) == "low"
        assert nearest_declared_reasoning_effort("xhigh", ("low", "high", "max")) == "max"

    def test_none_is_a_switch_that_is_never_rounded_in_either_direction(self):
        assert nearest_declared_reasoning_effort("none", ("low", "high", "max")) == "none"
        assert nearest_declared_reasoning_effort("medium", ("none",)) == "medium"

    def test_a_level_above_the_ceiling_takes_the_strongest_declared_one(self):
        assert nearest_declared_reasoning_effort("max", ("none", "high")) == "high"
        assert nearest_declared_reasoning_effort("xhigh", ("none", "low", "medium", "high")) == "high"

    def test_a_level_outside_the_strength_order_is_left_for_upstream(self):
        assert nearest_declared_reasoning_effort("turbo", ("none", "high")) == "turbo"
        assert nearest_declared_reasoning_effort("medium", ()) == "medium"

    def test_rounding_down_takes_the_strongest_declared_level_at_most_as_strong(self) -> None:
        assert nearest_declared_reasoning_effort("medium", ("low", "high", "max"), "down") == "low"
        assert nearest_declared_reasoning_effort("xhigh", ("low", "high", "max"), "down") == "high"
        assert nearest_declared_reasoning_effort("high", ("low", "high", "max"), "down") == "high"

    def test_rounding_down_below_the_floor_takes_the_weakest_declared_one(self) -> None:
        assert nearest_declared_reasoning_effort("minimal", ("none", "low", "high"), "down") == "low"

    def test_rounding_down_leaves_none_and_unknown_levels_alone(self) -> None:
        assert nearest_declared_reasoning_effort("none", ("low", "high"), "down") == "none"
        assert nearest_declared_reasoning_effort("turbo", ("low", "high"), "down") == "turbo"


KIMI_DEPLOYMENT: Final = {"id": "deployment-1", "reasoning_effort_levels": ["low", "high", "max", "none"]}


def _sent(
    requested: RequestedReasoningEffort,
    deployment_model_info: Mapping[str, object] | None = KIMI_DEPLOYMENT,
    *,
    model: str = "my-kimi",
    provider: str = "hosted_vllm",
    drop_params: bool = False,
    model_group: str = "kimi-k3",
) -> RequestedReasoningEffort:
    return resolve_deployment_reasoning_effort(
        requested,
        model=model,
        custom_llm_provider=provider,
        deployment_model_info=deployment_model_info,
        model_group=model_group,
        drop_params=drop_params,
    ).value


def _bare_map_model_declaring_fewer_levels() -> tuple[str, str]:
    def declares_fewer(key: str, entry: Mapping[str, object]) -> bool:
        levels: Final = entry.get("reasoning_effort_levels")
        return "/" not in key and isinstance(levels, list) and not set(REASONING_EFFORT_STRENGTH_ORDER) <= set(levels)

    model, entry = next((key, entry) for key, entry in litellm.model_cost.items() if declares_fewer(key, entry))
    return str(entry["litellm_provider"]), model


class TestResolveDeploymentReasoningEffort:
    @pytest.mark.parametrize(
        "requested, expected",
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("xhigh", "max"),
            ("max", "max"),
            ("none", "none"),
            ("turbo", "turbo"),
            ("default", None),
            ({"effort": "medium", "summary": "auto"}, {"effort": "high", "summary": "auto"}),
            ({"effort": "default", "summary": "auto"}, {"summary": "auto"}),
            ({"effort": "default"}, None),
        ],
    )
    def test_a_declaring_deployment_receives_a_level_it_declared(
        self, requested: RequestedReasoningEffort, expected: RequestedReasoningEffort
    ) -> None:
        assert _sent(requested) == expected

    def test_a_deployment_that_declares_nothing_gets_the_request_untouched(self, local_model_cost_map: None) -> None:
        requested: Final = {"effort": "minimal", "summary": "auto"}

        assert _sent(requested, {"id": "deployment-1"}) == requested
        assert _sent("default", None) == "default"

    def test_the_deployment_rounding_wins_over_the_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "reasoning_effort_rounding", "up")
        assert _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": "down"}) == "low"

        monkeypatch.setattr(litellm, "reasoning_effort_rounding", "down")
        assert _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": "up"}) == "high"
        assert _sent("medium") == "low"

    @pytest.mark.parametrize("spelling", ["DOWN", " Down "])
    def test_the_rounding_setting_ignores_case_and_surrounding_spaces(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        monkeypatch.setattr(litellm, "reasoning_effort_rounding", "up")
        assert _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": spelling}) == "low"

    @pytest.mark.parametrize("invalid", ["sideways", 1, None])
    def test_an_invalid_rounding_falls_back_to_the_next_setting(
        self, monkeypatch: pytest.MonkeyPatch, invalid: str | int | None
    ) -> None:
        monkeypatch.setattr(litellm, "reasoning_effort_rounding", "down")
        assert _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": invalid}) == "low"

        monkeypatch.setattr(litellm, "reasoning_effort_rounding", invalid)
        assert _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": invalid}) == "high"

    def test_a_misspelled_rounding_is_reported_rather_than_silently_rounding_up(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        verbose_logger.addHandler(caplog.handler)
        verbose_logger.propagate = False
        verbose_logger.setLevel(logging.INFO)
        monkeypatch.setattr(litellm, "reasoning_effort_rounding", None)

        _sent("medium", {**KIMI_DEPLOYMENT, "reasoning_effort_rounding": "donw\nforged"})

        assert [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING] == [
            "Ignoring reasoning_effort_rounding=donwforged, expected 'up' or 'down'"
        ]

    def test_only_the_deployment_declaration_rounds_never_the_model_map(self, local_model_cost_map: None) -> None:
        assert _sent("xhigh", None, model="kimi-k3", provider="fireworks_ai") == "xhigh"
        assert _sent("xhigh", {"id": "kimi-1"}, model="kimi-k3", provider="fireworks_ai") == "xhigh"
        assert _sent("xhigh", {"reasoning_effort_levels": ["medium"]}, model="kimi-k3", provider="fireworks_ai") == (
            "medium"
        )

    def test_a_model_info_echoing_a_map_key_rounds_nothing_the_deployment_did_not_declare(
        self, local_model_cost_map: None
    ) -> None:
        provider, model = _bare_map_model_declaring_fewer_levels()
        echoed: Final = {"key": f"{provider}/{model}", "litellm_provider": provider}

        sent: Final = [_sent(level, echoed, model=model, provider=provider) for level in REASONING_EFFORT_STRENGTH_ORDER]

        assert sent == list(REASONING_EFFORT_STRENGTH_ORDER)

    @pytest.mark.parametrize(
        "requested, expected",
        [
            ("turbo", None),
            ({"effort": "turbo", "summary": "auto"}, {"summary": "auto"}),
            ("none", "none"),
            ("medium", "high"),
        ],
    )
    def test_drop_params_drops_a_level_the_deployment_does_not_declare(
        self, requested: RequestedReasoningEffort, expected: RequestedReasoningEffort
    ) -> None:
        assert _sent(requested, drop_params=True) == expected

    def test_drop_params_drops_none_only_where_the_deployment_leaves_it_out(self) -> None:
        assert _sent("none", {"reasoning_effort_levels": ["low", "high"]}, drop_params=True) is None
        assert _sent("none", {"reasoning_effort_levels": ["low", "high"]}) == "none"

    def test_a_dropped_unrecognized_level_is_not_written_to_the_log(self, caplog: pytest.LogCaptureFixture) -> None:
        verbose_logger.addHandler(caplog.handler)
        verbose_logger.propagate = False
        verbose_logger.setLevel(logging.INFO)

        _sent("turbo\nforged line", drop_params=True)

        assert [record.getMessage() for record in caplog.records] == [
            "reasoning_effort unrecognized sent as None (rounding up) to deployment deployment-1, "
            "model group kimi-k3, model hosted_vllm/my-kimi"
        ]

    def test_line_breaks_in_logged_names_cannot_forge_a_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        verbose_logger.addHandler(caplog.handler)
        verbose_logger.propagate = False
        verbose_logger.setLevel(logging.INFO)

        _sent(
            "medium",
            {"id": "deployment-1\r\nforged id", "reasoning_effort_levels": ["low", "high"]},
            model="my-kimi\nforged model",
            model_group="kimi-k3\nforged group",
        )

        assert [record.getMessage() for record in caplog.records] == [
            "reasoning_effort medium sent as high (rounding up) to deployment deployment-1forged id, "
            "model group kimi-k3forged group, model hosted_vllm/my-kimiforged model"
        ]

    def test_a_remap_logs_one_record_naming_the_deployment_and_both_levels(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        verbose_logger.addHandler(caplog.handler)
        verbose_logger.propagate = False
        verbose_logger.setLevel(logging.INFO)

        _sent("high")
        _sent("medium")

        assert len(caplog.records) == 1
        logged: Final = json.loads(JsonFormatter().format(caplog.records[0]))
        assert logged["level"] == "INFO"
        assert {key: value for key, value in logged.items() if key.startswith("reasoning_effort_")} == {
            "reasoning_effort_deployment_id": "deployment-1",
            "reasoning_effort_model_group": "kimi-k3",
            "reasoning_effort_deployment_model": "hosted_vllm/my-kimi",
            "reasoning_effort_requested": "medium",
            "reasoning_effort_sent": "high",
            "reasoning_effort_rounding": "up",
        }


class TestAzureGpt6SolAndLunaAdvertiseTheOpenAiLevels:
    @pytest.mark.parametrize(
        "model,custom_llm_provider",
        [
            ("azure/gpt-6-sol", "azure"),
            ("azure/gpt-6-luna", "azure"),
            ("azure/eu/gpt-6-sol", "azure"),
            ("azure/eu/gpt-6-luna", "azure"),
            ("azure_ai/gpt-6-sol", "azure_ai"),
            ("azure_ai/gpt-6-luna", "azure_ai"),
        ],
    )
    def test_the_azure_entry_advertises_the_same_levels_as_openai(
        self, local_model_cost_map, model, custom_llm_provider
    ):
        """The Foundry deployments of sol and luna take the same effort set OpenAI documents for
        the direct API, so the resolved levels must match the OpenAI-direct entry."""
        from litellm.utils import get_model_info_helper

        azure_info = dict(get_model_info_helper(model=model, custom_llm_provider=custom_llm_provider))
        openai_info = dict(get_model_info_helper(model=model.rsplit("/", 1)[1], custom_llm_provider="openai"))

        assert resolve_supported_reasoning_efforts(
            azure_info, deployment_is_mapped=True
        ) == resolve_supported_reasoning_efforts(openai_info, deployment_is_mapped=True)


class TestThinkingBudgetEffort:
    @pytest.mark.parametrize(
        "model, custom_llm_provider, expected",
        [
            ("my-kimi", "hosted_vllm", "medium"),
            ("FW-Kimi-K3", "azure_ai", "medium"),
            ("claude-opus-5-5", "anthropic", "medium"),
            ("claude-sonnet-4-6", "anthropic", None),
        ],
    )
    def test_only_a_provider_that_would_send_an_effort_gets_one(
        self, model: str, custom_llm_provider: str, expected: str | None
    ) -> None:
        thinking: Final = {"type": "enabled", "budget_tokens": 2048}

        assert thinking_budget_effort(thinking, model=model, custom_llm_provider=custom_llm_provider, drop_params=False) == (
            expected
        )

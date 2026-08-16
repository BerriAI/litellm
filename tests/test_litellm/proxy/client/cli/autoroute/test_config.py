from typing import Any, Dict, Tuple

import pytest

from litellm.proxy.client.cli.commands.autoroute.config import (
    DEFAULT_KEYWORD_TIER_RULES,
    AutorouteConfig,
    ConfigGenerationError,
    DiscoveredModel,
    HeuristicClassifier,
    KeywordTierRule,
    LLMClassifier,
    NoSemanticMatching,
    SemanticMatching,
    build_generated_model_list,
    build_generated_proxy_config,
    chat_models,
    embedding_models,
    parse_discovered_models,
    validate_config,
)

DISCOVERED: Tuple[DiscoveredModel, ...] = (
    DiscoveredModel(name="gpt-4o-mini", mode="chat"),
    DiscoveredModel(name="gpt-4o", mode="chat"),
    DiscoveredModel(name="o1", mode="chat"),
    DiscoveredModel(name="text-embedding-3-small", mode="embedding"),
)


def _base_config(**overrides: Any) -> AutorouteConfig:
    defaults: Dict[str, Any] = {
        "base_url": "http://real-proxy.internal:4000",
        "api_key": "sk-real-key",
        "tiers": {
            "SIMPLE": ("gpt-4o-mini",),
            "MEDIUM": ("gpt-4o",),
            "COMPLEX": ("gpt-4o",),
            "REASONING": ("o1",),
        },
        "default_model": "gpt-4o",
    }
    defaults.update(overrides)
    return AutorouteConfig(**defaults)


class TestParseDiscoveredModels:
    def test_parses_valid_raw_list_into_typed_tuple(self):
        raw = [
            {
                "model_group": "gpt-4o",
                "mode": "chat",
                "input_cost_per_token": 0.01,
                "output_cost_per_token": 0.02,
            },
            {"model_group": "text-embedding-3-small", "mode": "embedding"},
        ]
        result = parse_discovered_models(raw)
        assert result == (
            DiscoveredModel(name="gpt-4o", mode="chat", input_cost_per_token=0.01, output_cost_per_token=0.02),
            DiscoveredModel(name="text-embedding-3-small", mode="embedding"),
        )

    def test_ignores_unknown_extra_fields(self):
        raw = [{"model_group": "gpt-4o", "mode": "chat", "totally_unknown_field": "whatever"}]
        result = parse_discovered_models(raw)
        assert result == (DiscoveredModel(name="gpt-4o", mode="chat"),)

    def test_missing_mode_defaults_to_chat(self):
        raw = [{"model_group": "gpt-4o"}]
        result = parse_discovered_models(raw)
        assert result[0].mode == "chat"


class TestChatAndEmbeddingFiltering:
    def test_filters_by_mode(self):
        models = (
            DiscoveredModel(name="gpt-4o", mode="chat"),
            DiscoveredModel(name="text-embedding-3-small", mode="embedding"),
            DiscoveredModel(name="claude", mode="chat"),
        )
        assert chat_models(models) == (models[0], models[2])
        assert embedding_models(models) == (models[1],)


class TestBuildGeneratedModelList:
    def test_dedups_model_used_in_multiple_roles(self):
        config = _base_config(classifier=LLMClassifier(model="gpt-4o"))
        model_list = build_generated_model_list(config)
        gpt4o_entries = [m for m in model_list if m["model_name"] == "gpt-4o"]
        assert len(gpt4o_entries) == 1

    def test_every_proxy_deployment_points_back_at_customer_proxy(self):
        config = _base_config()
        model_list = build_generated_model_list(config)
        proxy_entries = [m for m in model_list if m["model_name"] not in ("autorouter", "*")]
        names = {m["model_name"] for m in proxy_entries}
        assert names == {"gpt-4o-mini", "gpt-4o", "o1"}
        for entry in proxy_entries:
            assert entry["litellm_params"]["model"] == f"litellm_proxy/{entry['model_name']}"
            assert entry["litellm_params"]["api_base"] == config.base_url
            assert entry["litellm_params"]["api_key"] == config.api_key

    def test_no_wildcard_deployment_is_generated(self):
        # A bare "*" model_name looks like the obvious catch-all, but Router's auto-router
        # registry is keyed by the literal requested model string with no wildcard resolution
        # (litellm/router.py:10711-10717), so a "*" entry here would silently never match real
        # traffic. Regression guard: don't reintroduce it.
        config = _base_config()
        model_list = build_generated_model_list(config)
        assert not any(m["model_name"] == "*" for m in model_list)

    def test_complexity_router_config_reflects_llm_classifier(self):
        config = _base_config(classifier=LLMClassifier(model="gpt-4o", timeout_ms=1234))
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        router_config = autorouter["litellm_params"]["complexity_router_config"]
        assert router_config["classifier_type"] == "llm"
        assert router_config["classifier_llm_config"] == {"model": "gpt-4o", "timeout_ms": 1234}
        assert "semantic_keyword_matching" not in router_config
        assert "adaptive" not in router_config

    def test_complexity_router_config_reflects_semantic_matching(self):
        config = _base_config(
            semantic_matching=SemanticMatching(embedding_model="text-embedding-3-small", match_threshold=0.7)
        )
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        router_config = autorouter["litellm_params"]["complexity_router_config"]
        assert router_config["semantic_keyword_matching"] is True
        assert router_config["embedding_model"] == "text-embedding-3-small"
        assert router_config["match_threshold"] == 0.7
        assert router_config["keyword_tier_rules"]
        assert "classifier_type" not in router_config

    def test_semantic_matching_defaults_emit_builtin_keyword_rules(self):
        config = _base_config(semantic_matching=SemanticMatching(embedding_model="text-embedding-3-small"))
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        router_config = autorouter["litellm_params"]["complexity_router_config"]
        assert router_config["keyword_tier_rules"] == [
            {"keywords": list(rule.keywords), "tier": rule.tier} for rule in DEFAULT_KEYWORD_TIER_RULES
        ]

    def test_semantic_matching_serializes_custom_keyword_rules(self):
        config = _base_config(
            semantic_matching=SemanticMatching(
                embedding_model="text-embedding-3-small",
                keyword_tier_rules=(
                    KeywordTierRule(keywords=("yo", "sup"), tier="SIMPLE"),
                    KeywordTierRule(keywords=("architect", "design a system"), tier="COMPLEX"),
                ),
            )
        )
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        router_config = autorouter["litellm_params"]["complexity_router_config"]
        assert router_config["keyword_tier_rules"] == [
            {"keywords": ["yo", "sup"], "tier": "SIMPLE"},
            {"keywords": ["architect", "design a system"], "tier": "COMPLEX"},
        ]

    def test_complexity_router_config_reflects_adaptive(self):
        config = _base_config(adaptive=True)
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        assert autorouter["litellm_params"]["complexity_router_config"]["adaptive"] is True

    def test_default_classifier_and_semantic_matching_add_no_extra_keys(self):
        config = _base_config(classifier=HeuristicClassifier(), semantic_matching=NoSemanticMatching())
        autorouter = next(m for m in build_generated_model_list(config) if m["model_name"] == "autorouter")
        router_config = autorouter["litellm_params"]["complexity_router_config"]
        assert set(router_config.keys()) == {"tiers", "default_model"}


class TestBuildGeneratedProxyConfig:
    def test_embeds_master_key_under_general_settings(self):
        config = _base_config()
        proxy_config = build_generated_proxy_config(config, "sk-master-123")
        assert proxy_config["general_settings"] == {"master_key": "sk-master-123"}
        assert proxy_config["model_list"] == build_generated_model_list(config)


class TestValidateConfig:
    def test_passes_for_fully_valid_config(self):
        validate_config(_base_config(), DISCOVERED)

    def test_raises_for_tier_referencing_unknown_model(self):
        config = _base_config(
            tiers={
                "SIMPLE": ("unknown-model",),
                "MEDIUM": ("gpt-4o",),
                "COMPLEX": ("gpt-4o",),
                "REASONING": ("o1",),
            }
        )
        with pytest.raises(ConfigGenerationError, match="unknown-model"):
            validate_config(config, DISCOVERED)

    def test_raises_for_unknown_default_model(self):
        config = _base_config(default_model="unknown-model")
        with pytest.raises(ConfigGenerationError, match="unknown-model"):
            validate_config(config, DISCOVERED)

    def test_raises_for_unknown_llm_classifier_model(self):
        config = _base_config(classifier=LLMClassifier(model="unknown-model"))
        with pytest.raises(ConfigGenerationError, match="unknown-model"):
            validate_config(config, DISCOVERED)

    def test_raises_for_unknown_semantic_embedding_model(self):
        config = _base_config(semantic_matching=SemanticMatching(embedding_model="unknown-embedding"))
        with pytest.raises(ConfigGenerationError, match="unknown-embedding"):
            validate_config(config, DISCOVERED)

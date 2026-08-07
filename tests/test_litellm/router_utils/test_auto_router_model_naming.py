import pytest

from litellm.router_utils.auto_router_model_naming import (
    classify_strategy_router_model,
    validate_complexity_router_config_write,
    validate_strategy_router_model_write,
)

COMPLEXITY_FIELDS = frozenset({"complexity_router_config"})
SEMANTIC_FIELDS = frozenset(
    {"auto_router_config", "auto_router_default_model", "auto_router_embedding_model"}
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("anthropic/claude-sonnet-5", None),
        ("complexity_router", None),
        ("autorouter/complexity_router", None),
        ("auto_router/my-router", "semantic"),
        ("auto_router/complexity_router", "complexity"),
        ("auto_router/complexity_router-eu", "complexity"),
        ("auto_router/adaptive_router", "adaptive"),
        ("auto_router/quality_router", "quality"),
        ("auto_router/auto_router/complexity_router", "semantic"),
        ("auto_router/", "semantic"),
    ],
)
def test_classify_strategy_router_model(model, expected):
    assert classify_strategy_router_model(model) == expected


@pytest.mark.parametrize(
    "model,present_fields,expected_fragment",
    [
        ("auto_router/auto_router/complexity_router", COMPLEXITY_FIELDS, "repeats"),
        ("complexity_router", COMPLEXITY_FIELDS, "does not start with"),
        ("anthropic/claude-sonnet-5", COMPLEXITY_FIELDS, "does not start with"),
        ("auto_router/", frozenset(), "missing the router name"),
        ("auto_router/complexity_router", frozenset(), "requires"),
        ("auto_router/my-router", frozenset({"auto_router_config"}), "requires"),
        ("auto_router/adaptive_router", frozenset(), "requires"),
        ("auto_router/quality_router", frozenset(), "requires"),
    ],
)
def test_validate_rejects_incoherent_writes(model, present_fields, expected_fragment):
    violation = validate_strategy_router_model_write(model=model, present_fields=present_fields)
    assert violation is not None
    assert expected_fragment in violation


@pytest.mark.parametrize(
    "model,present_fields",
    [
        ("anthropic/claude-sonnet-5", frozenset()),
        ("openai/gpt-4o-mini", frozenset({"api_key"})),
        ("auto_router/complexity_router", COMPLEXITY_FIELDS),
        ("auto_router/complexity_router", frozenset({"complexity_router_default_model"})),
        ("auto_router/complexity_router-eu", COMPLEXITY_FIELDS),
        ("auto_router/my-router", SEMANTIC_FIELDS),
        (
            "auto_router/my-router",
            frozenset(
                {
                    "auto_router_config_path",
                    "auto_router_default_model",
                    "auto_router_embedding_model",
                }
            ),
        ),
        ("auto_router/adaptive_router", frozenset({"adaptive_router_config"})),
        ("auto_router/quality_router", frozenset({"quality_router_default_model"})),
        ("auto_router/quality_router", frozenset({"quality_router_config"})),
    ],
)
def test_validate_accepts_coherent_writes(model, present_fields):
    assert validate_strategy_router_model_write(model=model, present_fields=present_fields) is None


VALID_TIERS = {
    "SIMPLE": ["gpt-4o-mini"],
    "MEDIUM": ["gpt-4o-mini"],
    "COMPLEX": ["gpt-4o"],
    "REASONING": ["gpt-4o"],
}


@pytest.mark.parametrize(
    "keyword_tier_rules,expected_fragment",
    [
        ([{"keywords": [], "tier": "COMPLEX"}], "at least 1 item"),
        ([{"keywords": ["   "], "tier": "COMPLEX"}], "non-empty keyword"),
        (
            [{"keywords": ["invoice"], "tier": "MEDIUM"}, {"keywords": [], "tier": "COMPLEX"}],
            "at least 1 item",
        ),
    ],
)
def test_validate_rejects_unloadable_complexity_config(keyword_tier_rules, expected_fragment):
    """A rule with no keyword makes ComplexityRouterConfig unbuildable, so the row must never be
    written: without this the deployment is persisted, dropped at load, and the caller gets a 500."""
    violation = validate_complexity_router_config_write(
        complexity_router_config={
            "tiers": VALID_TIERS,
            "classifier_type": "heuristic",
            "keyword_tier_rules": keyword_tier_rules,
        }
    )
    assert violation is not None
    assert "complexity_router_config is invalid" in violation
    assert expected_fragment in violation


@pytest.mark.parametrize(
    "tier_labels,expected_fragment",
    [
        ({"SIMPLE": "Cheap", "MEDIUM": "Cheap"}, "unique across tiers"),
        ({"SIMPLE": "   "}, "non-empty"),
        ({"SIMPLE": "COMPLEX"}, "another tier's canonical name"),
    ],
)
def test_validate_rejects_ambiguous_tier_labels(tier_labels, expected_fragment):
    """Ambiguous labels must be refused at /model/new and /model/update, not at load.

    A stored config the router then refuses to build turns a 400 the operator could have fixed in
    the form into a 500 on the next proxy start.
    """
    violation = validate_complexity_router_config_write(
        complexity_router_config={
            "tiers": VALID_TIERS,
            "classifier_type": "heuristic",
            "tier_labels": tier_labels,
        }
    )
    assert violation is not None
    assert "complexity_router_config is invalid" in violation
    assert expected_fragment in violation


@pytest.mark.parametrize(
    "complexity_router_config",
    [
        {"tiers": VALID_TIERS, "classifier_type": "heuristic"},
        {
            "tiers": VALID_TIERS,
            "classifier_type": "heuristic",
            "keyword_tier_rules": [{"keywords": ["invoice", "refund"], "tier": "MEDIUM"}],
        },
        # extra="allow" on the model, so an unrecognised key is not this gate's business
        {"tiers": VALID_TIERS, "classifier_type": "heuristic", "some_future_key": "value"},
        {
            "tiers": VALID_TIERS,
            "classifier_type": "heuristic",
            "tier_labels": {"SIMPLE": "Cheap", "MEDIUM": "Standard", "COMPLEX": "Premium", "REASONING": "Deep"},
        },
        {"tiers": VALID_TIERS, "classifier_type": "heuristic", "tier_labels": {"REASONING": "Deep"}},
    ],
)
def test_validate_accepts_loadable_complexity_config(complexity_router_config):
    assert validate_complexity_router_config_write(complexity_router_config=complexity_router_config) is None


def test_naming_check_ignores_the_config_entirely():
    """The naming contract and the config's contents are separate questions with separate owners;
    a write may carry a config without naming a model, so neither can stand in for the other."""
    violation = validate_strategy_router_model_write(
        model="auto_router/complexity_router", present_fields=frozenset()
    )
    assert violation is not None
    assert "requires" in violation


def test_config_check_ignores_the_model_entirely():
    assert validate_complexity_router_config_write(complexity_router_config=None) is None
    assert (
        validate_complexity_router_config_write(
            complexity_router_config={"tiers": VALID_TIERS, "keyword_tier_rules": [{"keywords": [], "tier": "COMPLEX"}]}
        )
        is not None
    )

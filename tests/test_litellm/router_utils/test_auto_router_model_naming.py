import pytest

from litellm.router_utils.auto_router_model_naming import (
    classify_strategy_router_model,
    strategy_router_dependencies,
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


@pytest.mark.parametrize(
    "litellm_params, expected",
    [
        ({"model": "openai/gpt-4o"}, ()),
        (
            {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {"tiers": {"SIMPLE": "a", "MEDIUM": ["b", "c"]}},
                "complexity_router_default_model": "d",
            },
            (("a", "tier"), ("b", "tier"), ("c", "tier"), ("d", "default")),
        ),
        (
            {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "tiers": {"SIMPLE": "a"},
                    "classifier_type": "llm",
                    "classifier_llm_config": {"model": "clf"},
                },
            },
            (("a", "tier"), ("clf", "classifier")),
        ),
        (
            {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "tiers": {"SIMPLE": "a"},
                    "classifier_llm_config": {"model": "clf"},
                },
            },
            (("a", "tier"),),
        ),
        (
            {"model": "auto_router/my_router", "auto_router_default_model": "d", "auto_router_embedding_model": "e"},
            (("d", "default"), ("e", "embedding")),
        ),
        (
            {"model": "auto_router/adaptive_router", "adaptive_router_config": {"available_models": ["m1", "m2"]}},
            (("m1", "tier"), ("m2", "tier")),
        ),
        (
            {
                "model": "auto_router/quality_router",
                "quality_router_config": {"available_models": ["q1"], "default_model": "qd"},
            },
            (("q1", "tier"), ("qd", "default")),
        ),
    ],
)
def test_strategy_router_dependencies(litellm_params, expected):
    found = strategy_router_dependencies(litellm_params)
    assert tuple((d.model_name, d.role) for d in found) == expected


def test_complexity_default_model_param_wins_over_the_config_field():
    """ComplexityRouter overwrites config.default_model with the litellm_params one, so the
    config field is dead whenever the param is set and must not be able to red the router."""
    found = strategy_router_dependencies(
        {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {}, "default_model": "shadowed"},
            "complexity_router_default_model": "winner",
        }
    )

    assert tuple(d.model_name for d in found) == ("winner",)


def test_complexity_ignores_its_config_default_model_and_quality_does_not():
    """Router init derives a complexity default from the tiers (fallback_tier, MEDIUM, SIMPLE)
    and overwrites config.default_model, so that field names a model complexity never calls.
    Quality init really does fall back to it, so the two must not be treated alike."""
    complexity = strategy_router_dependencies(
        {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {"MEDIUM": "derived"}, "default_model": "never-called"},
        }
    )
    quality = strategy_router_dependencies(
        {
            "model": "auto_router/quality_router",
            "quality_router_config": {"available_models": ["q1"], "default_model": "really-used"},
        }
    )

    assert tuple(d.model_name for d in complexity) == ("derived",)
    assert tuple(d.model_name for d in quality) == ("q1", "really-used")


@pytest.mark.parametrize(
    "config",
    ["not-a-dict", None, {"tiers": "not-a-dict"}, {"tiers": {"SIMPLE": 7}}, {"tiers": {"SIMPLE": [None, ""]}}],
)
def test_strategy_router_dependencies_never_raises_on_a_malformed_config(config):
    """A config the router itself would refuse must not take the whole /health response down."""
    assert strategy_router_dependencies({"model": "auto_router/complexity_router", "complexity_router_config": config}) == ()


@pytest.mark.parametrize(
    "semantic_on, expected",
    [(False, ("t",)), (True, ("t", "emb"))],
)
def test_complexity_embedding_model_is_a_dependency_only_when_semantic_matching_is_on(semantic_on, expected):
    """The runtime reads embedding_model only under semantic_keyword_matching, so listing it
    unconditionally would red a router that never calls it."""
    found = strategy_router_dependencies(
        {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "t"},
                "embedding_model": "emb",
                "semantic_keyword_matching": semantic_on,
            },
        }
    )

    assert tuple(d.model_name for d in found) == expected

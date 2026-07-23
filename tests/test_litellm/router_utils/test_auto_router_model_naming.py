import pytest

from litellm.router_utils.auto_router_model_naming import (
    classify_strategy_router_model,
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

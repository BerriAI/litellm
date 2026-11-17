import logging

import pytest

import litellm
from litellm.litellm_core_utils.typed_model_config import _CONFIGS, typed_model_config
from litellm.llms.anthropic.compaction import supports_native_compaction

_CATALOG = {name: dict(row) for name, row in litellm.model_cost.items() if row.get("litellm_provider") in _CONFIGS}
_ROWS = tuple((name, row["litellm_provider"]) for name, row in _CATALOG.items())


@pytest.mark.parametrize(("model", "provider"), _ROWS)
def test_every_field_a_catalog_row_sets_reaches_the_typed_config(model, provider, caplog, monkeypatch):
    row = _CATALOG[model]
    monkeypatch.setitem(litellm.model_cost, model, row)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        config = typed_model_config(model, provider)
    assert not caplog.records
    assert config.model_dump(mode="json", exclude_none=True).keys() == {
        key for key, value in row.items() if value is not None
    }


@pytest.mark.parametrize(
    ("provider", "field"),
    [
        ("anthropic", "supports_prompt_cache_breakpoint"),
        ("anthropic", "cache_creation_input_token_cost_above_12hr"),
        ("openai", "cache_creation_input_token_cost_above_1hr"),
        ("openai", "supports_anthropic_compaction"),
    ],
)
def test_field_of_another_provider_is_dropped_with_a_warning(provider, field, caplog):
    model = next(name for name, row_provider in _ROWS if row_provider == provider)
    baseline = typed_model_config(model, provider)
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        config = typed_model_config(model, provider, {field: False})
    assert config == baseline
    assert not hasattr(config, field)
    assert field in caplog.text


def test_deployment_bookkeeping_keys_do_not_warn(caplog):
    model = next(name for name, provider in _ROWS if provider == "anthropic")
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        typed_model_config(model, "anthropic", {"id": "deployment-1", "db_model": True})
    assert not caplog.records


def test_deployment_model_info_overrides_the_catalog_row():
    model = next(name for name, row in litellm.model_cost.items() if row.get("supports_anthropic_compaction") is True)
    assert typed_model_config(model, "anthropic").supports_anthropic_compaction is True
    overridden = typed_model_config(model, "anthropic", {"supports_anthropic_compaction": False})
    assert overridden.supports_anthropic_compaction is False


def test_invalid_value_drops_only_that_field(caplog):
    model = next(name for name, provider in _ROWS if provider == "openai")
    baseline = typed_model_config(model, "openai")
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        config = typed_model_config(model, "openai", {"input_cost_per_token": -1, "max_input_tokens": 7})
    assert config.input_cost_per_token is None
    assert config.max_input_tokens == 7
    assert (
        config.model_copy(
            update={
                "input_cost_per_token": baseline.input_cost_per_token,
                "max_input_tokens": baseline.max_input_tokens,
            }
        )
        == baseline
    )
    assert "input_cost_per_token" in caplog.text


def test_unmapped_model_has_no_capabilities():
    config = typed_model_config("not-a-real-model", "anthropic")
    assert config.model_dump(exclude_unset=True) == {}
    assert supports_native_compaction({"model": "not-a-real-model"}) is False


@pytest.mark.parametrize("model", [name for name, provider in _ROWS if provider == "anthropic"])
def test_native_compaction_follows_the_catalog_flag(model):
    expected = litellm.model_cost[model].get("supports_anthropic_compaction") is True
    assert supports_native_compaction({"model": model, "custom_llm_provider": "anthropic"}) is expected


@pytest.mark.parametrize(
    ("deployment_model", "provider"), [("openai/gpt-5.4-mini", "openai"), ("anthropic/claude-sonnet-4-5", "anthropic")]
)
def test_router_registered_model_does_not_warn(deployment_model, provider, caplog, monkeypatch):
    model = deployment_model.split("/", 1)[1]
    monkeypatch.setitem(litellm.model_cost, model, dict(_CATALOG[model]))
    litellm.Router(model_list=[{"model_name": "x", "litellm_params": {"model": deployment_model, "api_key": "sk-x"}}])
    assert litellm.model_cost[model].keys() > _CATALOG[model].keys()
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        typed_model_config(model, provider)
    assert not caplog.records

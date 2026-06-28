import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "auto_update_price_and_context_window_file.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "auto_update_price_and_context_window_file", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _opus_model(model_id="claude-opus-4-9"):
    return {
        "id": model_id,
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
        "capabilities": {
            "image_input": {"supported": True},
            "pdf_input": {"supported": True},
            "structured_outputs": {"supported": True},
            "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}},
            "effort": {
                "supported": True,
                "xhigh": {"supported": True},
                "max": {"supported": True},
            },
        },
    }


def _openrouter_rows():
    return [
        {
            "id": "anthropic/claude-opus-4.9",
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000025",
                "input_cache_read": "0.0000005",
                "input_cache_write": "0.00000625",
            },
        },
        {
            "id": "openai/gpt-5.5",
            "pricing": {"prompt": "0.000005", "completion": "0.00003"},
        },
    ]


def test_canonical_id_normalizes_dots_dates_and_prefix():
    assert mod.canonical_model_id("anthropic/claude-opus-4.9") == "claude-opus-4-9"
    assert mod.canonical_model_id("claude-opus-4-5-20251101") == "claude-opus-4-5"
    assert (
        mod.canonical_model_id("anthropic/claude-opus-4.9:thinking")
        == "claude-opus-4-9"
    )


def test_price_index_prefers_base_over_variant():
    rows = [
        {
            "id": "anthropic/claude-opus-4.9:thinking",
            "pricing": {"prompt": "9", "completion": "9"},
        },
        {
            "id": "anthropic/claude-opus-4.9",
            "pricing": {"prompt": "0.000005", "completion": "0.000025"},
        },
    ]
    index = mod.build_anthropic_price_index(rows)
    assert index["claude-opus-4-9"].prompt == 0.000005
    assert index["claude-opus-4-9"].completion == 0.000025


def test_new_model_gets_price_context_and_capabilities():
    index = mod.build_anthropic_price_index(_openrouter_rows())
    entries = mod.transform_anthropic_data([_opus_model()], index, frozenset())

    assert "claude-opus-4-9" in entries
    entry = entries["claude-opus-4-9"]
    assert entry["litellm_provider"] == "anthropic"
    assert entry["mode"] == "chat"
    assert entry["max_input_tokens"] == 1000000
    assert entry["max_output_tokens"] == 128000
    assert entry["max_tokens"] == 128000
    assert entry["input_cost_per_token"] == 0.000005
    assert entry["output_cost_per_token"] == 0.000025
    assert entry["cache_read_input_token_cost"] == 0.0000005
    assert entry["cache_creation_input_token_cost"] == 0.00000625
    assert entry["supports_vision"] is True
    assert entry["supports_pdf_input"] is True
    assert entry["supports_reasoning"] is True
    assert entry["supports_adaptive_thinking"] is True
    assert entry["supports_response_schema"] is True
    assert entry["supports_xhigh_reasoning_effort"] is True
    assert entry["supports_max_reasoning_effort"] is True
    assert entry["supports_function_calling"] is True
    assert entry["supports_tool_choice"] is True
    assert entry["supports_prompt_caching"] is True


def test_model_without_price_match_is_skipped():
    index = mod.build_anthropic_price_index(_openrouter_rows())
    entries = mod.transform_anthropic_data(
        [_opus_model("claude-unlisted-model")], index, frozenset()
    )
    assert entries == {}


def test_existing_curated_model_is_not_re_added():
    index = mod.build_anthropic_price_index(_openrouter_rows())
    entries = mod.transform_anthropic_data(
        [_opus_model()], index, frozenset({"claude-opus-4-9"})
    )
    assert entries == {}


def test_false_capabilities_are_omitted():
    model = _opus_model()
    model["capabilities"]["image_input"]["supported"] = False
    model["capabilities"]["effort"]["xhigh"]["supported"] = False
    index = mod.build_anthropic_price_index(_openrouter_rows())
    entry = mod.transform_anthropic_data([model], index, frozenset())["claude-opus-4-9"]
    assert "supports_vision" not in entry
    assert "supports_xhigh_reasoning_effort" not in entry
    assert entry["supports_pdf_input"] is True

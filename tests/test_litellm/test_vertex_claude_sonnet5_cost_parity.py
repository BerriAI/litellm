"""
Regression for https://github.com/BerriAI/litellm/issues/35758

``vertex_ai/claude-sonnet-5`` must cost the same as ``vertex_ai/claude-sonnet-5@default``.

Root cause: ``_strip_stable_vertex_version`` treated the product generation
suffix ``-5`` as a Vertex publish version and rewrote bare ids to
``claude-sonnet``, while ``@default`` resource names were left intact.
"""

from __future__ import annotations

import pytest

import litellm
from litellm.cost_calculator import completion_cost, cost_per_token
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from litellm.utils import (
    _get_potential_model_names,
    _strip_model_name,
    _strip_stable_vertex_version,
    get_model_info,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Use the bundled cost map so tests do not depend on network-fetched main."""
    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    "model_name,expected",
    [
        # Product generation numbers must be preserved
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("claude-sonnet-5@default", "claude-sonnet-5@default"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-opus-4-1", "claude-opus-4-1"),
        # Real Vertex publish / date version suffixes still strip
        ("gemini-1.5-flash-001", "gemini-1.5-flash"),
        ("gemini-2.0-flash-001", "gemini-2.0-flash"),
        ("some-model-20250929", "some-model"),
    ],
)
def test_strip_stable_vertex_version_preserves_generation_numbers(model_name, expected):
    assert _strip_stable_vertex_version(model_name) == expected
    assert _strip_model_name(model_name, "vertex_ai") == expected


def test_potential_names_for_vertex_anthropic_subprovider(local_model_cost_map):
    """``vertex_ai-anthropic_models`` must still form ``vertex_ai/...`` cost keys."""
    names = _get_potential_model_names(
        model="claude-sonnet-5",
        custom_llm_provider="vertex_ai-anthropic_models",
    )
    assert names["combined_model_name"] == "vertex_ai/claude-sonnet-5"
    assert names["stripped_model_name"] == "claude-sonnet-5"
    assert names["combined_stripped_model_name"] == "vertex_ai/claude-sonnet-5"


def test_get_model_info_parity_bare_vs_default(local_model_cost_map):
    bare = get_model_info(model="vertex_ai/claude-sonnet-5")
    default = get_model_info(model="vertex_ai/claude-sonnet-5@default")
    assert bare["input_cost_per_token"] == default["input_cost_per_token"]
    assert bare["output_cost_per_token"] == default["output_cost_per_token"]
    assert bare["input_cost_per_token"] and bare["input_cost_per_token"] > 0
    # Provider-only bare id must resolve via vertex_ai cost-map prefix
    via_sub = get_model_info(
        model="claude-sonnet-5",
        custom_llm_provider="vertex_ai-anthropic_models",
    )
    assert via_sub["input_cost_per_token"] == bare["input_cost_per_token"]
    assert via_sub["key"] == "vertex_ai/claude-sonnet-5"


def _usage() -> Usage:
    return Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=200,
    )


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/claude-sonnet-5",
        "vertex_ai/claude-sonnet-5@default",
    ],
)
def test_completion_cost_nonzero_for_vertex_sonnet5(model, local_model_cost_map):
    resp = ModelResponse(
        model=model,
        choices=[Choices(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        usage=_usage(),
    )
    cost = completion_cost(
        completion_response=resp,
        model=model,
        custom_llm_provider="vertex_ai",
    )
    assert cost > 0


def test_completion_cost_parity_bare_vs_default(local_model_cost_map):
    usage = _usage()
    costs = []
    for model in ("vertex_ai/claude-sonnet-5", "vertex_ai/claude-sonnet-5@default"):
        resp = ModelResponse(
            model=model.split("/", 1)[1],
            choices=[Choices(message=Message(role="assistant", content="ok"), finish_reason="stop")],
            usage=usage,
        )
        costs.append(
            completion_cost(
                completion_response=resp,
                model=model,
                custom_llm_provider="vertex_ai",
            )
        )
    assert costs[0] == costs[1]
    assert costs[0] > 0


def test_cost_per_token_with_vertex_anthropic_provider_tag(local_model_cost_map):
    """Provider tag from the cost map must not zero out bare Claude-on-Vertex ids."""
    prompt, completion = cost_per_token(
        model="claude-sonnet-5",
        prompt_tokens=1000,
        completion_tokens=500,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=200,
        custom_llm_provider="vertex_ai-anthropic_models",
        usage_object=_usage(),
    )
    assert prompt + completion > 0

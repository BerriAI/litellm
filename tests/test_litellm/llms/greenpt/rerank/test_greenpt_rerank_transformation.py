import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import rerank_cost
from litellm.llms.greenpt.rerank.transformation import GreenPTRerankConfig


@pytest.fixture
def local_model_cost_map():
    original_model_cost = litellm.model_cost
    model_cost_path = Path(__file__).resolve().parents[5] / "model_prices_and_context_window.json"
    litellm.model_cost = json.loads(model_cost_path.read_text(encoding="utf-8"))
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_greenpt_supported_params():
    assert GreenPTRerankConfig().get_supported_cohere_rerank_params("green-rerank") == [
        "query",
        "documents",
        "top_n",
        "return_documents",
    ]


def test_validate_environment_with_api_key():
    headers = GreenPTRerankConfig().validate_environment(
        headers={},
        model="green-rerank",
        api_key="test-key",
    )
    assert headers == {
        "Authorization": "Bearer test-key",
        "accept": "application/json",
        "content-type": "application/json",
    }


def test_validate_environment_requires_api_key(monkeypatch):
    monkeypatch.delenv("GREENPT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GreenPT API key is required"):
        GreenPTRerankConfig().validate_environment(
            headers={},
            model="green-rerank",
        )


def test_transform_rerank_request_defaults_top_n():
    request = GreenPTRerankConfig().transform_rerank_request(
        model="green-rerank",
        optional_rerank_params={
            "query": "low-carbon inference",
            "documents": ["GreenPT", "Other provider"],
            "return_documents": False,
        },
        headers={},
    )
    assert request == {
        "model": "green-rerank",
        "query": "low-carbon inference",
        "documents": ["GreenPT", "Other provider"],
        "top_n": 2,
        "return_documents": False,
    }


def test_transform_rerank_request_preserves_top_n():
    request = GreenPTRerankConfig().transform_rerank_request(
        model="green-rerank",
        optional_rerank_params={
            "query": "low-carbon inference",
            "documents": ["GreenPT", "Other provider"],
            "top_n": 1,
        },
        headers={},
    )
    assert request["top_n"] == 1


def test_calculate_rerank_cost():
    prompt_cost, completion_cost = GreenPTRerankConfig().calculate_rerank_cost(
        model="green-rerank",
        billed_units={"total_tokens": 1000},
        model_info={"input_cost_per_token": 1.36524e-07},
    )
    assert prompt_cost == pytest.approx(0.000136524)
    assert completion_cost == 0.0


def test_public_rerank_cost_uses_token_pricing(local_model_cost_map):
    prompt_cost, completion_cost = rerank_cost(
        model="greenpt/green-rerank",
        custom_llm_provider=None,
        billed_units={"total_tokens": 1000},
    )
    assert prompt_cost == pytest.approx(0.000136524)
    assert completion_cost == 0.0


@pytest.mark.parametrize(
    ("billed_units", "model_info"),
    [
        (None, None),
        ({}, {"input_cost_per_token": 1.36524e-07}),
        ({"total_tokens": 1000}, {}),
    ],
)
def test_calculate_rerank_cost_missing_usage_or_pricing(billed_units, model_info):
    assert GreenPTRerankConfig().calculate_rerank_cost(
        model="green-rerank",
        billed_units=billed_units,
        model_info=model_info,
    ) == (0.0, 0.0)

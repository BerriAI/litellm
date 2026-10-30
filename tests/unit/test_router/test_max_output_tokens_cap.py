"""
Unit tests for per-deployment litellm_params.max_output_tokens_cap.

The cap only lowers output-token budgets the caller already requested; smaller
caller budgets win, and requests without any budget field are left alone (a
default budget belongs in router default_litellm_params).
"""

import pytest

from litellm import Router
from litellm.types.router import LiteLLM_Params

MODEL_LIST = [
    {
        "model_name": "capped-model",
        "litellm_params": {
            "model": "openai/capped-model",
            "api_base": "http://localhost:9911/v1",
            "api_key": "sk-not-real",
            "max_output_tokens_cap": 500,
        },
    },
    {
        "model_name": "uncapped-model",
        "litellm_params": {
            "model": "openai/uncapped-model",
            "api_base": "http://localhost:9911/v1",
            "api_key": "sk-not-real",
        },
    },
]

BUDGET_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _deployment(model_name: str) -> dict:
    for entry in MODEL_LIST:
        if entry["model_name"] == model_name:
            return entry
    raise ValueError(model_name)


def _apply_capped(kwargs: dict) -> dict:
    Router._apply_max_output_tokens_cap_to_request(deployment=_deployment("capped-model"), kwargs=kwargs)
    return kwargs


@pytest.mark.parametrize("field", BUDGET_FIELDS)
def test_cap_lowers_oversized_budget(field):
    kwargs = _apply_capped({field: 999_999})
    assert kwargs[field] == 500


@pytest.mark.parametrize("field", BUDGET_FIELDS)
def test_cap_keeps_smaller_budget(field):
    kwargs = _apply_capped({field: 200})
    assert kwargs[field] == 200


def test_cap_lowers_every_oversized_budget_in_one_request():
    kwargs = _apply_capped(
        {
            "max_tokens": 999_999,
            "max_completion_tokens": 50_000,
            "max_output_tokens": 10_000,
        }
    )
    assert kwargs["max_tokens"] == 500
    assert kwargs["max_completion_tokens"] == 500
    assert kwargs["max_output_tokens"] == 500


def test_request_without_budget_is_untouched():
    kwargs = _apply_capped({"messages": [{"role": "user", "content": "hi"}]})
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
    assert "max_output_tokens" not in kwargs


def test_uncapped_deployment_is_untouched():
    kwargs = {"max_tokens": 999_999}
    Router._apply_max_output_tokens_cap_to_request(deployment=_deployment("uncapped-model"), kwargs=kwargs)
    assert kwargs["max_tokens"] == 999_999


def test_non_int_cap_values_are_ignored():
    for bad_cap in ("500", 500.0, True, 0, -1, None):
        deployment = {
            "model_name": "capped-model",
            "litellm_params": {"model": "openai/capped-model", "max_output_tokens_cap": bad_cap},
        }
        kwargs = {"max_tokens": 999_999}
        Router._apply_max_output_tokens_cap_to_request(deployment=deployment, kwargs=kwargs)
        assert kwargs["max_tokens"] == 999_999, bad_cap


def test_pydantic_deployment_params_are_supported():
    deployment = {
        "model_name": "capped-model",
        "litellm_params": LiteLLM_Params(
            model="openai/capped-model",
            api_base="http://localhost:9911/v1",
            api_key="sk-not-real",
            max_output_tokens_cap=500,
        ),
    }
    kwargs = {"max_tokens": 999_999}
    Router._apply_max_output_tokens_cap_to_request(deployment=deployment, kwargs=kwargs)
    assert kwargs["max_tokens"] == 500


def test_cap_applies_after_default_litellm_params_at_deployment_time():
    """
    End to end through Router._update_kwargs_with_deployment: a default
    max_tokens from router default_litellm_params must also be clamped, so the
    cap is a hard ceiling for every path that adds a budget.
    """
    router = Router(
        model_list=MODEL_LIST,
        default_litellm_params={"max_tokens": 999_999},
    )
    kwargs = {
        "model": "capped-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    router._update_kwargs_with_deployment(deployment=_deployment("capped-model"), kwargs=kwargs)
    assert kwargs["max_tokens"] == 500  # default applied, then capped

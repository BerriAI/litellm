from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.proxy.utils import (
    _check_and_merge_model_level_guardrails,
    _merge_guardrails_with_existing,
)
from litellm.router import Router


def normalize(value):
    return value


def _router_with_deployment(guardrails):
    deployment = SimpleNamespace(litellm_params={"guardrails": guardrails})
    router = MagicMock()
    router.get_deployment.return_value = deployment
    return router


def _router_without_deployment():
    router = MagicMock()
    router.get_deployment.return_value = None
    return router


def test_check_and_merge_model_level_guardrails_happy_path_merges_lists():
    router = _router_with_deployment(["pii-redact", "toxic-filter"])
    data = {
        "model": "gpt-4o",
        "metadata": {
            "model_info": {"id": "deployment-123"},
            "guardrails": ["user-policy"],
        },
    }
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "model": result["model"],
        "model_info_id": result["metadata"]["model_info"]["id"],
        "guardrails_sorted": sorted(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "model": "gpt-4o",
        "model_info_id": "deployment-123",
        "guardrails_sorted": ["pii-redact", "toxic-filter", "user-policy"],
    }


def test_check_and_merge_model_level_guardrails_preserves_governance_dict():
    router = Router(
        model_list=[
            {
                "model_name": "governed-model",
                "litellm_params": {
                    "model": "openai/provider-model",
                    "api_key": "test-key",
                    "guardrails": ["input_token_limit"],
                },
                "model_info": {"id": "deployment-123"},
            }
        ]
    )
    data = {
        "model": "governed-model",
        "metadata": {
            "model_info": {"id": "deployment-123"},
            "guardrails": {
                "input_token_limit": True,
                "min_version": False,
            },
        },
    }

    result = _check_and_merge_model_level_guardrails(data, router)

    assert result["metadata"]["guardrails"] == {
        "input_token_limit": True,
        "min_version": False,
    }


def test_check_and_merge_model_level_guardrails_returns_data_when_router_none():
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m", "other": 1}
    result = _check_and_merge_model_level_guardrails(data, None)
    assert result is data
    assert normalize(result) == {
        "metadata": {"model_info": {"id": "x"}},
        "model": "m",
        "other": 1,
    }


def test_check_and_merge_model_level_guardrails_returns_data_when_model_id_missing():
    router = _router_with_deployment(["pii"])
    data = {"metadata": {"model_info": {}}, "model": "m", "extra": "v"}
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "is_same_object": result is data,
        "metadata": result["metadata"],
        "model": result["model"],
        "extra": result["extra"],
    }
    assert snapshot == {
        "is_same_object": True,
        "metadata": {"model_info": {}},
        "model": "m",
        "extra": "v",
    }
    router.get_deployment.assert_not_called()


def test_check_and_merge_model_level_guardrails_returns_data_when_deployment_none():
    router = _router_without_deployment()
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert result is data


def test_check_and_merge_model_level_guardrails_returns_data_when_guardrails_none():
    router = _router_with_deployment(None)
    data = {"metadata": {"model_info": {"id": "x"}}, "model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    assert result is data


def test_check_and_merge_model_level_guardrails_handles_missing_metadata():
    router = _router_with_deployment(["pii"])
    data = {"model": "m"}
    result = _check_and_merge_model_level_guardrails(data, router)
    snapshot = {
        "is_same_object": result is data,
        "model": result["model"],
        "metadata_present": "metadata" in result,
    }
    assert snapshot == {
        "is_same_object": True,
        "model": "m",
        "metadata_present": False,
    }


def test_check_and_merge_model_level_guardrails_raises_when_metadata_is_not_dict():
    router = _router_with_deployment(["pii"])
    data = {"metadata": "not-a-dict", "model": "m"}
    with pytest.raises(AttributeError):
        _check_and_merge_model_level_guardrails(data, router)


def test_merge_guardrails_with_existing_happy_path_combines_lists():
    data = {
        "metadata": {"guardrails": ["a", "b"], "user": "u"},
        "model": "m",
    }
    result = _merge_guardrails_with_existing(data, ["c", "a"])
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "user": result["metadata"]["user"],
        "model": result["model"],
        "is_copy": result is not data,
    }
    assert snapshot == {
        "guardrails": ["a", "b", "c"],
        "user": "u",
        "model": "m",
        "is_copy": True,
    }


def test_merge_guardrails_with_existing_wraps_scalar_existing_guardrail():
    data = {"metadata": {"guardrails": "single-policy"}}
    result = _merge_guardrails_with_existing(data, ["model-policy"])
    snapshot = {
        "guardrails_sorted": sorted(result["metadata"]["guardrails"]),
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails_sorted": ["model-policy", "single-policy"],
        "is_list": True,
        "count": 2,
    }


def test_merge_guardrails_with_existing_wraps_scalar_model_guardrail():
    data = {"metadata": {}}
    result = _merge_guardrails_with_existing(data, "model-policy")
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails": ["model-policy"],
        "is_list": True,
        "count": 1,
    }


def test_merge_guardrails_with_existing_empty_existing_empty_model_yields_empty():
    data = {"metadata": {"guardrails": None}}
    result = _merge_guardrails_with_existing(data, None)
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "is_list": isinstance(result["metadata"]["guardrails"], list),
        "count": len(result["metadata"]["guardrails"]),
    }
    assert snapshot == {
        "guardrails": [],
        "is_list": True,
        "count": 0,
    }


def test_merge_guardrails_with_existing_creates_metadata_when_missing():
    data = {"model": "m"}
    result = _merge_guardrails_with_existing(data, ["g1"])
    snapshot = {
        "guardrails": result["metadata"]["guardrails"],
        "model_preserved": result["model"],
        "original_data_unchanged": "metadata" not in data,
    }
    assert snapshot == {
        "guardrails": ["g1"],
        "model_preserved": "m",
        "original_data_unchanged": True,
    }


def test_merge_guardrails_with_existing_raises_on_unhashable_guardrail():
    data = {"metadata": {"guardrails": [{"unhashable": True}]}}
    with pytest.raises(TypeError):
        _merge_guardrails_with_existing(data, ["g1"])

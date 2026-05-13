import datetime as real_datetime
import json
import os
import sys

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock

from litellm.proxy.utils import get_custom_url, join_paths


def test_get_custom_url(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    custom_url = get_custom_url(request_base_url="http://0.0.0.0:4000", route="ui/")
    assert custom_url == "http://0.0.0.0:4000/litellm/ui/"


def test_proxy_only_error_true_for_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._is_proxy_only_llm_api_error(
        original_exception=Exception(),
        error_type=ProxyErrorTypes.auth_error,
        route="/v1/chat/completions",
    )


def test_proxy_only_error_true_for_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )
        is True
    )


def test_proxy_only_error_false_for_non_llm_non_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/generate",
        )
        is False
    )


def test_proxy_only_error_false_for_other_error_type():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=None,
            route="/v1/chat/completions",
        )
        is False
    )


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(
        base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path"
    )
    assert result == "http://0.0.0.0:4000/my-custom-path"


def test_join_paths_normal_join():
    """Test normal path joining"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="/api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_with_trailing_slash():
    """Test path joining with trailing slash on base_path"""
    result = join_paths(base_path="http://0.0.0.0:4000/", route="api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_empty_base():
    """Test path joining with empty base_path"""
    result = join_paths(base_path="", route="api/v1")
    assert result == "/api/v1"


def test_join_paths_empty_route():
    """Test path joining with empty route"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="")
    assert result == "http://0.0.0.0:4000"


def test_join_paths_both_empty():
    """Test path joining with both empty"""
    result = join_paths(base_path="", route="")
    assert result == "/"


def test_join_paths_nested_path():
    """Test path joining with nested paths"""
    result = join_paths(base_path="http://0.0.0.0:4000/v1", route="chat/completions")
    assert result == "http://0.0.0.0:4000/v1/chat/completions"


def _patch_today(monkeypatch, year, month, day):
    class PatchedDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date(year, month, day)

    monkeypatch.setattr("litellm.proxy.utils.date", PatchedDate)


def test_get_projected_spend_over_limit_day_one(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = _get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)


# ---------------------------------------------------------------------------
# L2: _enrich_http_exception_with_guardrail_context
# Regression coverage for case 2026-04-10-internal-bedrock-guardrail-streaming-error.
# ---------------------------------------------------------------------------


def test_enrich_http_exception_with_guardrail_context_dict_detail():
    """L2: dict-detail HTTPException is enriched with guardrail_name and mode."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "bedrock-pii-guard"
        event_hook = "post_call"

    exc = HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "bedrock-pii-guard"
    assert exc.detail["guardrail_mode"] == "post_call"


def test_enrich_http_exception_string_detail_noop():
    """L2: string-detail HTTPException is not mutated (can't add fields to a str)."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = HTTPException(status_code=400, detail="Content blocked")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == "Content blocked"


def test_enrich_http_exception_setdefault_does_not_overwrite():
    """L2: a guardrail that already populates guardrail_name explicitly wins."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "inferred-name"
        event_hook = "pre_call"

    exc = HTTPException(
        status_code=400,
        detail={"error": "x", "guardrail_name": "explicit-name"},
    )
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "explicit-name"


def test_enrich_http_exception_non_http_exception_noop():
    """L2: non-HTTPException is left alone and the helper does not raise."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = ValueError("not an HTTPException")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert str(exc) == "not an HTTPException"


def test_enrich_http_exception_callback_without_guardrail_name_noop():
    """L2: callback without guardrail_name attribute leaves detail alone."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        pass

    exc = HTTPException(status_code=400, detail={"error": "x"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == {"error": "x"}


def _router_with_group_info(group_info):
    router = MagicMock()
    router.get_model_group_info.return_value = group_info
    return router


def test_create_model_info_response_include_model_info():
    """`include_model_info=True` merges router group info under `model_info`.

    When false (default), the key must be absent — preserving the OpenAI
    /v1/models response shape.
    """
    from litellm.proxy.utils import create_model_info_response
    from litellm.types.router import ModelGroupInfo

    # Default: no model_info key, no router lookup.
    router = _router_with_group_info(None)
    response = create_model_info_response(
        model_id="gpt-4o", provider="openai", llm_router=router
    )
    assert "model_info" not in response
    assert response["id"] == "gpt-4o"
    assert response["owned_by"] == "openai"
    router.get_model_group_info.assert_not_called()

    # Flag on: pricing/tokens/mode/supports_* flow through.
    group_info = ModelGroupInfo(
        model_group="gpt-4o",
        providers=["openai"],
        max_input_tokens=128000,
        max_output_tokens=16384,
        input_cost_per_token=0.0000025,
        output_cost_per_token=0.00001,
        mode="chat",
        tpm=30000,
        rpm=500,
        supports_function_calling=True,
        supports_vision=True,
    )
    router = _router_with_group_info(group_info)
    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_model_info=True,
        llm_router=router,
    )
    router.get_model_group_info.assert_called_once_with(model_group="gpt-4o")
    info = response["model_info"]
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 128000
    assert info["max_output_tokens"] == 16384
    assert info["input_cost_per_token"] == 0.0000025
    assert info["output_cost_per_token"] == 0.00001
    assert info["tpm"] == 30000
    assert info["rpm"] == 500
    assert info["supports_function_calling"] is True
    assert info["supports_vision"] is True

    # Unknown / wildcard model: empty dict, not missing key.
    response = create_model_info_response(
        model_id="openai/*",
        provider="openai",
        include_model_info=True,
        llm_router=_router_with_group_info(None),
    )
    assert response["model_info"] == {}

    # Router error must not bubble — endpoint should still return a 200.
    failing = MagicMock()
    failing.get_model_group_info.side_effect = RuntimeError("boom")
    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_model_info=True,
        llm_router=failing,
    )
    assert response["model_info"] == {}

    # No router available: still emit `model_info: {}` so callers get a
    # consistent response shape regardless of proxy state.
    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_model_info=True,
        llm_router=None,
    )
    assert response["model_info"] == {}

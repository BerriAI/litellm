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

from litellm.proxy.utils import get_custom_url


def test_update_config_fields_deep_merge_db_wins():
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()

    current_config = {
        "router_settings": {
            "routing_mode": "cost_optimized",
            "model_group_alias": {
                # Existing alias with older model + different hidden flag
                "claude-sonnet-4": {
                    "model": "claude-sonnet-4-20240219",
                    "hidden": True,
                },
                # An extra alias that should remain untouched unless DB overrides it
                "legacy-sonnet": {
                    "model": "claude-2.1",
                    "hidden": True,
                },
            },
        }
    }

    db_param_value = {
        "model_group_alias": {
            # Conflict: DB should win (both 'model' and 'hidden')
            "claude-sonnet-4": {
                "model": "claude-sonnet-4-20250514",
                "hidden": False,
            },
            # New alias to be added by the merge
            "claude-sonnet-latest": {
                "model": "claude-sonnet-4-20250514",
                "hidden": True,
            },
            # Demonstrate that None values from DB are skipped (preserve existing)
            "legacy-sonnet": {
                "hidden": None  # should not clobber current True
            },
        }
    }

    updated = proxy_config._update_config_fields(
        current_config=current_config,
        param_name="router_settings",
        db_param_value=db_param_value,
    )

    rs = updated["router_settings"]
    aliases = rs["model_group_alias"]

    # DB wins on conflicts (deep) for existing alias
    assert aliases["claude-sonnet-4"]["model"] == "claude-sonnet-4-20250514"
    assert aliases["claude-sonnet-4"]["hidden"] is False

    # New alias introduced by DB is present with its values
    assert "claude-sonnet-latest" in aliases
    assert aliases["claude-sonnet-latest"]["model"] == "claude-sonnet-4-20250514"
    assert aliases["claude-sonnet-latest"]["hidden"] is True

    # None in DB does not overwrite existing values
    assert aliases["legacy-sonnet"]["model"] == "claude-2.1"
    assert aliases["legacy-sonnet"]["hidden"] is True

    # Unrelated router_settings keys are preserved
    assert rs["routing_mode"] == "cost_optimized"


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


def test_proxy_only_error_false_for_non_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
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
    from litellm.proxy.proxy_server import _get_model_group_info
    from litellm import Router

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

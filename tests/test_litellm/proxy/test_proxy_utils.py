import datetime as real_datetime
import json
import os
import sys

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import Choices, Message, ModelResponse, Usage

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


class _TrackingPostCallGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str,
        label: str,
        seen: list[str],
        output_parse_pii: bool,
        apply_to_output: bool = False,
    ):
        super().__init__(guardrail_name=guardrail_name, event_hook="post_call")
        self.label = label
        self.seen = seen
        self.output_parse_pii = output_parse_pii
        self.apply_to_output = apply_to_output

    def should_run_guardrail(self, data, event_type) -> bool:
        return event_type == GuardrailEventHooks.post_call

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        self.seen.append(self.label)
        return response


@pytest.mark.asyncio
async def test_post_call_guardrails_preserve_registration_order_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_RUN_OUTPUT_PARSE_PII_LAST", raising=False)
    seen: list[str] = []
    callbacks = [
        _TrackingPostCallGuardrail(
            guardrail_name="output-parse",
            label="output-parse",
            seen=seen,
            output_parse_pii=True,
        ),
        _TrackingPostCallGuardrail(
            guardrail_name="audit",
            label="audit",
            seen=seen,
            output_parse_pii=False,
        ),
    ]

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    response = ModelResponse(
        id="resp",
        choices=[
            Choices(
                message=Message(content="ok", role="assistant"),
                index=0,
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    monkeypatch.setattr(litellm, "callbacks", callbacks)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    await proxy_logging.post_call_success_hook(
        data={"model": "test-model"},
        response=response,
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
    )

    assert seen == ["output-parse", "audit"]


@pytest.mark.asyncio
async def test_post_call_guardrails_can_opt_in_to_run_output_parse_last(monkeypatch):
    monkeypatch.setenv("LITELLM_RUN_OUTPUT_PARSE_PII_LAST", "true")
    seen: list[str] = []
    callbacks = [
        _TrackingPostCallGuardrail(
            guardrail_name="output-parse",
            label="output-parse",
            seen=seen,
            output_parse_pii=True,
        ),
        _TrackingPostCallGuardrail(
            guardrail_name="audit",
            label="audit",
            seen=seen,
            output_parse_pii=False,
        ),
    ]

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    response = ModelResponse(
        id="resp",
        choices=[
            Choices(
                message=Message(content="ok", role="assistant"),
                index=0,
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    monkeypatch.setattr(litellm, "callbacks", callbacks)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    await proxy_logging.post_call_success_hook(
        data={"model": "test-model"},
        response=response,
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
    )

    assert seen == ["audit", "output-parse"]


@pytest.mark.asyncio
async def test_post_call_guardrails_auto_reorder_presidio_residual_masking(
    monkeypatch,
):
    monkeypatch.delenv("LITELLM_RUN_OUTPUT_PARSE_PII_LAST", raising=False)
    seen: list[str] = []
    callbacks = [
        _TrackingPostCallGuardrail(
            guardrail_name="presidio",
            label="unmask",
            seen=seen,
            output_parse_pii=True,
        ),
        _TrackingPostCallGuardrail(
            guardrail_name="presidio",
            label="residual-mask",
            seen=seen,
            output_parse_pii=False,
            apply_to_output=True,
        ),
    ]

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    response = ModelResponse(
        id="resp",
        choices=[
            Choices(
                message=Message(content="ok", role="assistant"),
                index=0,
                finish_reason="stop",
            )
        ],
        model="test-model",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )

    monkeypatch.setattr(litellm, "callbacks", callbacks)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    await proxy_logging.post_call_success_hook(
        data={"model": "test-model"},
        response=response,
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
    )

    assert seen == ["residual-mask", "unmask"]

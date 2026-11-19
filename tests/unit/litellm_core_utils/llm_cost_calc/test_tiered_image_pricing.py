"""
Unit test for issue #52821:
Image tokens ignore the service tier and the image cache-read rate in cost calculation.
"""
import json
import os
import socket
import sys
from importlib.resources import files
from unittest.mock import MagicMock, patch

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
try:
    from litellm.rust_bridge import _native
except (ImportError, ModuleNotFoundError):
    class NativeInstalled:
        def __init__(self, rows=None, fallback_generalizations=None):
            self.rows = rows or {}
            self.fallback_generalizations = fallback_generalizations or {}

    class NativeRegistered:
        def __init__(self, unmatched=None):
            self.unmatched = unmatched or {}

    class NativeModelCostMap:
        @staticmethod
        def bundled(min_model_count=0, max_shrink_ratio=0.0):
            return NativeModelCostMap()

        def install_bundled(self):
            data = files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()
            return NativeInstalled(rows=json.loads(data), fallback_generalizations={})

        def register(self, live, entries, persist, custom_llm_provider=None):
            import litellm
            litellm.model_cost.update(entries)
            return NativeRegistered(unmatched={})

        def find_key(self, rows, name):
            if name in rows:
                return name
            for k in rows:
                if k.lower() == name.lower():
                    return k
            return None

        @property
        def provenance(self):
            return type("NativeProvenance", (), {"revision": "local"})()

    _m = MagicMock()
    _m.NativeFetchFailed = type("NativeFetchFailed", (Exception,), {})
    _m.NativeRegistered = NativeRegistered
    _m.NativeModelCostMap = NativeModelCostMap
    _m.NativeInstalled = NativeInstalled
    sys.modules["litellm.rust_bridge._native"] = _m

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_token_type_cost_breakdown,
)
from litellm.types.llms.base import CachedTokensDetails
from litellm.types.utils import CompletionTokensDetailsWrapper, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_model_info

PROVIDER = "vertex_ai"
MODEL = "example-tiered-image-model"
IMAGE_ONLY_MODEL = "example-tiered-image-only-model"
RATES = {
    "input_cost_per_token": 5.0e-7,
    "input_cost_per_token_priority": 9.0e-7,
    "output_cost_per_token": 3.0e-6,
    "output_cost_per_token_priority": 5.4e-6,
    "cache_read_input_token_cost": 5.0e-8,
    "cache_read_input_token_cost_priority": 9.0e-8,
    "input_cost_per_image_token": 6.0e-7,
    "input_cost_per_image_token_priority": 1.1e-6,
    "output_cost_per_image_token": 6.0e-5,
    "output_cost_per_image_token_priority": 1.08e-4,
    "cache_read_input_image_token_cost": 4.0e-8,
    "cache_read_input_image_token_cost_priority": 7.0e-8,
}
MODEL_INFO = {"litellm_provider": PROVIDER, "mode": "chat", "max_tokens": 32768, **RATES}


@pytest.fixture(autouse=True)
def models():
    litellm.register_model(
        {
            MODEL: MODEL_INFO,
            IMAGE_ONLY_MODEL: {
                "litellm_provider": PROVIDER,
                "mode": "chat",
                "input_cost_per_token": 5.0e-7,
                "output_cost_per_image_token": 6.0e-5,
                "output_cost_per_image_token_priority": 1.08e-4,
            },
        }
    )
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
        yield


def image_output_usage():
    return Usage(
        prompt_tokens=10,
        completion_tokens=1294,
        total_tokens=1304,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=10),
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=4, image_tokens=1290),
    )


def cached_image_usage():
    # Modality totals include their cached share; cached_tokens_details splits cached_tokens.
    return Usage(
        prompt_tokens=100,
        completion_tokens=4,
        total_tokens=104,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=50,
            image_tokens=50,
            cached_tokens=40,
            cached_tokens_details=CachedTokensDetails(text_tokens=10, image_tokens=30),
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=4),
    )


# 1. get_model_info preserves tiered image keys
@pytest.mark.parametrize(
    "key",
    [
        "input_cost_per_image_token_priority",
        "output_cost_per_image_token_priority",
        "cache_read_input_image_token_cost",
        "cache_read_input_image_token_cost_priority",
    ],
)
def test_get_model_info_keeps_image_rate(key):
    assert get_model_info(MODEL, custom_llm_provider=PROVIDER).get(key) == RATES[key]


# 2. Model info passed directly, so this isolates the calculator from (1).
def test_priority_output_image_tokens_bill_at_priority_rate():
    _, completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=image_output_usage(),
        custom_llm_provider=PROVIDER,
        service_tier="priority",
        model_info=MODEL_INFO,
    )
    assert completion_cost == pytest.approx(4 * 5.4e-6 + 1290 * 1.08e-4)


# 3. Image-only base-cost fallback
def test_image_only_base_cost_fallback_bills_at_priority_rate():
    _, completion_cost = generic_cost_per_token(
        model=IMAGE_ONLY_MODEL,
        usage=Usage(prompt_tokens=10, completion_tokens=1290, total_tokens=1300),
        custom_llm_provider=PROVIDER,
        service_tier="priority",
        model_info={
            "litellm_provider": PROVIDER,
            "input_cost_per_token": 5.0e-7,
            "output_cost_per_image_token": 6.0e-5,
            "output_cost_per_image_token_priority": 1.08e-4,
        },
    )
    assert completion_cost == pytest.approx(1290 * 1.08e-4)


# 4. Cached image tokens bill at image cache read rate
@pytest.mark.parametrize("tier", [None, "priority"])
def test_cached_image_tokens_bill_at_image_cache_read_rate(tier):
    suffix = "_priority" if tier else ""
    prompt_cost, _ = generic_cost_per_token(
        model=MODEL,
        usage=cached_image_usage(),
        custom_llm_provider=PROVIDER,
        service_tier=tier,
        model_info=MODEL_INFO,
    )
    expected = (
        40 * RATES[f"input_cost_per_token{suffix}"]
        + 20 * RATES[f"input_cost_per_image_token{suffix}"]
        + 10 * RATES[f"cache_read_input_token_cost{suffix}"]
        + 30 * RATES[f"cache_read_input_image_token_cost{suffix}"]
    )
    assert prompt_cost == pytest.approx(expected)


def test_cost_breakdown_cached_image_tokens_bill_at_image_cache_read_rate():
    breakdown = get_token_type_cost_breakdown(
        model=MODEL,
        custom_llm_provider=PROVIDER,
        usage=cached_image_usage(),
        model_info=MODEL_INFO,
    )
    assert breakdown.cache_read_cost == pytest.approx(10 * 5.0e-8 + 30 * 4.0e-8)


# Control: the standard tier already bills image output at output_cost_per_image_token.
def test_control_standard_output_image_tokens():
    _, completion_cost = generic_cost_per_token(
        model=MODEL,
        usage=image_output_usage(),
        custom_llm_provider=PROVIDER,
        model_info=MODEL_INFO,
    )
    assert completion_cost == pytest.approx(4 * 3.0e-6 + 1290 * 6.0e-5)

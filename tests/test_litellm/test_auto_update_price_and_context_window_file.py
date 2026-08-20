import importlib.util
import os
import sys

import pytest

SCRIPT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        ".github",
        "workflows",
        "auto_update_price_and_context_window_file.py",
    )
)


@pytest.fixture(scope="module")
def transform_vercel_ai_gateway_data():
    spec = importlib.util.spec_from_file_location(
        "auto_update_price_and_context_window_file", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.transform_vercel_ai_gateway_data


def test_transform_skips_image_video_and_empty_pricing_rows(
    transform_vercel_ai_gateway_data,
):
    data = [
        {
            "id": "vendor/image-model",
            "context_window": 0,
            "max_tokens": 0,
            "pricing": {"image": "0.05"},
        },
        {
            "id": "vendor/video-model",
            "context_window": 0,
            "max_tokens": 0,
            "pricing": {"video_duration_pricing": "0.10"},
        },
        {
            "id": "vendor/no-pricing-model",
            "context_window": 0,
            "max_tokens": 0,
            "pricing": {},
        },
        {
            "id": "vendor/missing-pricing-model",
            "context_window": 0,
            "max_tokens": 0,
        },
    ]

    assert transform_vercel_ai_gateway_data(data) == {}


def test_transform_includes_embedding_rows_with_input_only(
    transform_vercel_ai_gateway_data,
):
    data = [
        {
            "id": "vendor/qwen3-embedding-0.6b",
            "context_window": 32768,
            "max_tokens": 32768,
            "pricing": {"input": "0.00000001"},
        }
    ]

    result = transform_vercel_ai_gateway_data(data)

    key = "vercel_ai_gateway/vendor/qwen3-embedding-0.6b"
    assert key in result
    entry = result[key]
    assert entry["mode"] == "embedding"
    assert entry["input_cost_per_token"] == pytest.approx(1e-8)
    assert entry["output_cost_per_token"] == 0.0
    assert entry["max_tokens"] == 32768
    assert entry["max_input_tokens"] == 32768
    assert entry["max_output_tokens"] == 32768
    assert entry["litellm_provider"] == "vercel_ai_gateway"


def test_transform_standard_chat_row_with_cache_pricing(
    transform_vercel_ai_gateway_data,
):
    data = [
        {
            "id": "vendor/chat-model",
            "context_window": 128000,
            "max_tokens": 16384,
            "pricing": {
                "input": "0.000001",
                "output": "0.000002",
                "input_cache_read": "0.00000025",
                "input_cache_write": "0.0000005",
            },
        }
    ]

    result = transform_vercel_ai_gateway_data(data)

    entry = result["vercel_ai_gateway/vendor/chat-model"]
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == pytest.approx(1e-6)
    assert entry["output_cost_per_token"] == pytest.approx(2e-6)
    assert entry["cache_read_input_token_cost"] == pytest.approx(2.5e-7)
    assert entry["cache_creation_input_token_cost"] == pytest.approx(5e-7)


def test_transform_mixed_payload_does_not_raise(transform_vercel_ai_gateway_data):
    data = [
        {
            "id": "vendor/image-only",
            "context_window": 0,
            "max_tokens": 0,
            "pricing": {"image": "0.05"},
        },
        {
            "id": "vendor/chat-model",
            "context_window": 8192,
            "max_tokens": 4096,
            "pricing": {"input": "0.000001", "output": "0.000002"},
        },
        {
            "id": "vendor/embedding-model",
            "context_window": 8192,
            "max_tokens": 8192,
            "pricing": {"input": "0.00000001"},
        },
    ]

    result = transform_vercel_ai_gateway_data(data)

    assert set(result.keys()) == {
        "vercel_ai_gateway/vendor/chat-model",
        "vercel_ai_gateway/vendor/embedding-model",
    }

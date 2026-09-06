import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "auto_update_price_and_context_window_file.py"

_spec = importlib.util.spec_from_file_location("auto_update_price_and_context_window", SCRIPT)
assert _spec is not None and _spec.loader is not None
updater: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(updater)


def test_vercel_transform_omits_missing_and_null_optional_fields() -> None:
    transformed = updater.transform_vercel_ai_gateway_data(
        [
            {
                "id": "alibaba/qwen3-embedding-0.6b",
                "context_window": 32768,
                "max_tokens": None,
                "pricing": {"input": "0.00000002"},
            },
            {"id": "provider/video-model"},
            {
                "id": "provider/chat-model",
                "context_window": None,
                "max_tokens": None,
                "pricing": None,
            },
        ]
    )

    assert transformed == {
        "vercel_ai_gateway/alibaba/qwen3-embedding-0.6b": {
            "input_cost_per_token": 2e-08,
            "litellm_provider": "vercel_ai_gateway",
            "max_input_tokens": 32768,
            "max_tokens": 32768,
            "mode": "embedding",
        },
        "vercel_ai_gateway/provider/video-model": {
            "litellm_provider": "vercel_ai_gateway",
            "mode": "chat",
        },
        "vercel_ai_gateway/provider/chat-model": {
            "litellm_provider": "vercel_ai_gateway",
            "mode": "chat",
        },
    }


def test_sync_preserves_existing_values_omitted_by_vercel() -> None:
    local_data = {
        "vercel_ai_gateway/alibaba/qwen3-embedding-0.6b": {
            "input_cost_per_token": 1e-08,
            "litellm_provider": "vercel_ai_gateway",
            "max_input_tokens": 32768,
            "mode": "embedding",
            "output_cost_per_token": 9e-08,
            "supports_pdf_input": True,
        }
    }
    remote_data = updater.transform_vercel_ai_gateway_data(
        [
            {
                "id": "alibaba/qwen3-embedding-0.6b",
                "context_window": 65536,
                "pricing": {"input": "0.00000002"},
            }
        ]
    )

    updater.sync_local_data_with_remote(local_data, remote_data)

    assert local_data["vercel_ai_gateway/alibaba/qwen3-embedding-0.6b"] == {
        "input_cost_per_token": 2e-08,
        "litellm_provider": "vercel_ai_gateway",
        "max_input_tokens": 65536,
        "max_tokens": 65536,
        "mode": "embedding",
        "output_cost_per_token": 9e-08,
        "supports_pdf_input": True,
    }

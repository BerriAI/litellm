import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "auto_update_price_and_context_window_file.py"

_spec = importlib.util.spec_from_file_location("auto_update_price_and_context_window_file", SCRIPT)
assert _spec is not None and _spec.loader is not None
price_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(price_sync)

LANGUAGE_ROW = {
    "id": "anthropic/claude-sonnet-4.5",
    "type": "language",
    "context_window": 200000,
    "max_tokens": 64000,
    "pricing": {
        "input": "0.000003",
        "output": "0.000015",
        "input_cache_read": "0.0000003",
        "input_cache_write": "0.00000375",
    },
}
EMBEDDING_ROW = {
    "id": "google/gemini-embedding-001",
    "type": "embedding",
    "context_window": 2048,
    "max_tokens": 0,
    "pricing": {"input": "0.00000015"},
}
UNPRICED_LANGUAGE_ROW = {
    "id": "perplexity/sonar",
    "type": "language",
    "context_window": 128000,
    "max_tokens": 8000,
}
VIDEO_ROW = {
    "id": "google/veo-3.1",
    "type": "video",
    "pricing": {"video_duration_pricing": [{"resolution": "720p", "price_per_second": "0.4"}]},
}
IMAGE_ROW = {"id": "openai/gpt-image-1", "type": "image"}


def test_vercel_rows_without_per_token_pricing_are_skipped_instead_of_crashing() -> None:
    result = price_sync.transform_vercel_ai_gateway_data(
        [LANGUAGE_ROW, EMBEDDING_ROW, UNPRICED_LANGUAGE_ROW, VIDEO_ROW, IMAGE_ROW]
    )

    assert result == {
        "vercel_ai_gateway/anthropic/claude-sonnet-4.5": {
            "max_tokens": 64000,
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "cache_read_input_token_cost": 3e-07,
            "cache_creation_input_token_cost": 3.75e-06,
            "litellm_provider": "vercel_ai_gateway",
            "mode": "chat",
        },
        "vercel_ai_gateway/google/gemini-embedding-001": {
            "max_tokens": 0,
            "max_input_tokens": 2048,
            "max_output_tokens": 0,
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 0.0,
            "litellm_provider": "vercel_ai_gateway",
            "mode": "embedding",
        },
    }


def test_write_to_file_matches_the_checked_in_cost_map_format(tmp_path: Path) -> None:
    target = tmp_path / "model_prices_and_context_window.json"

    price_sync.write_to_file(str(target), {"vercel_ai_gateway/x": {"mode": "chat", "input_cost_per_token": 1e-06}})

    assert target.read_text(encoding="utf-8") == (
        '{\n    "vercel_ai_gateway/x": {\n        "mode": "chat",\n        "input_cost_per_token": 1e-06\n    }\n}\n'
    )

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

import litellm

REPO_ROOT: Final = Path(__file__).parents[2]
MAIN_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

FLASH_TTS_KEYS: Final = ("gemini-2.5-flash-preview-tts", "gemini/gemini-2.5-flash-preview-tts")
PRO_TTS_KEYS: Final = ("gemini-2.5-pro-preview-tts", "gemini/gemini-2.5-pro-preview-tts")
NATIVE_AUDIO_KEYS: Final = tuple(
    f"{prefix}gemini-2.5-flash-native-audio-{suffix}"
    for prefix in ("", "gemini/")
    for suffix in ("latest", "preview-09-2025", "preview-12-2025")
)

LIVE_NATIVE_AUDIO_KEYS: Final = (
    "gemini-live-2.5-flash-preview-native-audio-09-2025",
    "gemini/gemini-live-2.5-flash-preview-native-audio-09-2025",
)

FLASH_TTS_INPUT: Final = 5e-07
FLASH_TTS_AUDIO_OUTPUT: Final = 1e-05
PRO_TTS_INPUT: Final = 1e-06
PRO_TTS_AUDIO_OUTPUT: Final = 2e-05
NATIVE_AUDIO_TEXT_INPUT: Final = 5e-07
NATIVE_AUDIO_AUDIO_INPUT: Final = 3e-06
NATIVE_AUDIO_TEXT_OUTPUT: Final = 2e-06
NATIVE_AUDIO_AUDIO_OUTPUT: Final = 1.2e-05

PUBLISHED_RATES: Final = {
    **{
        key: {"input_cost_per_token": FLASH_TTS_INPUT, "output_cost_per_token": FLASH_TTS_AUDIO_OUTPUT}
        for key in FLASH_TTS_KEYS
    },
    **{
        key: {"input_cost_per_token": PRO_TTS_INPUT, "output_cost_per_token": PRO_TTS_AUDIO_OUTPUT}
        for key in PRO_TTS_KEYS
    },
    **{
        key: {
            "input_cost_per_token": NATIVE_AUDIO_TEXT_INPUT,
            "input_cost_per_audio_token": NATIVE_AUDIO_AUDIO_INPUT,
            "output_cost_per_token": NATIVE_AUDIO_TEXT_OUTPUT,
            "output_cost_per_audio_token": NATIVE_AUDIO_AUDIO_OUTPUT,
        }
        for key in (*NATIVE_AUDIO_KEYS, *LIVE_NATIVE_AUDIO_KEYS)
    },
}
ALL_KEYS: Final = tuple(PUBLISHED_RATES)
NATIVE_AUDIO_BILLING_CASES: Final = (
    *((key, "gemini") for key in NATIVE_AUDIO_KEYS),
    ("gemini-live-2.5-flash-preview-native-audio-09-2025", "vertex_ai"),
    ("gemini/gemini-live-2.5-flash-preview-native-audio-09-2025", "gemini"),
)
LONG_CONTEXT_TIER_FIELDS: Final = (
    "input_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens",
)


def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", ALL_KEYS)
def test_backup_matches_main(model: str):
    assert _load(BACKUP_PATH)[model] == _load(MAIN_PATH)[model]

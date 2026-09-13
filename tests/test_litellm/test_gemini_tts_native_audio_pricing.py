import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import CompletionTokensDetailsWrapper, PromptTokensDetailsWrapper, Usage

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
@pytest.mark.parametrize("path", (MAIN_PATH, BACKUP_PATH), ids=("main", "backup"))
def test_published_rates_are_registered(model: str, path: Path):
    info = _load(path)[model]
    for field, value in PUBLISHED_RATES[model].items():
        assert info[field] == value, f"{model} {field} in {path.name}: {info.get(field)} != {value}"


@pytest.mark.parametrize("model", PRO_TTS_KEYS)
@pytest.mark.parametrize("path", (MAIN_PATH, BACKUP_PATH), ids=("main", "backup"))
def test_pro_tts_has_no_long_context_tier(model: str, path: Path):
    info = _load(path)[model]
    for field in LONG_CONTEXT_TIER_FIELDS:
        assert field not in info, f"{model} has {field} but Google publishes one flat TTS rate"


@pytest.mark.parametrize("model", ALL_KEYS)
def test_backup_matches_main(model: str):
    assert _load(BACKUP_PATH)[model] == _load(MAIN_PATH)[model]


@pytest.mark.parametrize(
    ("model", "provider", "input_rate", "audio_output_rate"),
    (
        ("gemini-2.5-flash-preview-tts", "gemini", FLASH_TTS_INPUT, FLASH_TTS_AUDIO_OUTPUT),
        ("gemini-2.5-pro-preview-tts", "gemini", PRO_TTS_INPUT, PRO_TTS_AUDIO_OUTPUT),
        ("gemini-2.5-pro-preview-tts", "vertex_ai", PRO_TTS_INPUT, PRO_TTS_AUDIO_OUTPUT),
    ),
)
def test_tts_audio_output_is_billed_at_the_audio_rate(
    model: str, provider: str, input_rate: float, audio_output_rate: float, local_model_cost_map
):
    usage: Final = Usage(
        prompt_tokens=9,
        completion_tokens=49,
        total_tokens=58,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=9),
        completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=49, text_tokens=0),
    )
    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider=provider)
    assert prompt_cost == pytest.approx(9 * input_rate)
    assert completion_cost == pytest.approx(49 * audio_output_rate)


@pytest.mark.parametrize("model, provider", NATIVE_AUDIO_BILLING_CASES)
def test_native_audio_output_is_billed_at_the_audio_rate(model: str, provider: str, local_model_cost_map):
    usage: Final = Usage(
        prompt_tokens=377,
        completion_tokens=84,
        total_tokens=461,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=377),
        completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=48, reasoning_tokens=36, text_tokens=0),
    )
    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider=provider)
    assert prompt_cost == pytest.approx(377 * NATIVE_AUDIO_TEXT_INPUT)
    assert completion_cost == pytest.approx(48 * NATIVE_AUDIO_AUDIO_OUTPUT + 36 * NATIVE_AUDIO_TEXT_OUTPUT)


@pytest.mark.parametrize("model, provider", NATIVE_AUDIO_BILLING_CASES)
def test_native_audio_input_is_billed_at_the_audio_rate(model: str, provider: str, local_model_cost_map):
    usage: Final = Usage(
        prompt_tokens=1000,
        completion_tokens=0,
        total_tokens=1000,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=100, audio_tokens=900),
    )
    prompt_cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider=provider)
    assert prompt_cost == pytest.approx(100 * NATIVE_AUDIO_TEXT_INPUT + 900 * NATIVE_AUDIO_AUDIO_INPUT)

"""LLM_TRANSLATION_V2 is the single opt-in; pin its exact parse semantics."""

import pytest

from litellm.translation import TRANSLATION_V2_ENV, is_translation_v2_enabled


def test_unset_means_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRANSLATION_V2_ENV, raising=False)
    assert is_translation_v2_enabled() is False


@pytest.mark.parametrize("value", ["True", "true", "TRUE", "1", " true "])
def test_truthy_values_turn_v2_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(TRANSLATION_V2_ENV, value)
    assert is_translation_v2_enabled() is True


@pytest.mark.parametrize("value", ["", "False", "false", "0", "no", "anthropic"])
def test_everything_else_stays_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(TRANSLATION_V2_ENV, value)
    assert is_translation_v2_enabled() is False

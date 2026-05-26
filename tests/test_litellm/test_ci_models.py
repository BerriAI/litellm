"""Tests for ``tests/ci_models.py`` -- the CI model name registry."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _clean_ci_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ``CI_MODEL_*`` env var before each test.

    Without this, any ``CI_MODEL_*`` set in the surrounding CI environment
    (which is exactly the environment this module is meant to support) can
    poison the default-value assertions in tests that don't use the
    ``fresh_ci_models`` fixture.
    """
    for key in [k for k in os.environ if k.startswith("CI_MODEL_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fresh_ci_models() -> Iterator[object]:
    """Return a freshly-imported ``tests.ci_models`` module.

    The module resolves its constants at import time, so any test that
    wants to observe an env-var override must re-import the module after
    setting the env var. ``_clean_ci_model_env`` (autouse, module-wide)
    ensures the outer environment doesn't leak into the import.
    """
    sys.modules.pop("tests.ci_models", None)
    module = importlib.import_module("tests.ci_models")
    yield module
    sys.modules.pop("tests.ci_models", None)


class TestDefaults:
    def test_known_constants_are_exposed(self, fresh_ci_models):
        # Spot-check a representative sample of the constants we expose.
        for name in (
            "GPT_3_5_TURBO",
            "GPT_4",
            "GPT_4O",
            "GPT_4O_MINI",
            "GPT_5_MINI",
            "CLAUDE_SONNET_4_5",
            "CLAUDE_SONNET_4",
            "BEDROCK_CLAUDE_HAIKU_4_5",
            "GEMINI_2_5_FLASH",
            "TEXT_EMBEDDING_ADA_002",
        ):
            assert hasattr(fresh_ci_models, name), f"missing constant {name!r}"
            assert isinstance(getattr(fresh_ci_models, name), str)
            assert getattr(fresh_ci_models, name), f"{name} resolved to empty string"

    def test_defaults_match_well_known_identifiers(self, fresh_ci_models):
        """Defaults should not silently drift; pin a few critical ones."""
        assert fresh_ci_models.GPT_3_5_TURBO == "gpt-3.5-turbo"
        assert fresh_ci_models.GPT_4 == "gpt-4"
        assert fresh_ci_models.GPT_4O == "gpt-4o"
        assert fresh_ci_models.GPT_4O_MINI == "gpt-4o-mini"
        assert fresh_ci_models.CLAUDE_SONNET_4_5 == "claude-sonnet-4-5-20250929"
        assert (
            fresh_ci_models.BEDROCK_CLAUDE_HAIKU_4_5
            == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )


class TestEnvOverride:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CI_MODEL_GPT_4O", "gpt-4o-2024-08-06")
        sys.modules.pop("tests.ci_models", None)
        ci_models = importlib.import_module("tests.ci_models")
        try:
            assert ci_models.GPT_4O == "gpt-4o-2024-08-06"
            # Unrelated constants are untouched (autouse fixture above
            # guarantees no stray CI_MODEL_GPT_4 from the outer env).
            assert ci_models.GPT_4 == "gpt-4"
        finally:
            sys.modules.pop("tests.ci_models", None)

    def test_empty_env_var_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CI_MODEL_GPT_4O", "")
        sys.modules.pop("tests.ci_models", None)
        ci_models = importlib.import_module("tests.ci_models")
        try:
            assert ci_models.GPT_4O == "gpt-4o"
        finally:
            sys.modules.pop("tests.ci_models", None)

    def test_post_import_env_change_does_not_affect_constants(
        self, fresh_ci_models, monkeypatch: pytest.MonkeyPatch
    ):
        # Setting the env var AFTER import must not mutate already-resolved
        # constants -- this is documented behaviour.
        assert fresh_ci_models.GPT_4 == "gpt-4"
        monkeypatch.setenv("CI_MODEL_GPT_4", "should-not-apply")
        assert fresh_ci_models.GPT_4 == "gpt-4"


class TestGetCiModel:
    def test_returns_env_value_when_set(
        self, monkeypatch: pytest.MonkeyPatch, fresh_ci_models
    ):
        monkeypatch.setenv("CI_MODEL_GPT_4O", "override-value")
        assert (
            fresh_ci_models.get_ci_model("CI_MODEL_GPT_4O", "fallback")
            == "override-value"
        )

    def test_returns_default_when_unset(self, fresh_ci_models):
        assert (
            fresh_ci_models.get_ci_model("CI_MODEL_NEVER_SET", "fallback")
            == "fallback"
        )

    def test_returns_default_when_empty(
        self, monkeypatch: pytest.MonkeyPatch, fresh_ci_models
    ):
        monkeypatch.setenv("CI_MODEL_EMPTY", "")
        assert (
            fresh_ci_models.get_ci_model("CI_MODEL_EMPTY", "fallback")
            == "fallback"
        )

    def test_reads_env_on_each_call(
        self, monkeypatch: pytest.MonkeyPatch, fresh_ci_models
    ):
        # Unlike module-level constants, get_ci_model picks up changes to
        # os.environ after import.
        monkeypatch.setenv("CI_MODEL_DYNAMIC", "first")
        assert fresh_ci_models.get_ci_model("CI_MODEL_DYNAMIC", "x") == "first"
        monkeypatch.setenv("CI_MODEL_DYNAMIC", "second")
        assert fresh_ci_models.get_ci_model("CI_MODEL_DYNAMIC", "x") == "second"

    def test_rejects_env_var_missing_prefix(self, fresh_ci_models):
        with pytest.raises(ValueError, match="CI_MODEL_"):
            fresh_ci_models.get_ci_model("OTHER_VAR", "fallback")

    def test_rejects_env_var_with_wrong_prefix(self, fresh_ci_models):
        with pytest.raises(ValueError, match="CI_MODEL_"):
            fresh_ci_models.get_ci_model("MY_MODEL", "fallback")

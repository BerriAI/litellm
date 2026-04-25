"""
Regression tests for is_model_gpt_5_model() in both OpenAI and Azure GPT-5 config
classes.

Background
----------
In v1.82.3 a substring check was introduced::

    return "gpt-5" in model and "gpt-5-chat" not in model

This inadvertently treated versioned chat models like ``gpt-5.3-chat`` and
``gpt-5.1-chat`` as *non*-GPT-5 models, because the string ``"gpt-5-chat"`` is
a substring of ``"gpt-5.3-chat"``.  Those models were then routed through the
regular Azure chat path which does not suppress ``parallel_tool_calls``, causing
Azure to return ``finish_reason="stop"`` together with tool_calls and breaking
n8n AI-agent workflows.

There are two distinct families:

* **gpt-5-chat family** (``gpt-5-chat``, ``gpt-5-chat-latest``,
  ``gpt-5-chat-2025-08-07``, …) — regular chat models that support ``temperature``
  and ``tool_choice`` but NOT ``reasoning_effort``.  Must NOT be on the GPT-5
  reasoning path.

* **Versioned chat models** (``gpt-5.1-chat``, ``gpt-5.2-chat``,
  ``gpt-5.3-chat``, …) — ARE GPT-5 reasoning models and must stay on the GPT-5
  path.

The fix uses a prefix check (``startswith("gpt-5-chat")``) on the normalised model
name instead of a substring check, which correctly distinguishes the two families.
"""

import pytest

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.azure.chat.gpt_5_transformation import AzureOpenAIGPT5Config

# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------

# Models that MUST be classified as GPT-5 (routed through GPT-5 reasoning path)
GPT5_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.3",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.5-2026-04-23",  # dated variant
    "gpt-5.5-pro-2026-04-23",  # dated variant
    "gpt-5.1-chat",  # versioned chat — THE KEY REGRESSION CASE
    "gpt-5.2-chat",  # versioned chat — also a regression case
    "gpt-5.3-chat",  # versioned chat — THE KEY REGRESSION CASE
    "gpt-5.2-chat-latest",  # versioned chat with date suffix
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5-codex",
]

# Models that must NOT be classified as GPT-5 (regular chat path)
NON_GPT5_MODELS = [
    "gpt-5-chat",  # gpt-5-chat family — regular chat path
    "gpt-5-chat-latest",  # gpt-5-chat family with alias suffix
    "gpt-5-chat-2025-08-07",  # gpt-5-chat family with date suffix
    "gpt-4",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o3",
    "o3-mini",
]


# ---------------------------------------------------------------------------
# OpenAIGPT5Config
# ---------------------------------------------------------------------------


class TestOpenAIGPT5ConfigIsModelGpt5Model:

    @pytest.mark.parametrize("model", GPT5_MODELS)
    def test_gpt5_models_are_classified_as_gpt5(self, model: str):
        assert OpenAIGPT5Config.is_model_gpt_5_model(
            model
        ), f"Expected '{model}' to be classified as a GPT-5 model"

    @pytest.mark.parametrize("model", NON_GPT5_MODELS)
    def test_non_gpt5_models_are_not_classified_as_gpt5(self, model: str):
        assert not OpenAIGPT5Config.is_model_gpt_5_model(
            model
        ), f"Expected '{model}' NOT to be classified as a GPT-5 model"

    def test_versioned_chat_models_are_not_excluded_by_prefix(self):
        """Core regression guard: gpt-5-chat prefix must not match versioned models."""
        versioned_chat_models = ["gpt-5.1-chat", "gpt-5.2-chat", "gpt-5.3-chat"]
        for model in versioned_chat_models:
            assert OpenAIGPT5Config.is_model_gpt_5_model(
                model
            ), f"Regression: '{model}' was incorrectly excluded from GPT-5 path"

    def test_gpt5_chat_family_is_excluded(self):
        """gpt-5-chat family should stay on the regular chat path."""
        for model in ["gpt-5-chat", "gpt-5-chat-latest", "gpt-5-chat-2025-08-07"]:
            assert not OpenAIGPT5Config.is_model_gpt_5_model(
                model
            ), f"Expected '{model}' (gpt-5-chat family) NOT to be on the GPT-5 path"


# ---------------------------------------------------------------------------
# AzureOpenAIGPT5Config
# ---------------------------------------------------------------------------


class TestAzureOpenAIGPT5ConfigIsModelGpt5Model:

    @pytest.mark.parametrize("model", GPT5_MODELS)
    def test_gpt5_models_are_classified_as_gpt5(self, model: str):
        assert AzureOpenAIGPT5Config.is_model_gpt_5_model(
            model
        ), f"Expected Azure '{model}' to be classified as a GPT-5 model"

    @pytest.mark.parametrize("model", NON_GPT5_MODELS)
    def test_non_gpt5_models_are_not_classified_as_gpt5(self, model: str):
        assert not AzureOpenAIGPT5Config.is_model_gpt_5_model(
            model
        ), f"Expected Azure '{model}' NOT to be classified as a GPT-5 model"

    def test_versioned_chat_models_are_not_excluded_by_prefix(self):
        """Core regression guard: gpt-5-chat prefix must not match versioned models."""
        versioned_chat_models = ["gpt-5.1-chat", "gpt-5.2-chat", "gpt-5.3-chat"]
        for model in versioned_chat_models:
            assert AzureOpenAIGPT5Config.is_model_gpt_5_model(
                model
            ), f"Regression: Azure '{model}' was incorrectly excluded from GPT-5 path"

    def test_gpt5_chat_family_is_excluded(self):
        """gpt-5-chat family should stay on the regular chat path."""
        for model in ["gpt-5-chat", "gpt-5-chat-latest", "gpt-5-chat-2025-08-07"]:
            assert not AzureOpenAIGPT5Config.is_model_gpt_5_model(
                model
            ), f"Expected Azure '{model}' (gpt-5-chat family) NOT to be on the GPT-5 path"

    def test_gpt5_series_routing_prefix_is_always_classified_as_gpt5(self):
        """Models using the gpt5_series/ manual-routing prefix must always match."""
        series_models = ["gpt5_series/my-deployment", "gpt5_series/prod"]
        for model in series_models:
            assert AzureOpenAIGPT5Config.is_model_gpt_5_model(
                model
            ), f"Azure '{model}' with gpt5_series/ prefix should be classified as GPT-5"

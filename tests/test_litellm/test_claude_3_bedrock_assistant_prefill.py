"""
Regression test: the AWS Bedrock Anthropic Claude 3 family entries in the
model-cost maps were missing ``supports_assistant_prefill`` entirely. Because
the key was absent rather than ``true``, callers that gate on its truthiness
silently treated these models as not supporting trailing-assistant prefill,
even though every Claude 3 model supports it (the direct ``anthropic`` API
entries for ``claude-3-haiku``/``claude-3-opus`` already carry ``true``).

These tests pin ``supports_assistant_prefill == true`` for the affected entries
in both the primary price map and the ``litellm/`` backup, and verify
``get_model_info`` surfaces it, so the field cannot silently drop back to absent.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

import litellm

CLAUDE_3_BEDROCK_MODELS = (
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "eu.anthropic.claude-3-haiku-20240307-v1:0",
    "eu.anthropic.claude-3-sonnet-20240229-v1:0",
    "eu.anthropic.claude-3-opus-20240229-v1:0",
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "apac.anthropic.claude-3-haiku-20240307-v1:0",
    "apac.anthropic.claude-3-sonnet-20240229-v1:0",
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/us-gov-east-1/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/us-gov-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/us-gov-west-1/anthropic.claude-3-5-sonnet-20240620-v1:0",
)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _backup_path() -> str:
    return os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )


def _main_path() -> str:
    # This test lives at ``tests/test_litellm/``; the primary price map sits at
    # the repo root, two directories up. Resolve it relative to this file so the
    # test works regardless of where ``litellm`` itself is installed.
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model_prices_and_context_window.json",
    )


class TestClaude3BedrockAssistantPrefillData:
    """Both JSON price maps must mark every Claude 3 Bedrock entry as supporting
    assistant prefill."""

    def test_backup_supports_assistant_prefill(self):
        price_map = _load_json(_backup_path())
        for model in CLAUDE_3_BEDROCK_MODELS:
            assert price_map[model]["supports_assistant_prefill"] is True, model

    def test_main_supports_assistant_prefill(self):
        price_map = _load_json(_main_path())
        for model in CLAUDE_3_BEDROCK_MODELS:
            assert price_map[model]["supports_assistant_prefill"] is True, model


class TestClaude3BedrockAssistantPrefillModelInfo:
    """``get_model_info`` must report ``supports_assistant_prefill`` for these
    models."""

    def test_get_model_info_supports_assistant_prefill(self):
        # Patch litellm.model_cost with the local backup so the assertion does
        # not depend on the remote fetch hitting a not-yet-merged main branch.
        original = litellm.model_cost
        try:
            litellm.model_cost = _load_json(_backup_path())
            for model in CLAUDE_3_BEDROCK_MODELS:
                info = litellm.get_model_info(model)
                assert info["supports_assistant_prefill"] is True, model
        finally:
            litellm.model_cost = original

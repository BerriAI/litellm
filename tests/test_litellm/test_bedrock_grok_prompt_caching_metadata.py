"""Regression tests: Bedrock-hosted xAI Grok must NOT advertise prompt caching.

AWS Bedrock does not support prompt caching for Grok. When the cost-map marks
``us.xai.grok-4.6`` / ``global.xai.grok-4.6`` (``litellm_provider:
bedrock_converse``) with ``supports_prompt_caching: true``,
``AnthropicCacheControlHook`` auto-injects ``cache_control`` breakpoints and
Bedrock rejects the whole request:

    "You invoked an unsupported model or your request did not allow prompt
    caching."

So the flag must be ``false`` and the rows must carry no cache-read price. These
data-level checks read the JSON files directly (rather than ``litellm.model_cost``,
which may be refreshed from the remote map at import time) so they pin the shipped
map regardless of network state.
"""

import json
import os

import litellm

BEDROCK_GROK_MODELS = ["us.xai.grok-4.6", "global.xai.grok-4.6"]


def _load_backup_json() -> dict:
    backup_path = os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )
    with open(backup_path, encoding="utf-8") as f:
        return json.load(f)


def _load_root_json() -> dict:
    root_path = os.path.join(
        os.path.dirname(os.path.dirname(litellm.__file__)),
        "model_prices_and_context_window.json",
    )
    with open(root_path, encoding="utf-8") as f:
        return json.load(f)


class TestBedrockGrokPromptCaching:
    def test_backup_marks_bedrock_grok_no_prompt_caching(self):
        data = _load_backup_json()
        for model in BEDROCK_GROK_MODELS:
            entry = data.get(model, {})
            assert entry.get("litellm_provider") == "bedrock_converse", model
            assert entry.get("supports_prompt_caching") is False, model

    def test_backup_has_no_cache_read_price_for_bedrock_grok(self):
        data = _load_backup_json()
        for model in BEDROCK_GROK_MODELS:
            entry = data.get(model, {})
            assert "cache_read_input_token_cost" not in entry, model

    def test_root_and_backup_agree_for_bedrock_grok(self):
        root = _load_root_json()
        backup = _load_backup_json()
        for model in BEDROCK_GROK_MODELS:
            assert root.get(model) == backup.get(model), model

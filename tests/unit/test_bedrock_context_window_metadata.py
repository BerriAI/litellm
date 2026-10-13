"""
Regression tests for #46216 – the two bare Bedrock Claude 3.5 Sonnet ids
listed a 1,000,000-token context window and carried the Sonnet-4
long-context price tier (``*_above_200k_tokens``), while every regional
sibling of the same models correctly listed 200000 with no long-context
tier. Anything that reads ``max_input_tokens`` (``context_window_fallbacks``,
pre-call context checks, context-usage reporting) treated bare-id
Claude 3.5 Sonnet as having 5x its real window.

This was corrected before in #35635, then lost when that PR was closed in
favor of the registry audit #37902, which was declined. These tests pin the
correction in both the primary and the backup cost map so it cannot silently
regress again.

The assertions are sibling-agreement invariants (a bare id and its regional
copies describe the same underlying model, so their context windows must
agree) plus one cited vendor fact where the value itself is load-bearing.
"""

import json
import os

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_map(filename: str) -> dict:
    with open(os.path.join(_REPO_ROOT, filename), encoding="utf-8") as f:
        return json.load(f)


def _maps():
    return (
        _load_map("model_prices_and_context_window.json"),
        _load_map(os.path.join("litellm", "model_prices_and_context_window_backup.json")),
    )


CLAUDE_3_5_SONNET_BARE_IDS = (
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
)


class TestBedrockClaude35SonnetContextWindow:
    """Bare Bedrock Claude 3.5 Sonnet must match its regional siblings."""

    def test_bare_ids_match_their_regional_siblings(self):
        for data in _maps():
            for key in CLAUDE_3_5_SONNET_BARE_IDS:
                bare = data[key]["max_input_tokens"]
                for regional_prefix in ("us.", "eu."):
                    regional = data[regional_prefix + key]["max_input_tokens"]
                    assert bare == regional, regional_prefix + key

    def test_claude_3_5_sonnet_window_is_200k(self):
        # Claude 3.5 Sonnet's documented context window is 200K tokens
        # (Anthropic model overview and AWS Bedrock model page for
        # anthropic.claude-3-5-sonnet-*, accessed 2026-10-13).
        for data in _maps():
            for key in CLAUDE_3_5_SONNET_BARE_IDS:
                assert data[key]["max_input_tokens"] == 200000, key

    def test_no_long_context_tier_below_the_tier_threshold(self):
        # A ``*_above_200k_tokens`` price tier on an entry whose window is
        # at most 200K tokens can never apply - it is a copy-paste from a
        # long-context model (Sonnet 4 in this case).
        for data in _maps():
            for key in CLAUDE_3_5_SONNET_BARE_IDS:
                assert data[key]["max_input_tokens"] <= 200000, key
                tier_fields = [f for f in data[key] if "above_200k" in f]
                assert tier_fields == [], f"{key} still carries {tier_fields}"


class TestBedrockSiblingContextWindowAlignment:
    """Bare and regional ids of the same model must not disagree (#46216)."""

    def test_deepseek_v3_2_bare_matches_all_bedrock_region_copies(self):
        for data in _maps():
            bare = data["deepseek.v3.2"]["max_input_tokens"]
            regional = data["bedrock/us-east-1/deepseek.v3.2"]["max_input_tokens"]
            assert bare == regional

    def test_kimi_k2_5_bare_matches_bedrock_region_copies(self):
        for data in _maps():
            bare = data["moonshotai.kimi-k2.5"]["max_input_tokens"]
            regional = data["bedrock/ap-northeast-1/moonshotai.kimi-k2.5"]["max_input_tokens"]
            assert bare == regional

    def test_qwen3_bare_ids_match_bedrock_region_copies(self):
        for data in _maps():
            for key in ("qwen.qwen3-coder-next", "qwen.qwen3-next-80b-a3b"):
                bare = data[key]["max_input_tokens"]
                regional = data[f"bedrock/ap-northeast-1/{key}"]["max_input_tokens"]
                assert bare == regional, key

    def test_us_gov_llama3_copies_match_the_bare_entry(self):
        # The us-gov copies were the only ones disagreeing with the bare
        # entry (8000 vs 8192).
        for data in _maps():
            for key in (
                "meta.llama3-8b-instruct-v1:0",
                "meta.llama3-70b-instruct-v1:0",
            ):
                bare = data[key]["max_input_tokens"]
                for region in ("us-gov-east-1", "us-gov-west-1"):
                    regional = data[f"bedrock/{region}/{key}"]["max_input_tokens"]
                    assert regional == bare, f"bedrock/{region}/{key}"

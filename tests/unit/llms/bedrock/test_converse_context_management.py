"""Bedrock Converse context_management forwarding (compact_20260112, and clear_thinking_20251015 while thinking is on)."""

import pytest

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

CLAUDE_MODEL = "anthropic.claude-opus-4-7-20250115-v1:0"


def test_supported_params_include_context_management_for_anthropic():
    cfg = AmazonConverseConfig()
    params = cfg.get_supported_openai_params(CLAUDE_MODEL)
    assert "context_management" in params


def test_supported_params_exclude_context_management_for_non_anthropic():
    cfg = AmazonConverseConfig()
    params = cfg.get_supported_openai_params("meta.llama3-70b-instruct-v1:0")
    assert "context_management" not in params


def test_map_openai_params_forwards_anthropic_shape():
    cfg = AmazonConverseConfig()
    optional_params: dict = {}
    cfg.map_openai_params(
        non_default_params={
            "context_management": {"edits": [{"type": "compact_20260112"}]}
        },
        optional_params=optional_params,
        model=CLAUDE_MODEL,
        drop_params=False,
    )
    assert optional_params.get("context_management") == {
        "edits": [{"type": "compact_20260112"}]
    }


def test_map_openai_params_normalizes_openai_list_shape():
    """OpenAI Responses-API style list of {type: "compaction"} normalizes to Anthropic dict."""
    cfg = AmazonConverseConfig()
    optional_params: dict = {}
    cfg.map_openai_params(
        non_default_params={"context_management": [{"type": "compaction"}]},
        optional_params=optional_params,
        model=CLAUDE_MODEL,
        drop_params=False,
    )
    forwarded = optional_params.get("context_management")
    assert isinstance(forwarded, dict)
    edits = forwarded.get("edits")
    assert isinstance(edits, list) and len(edits) == 1
    assert edits[0].get("type") == "compact_20260112"


def test_filter_keeps_only_compact_edits_and_adds_beta_header():
    additional = {
        "context_management": {
            "edits": [
                {"type": "clear_tool_uses_20250919"},
                {"type": "compact_20260112"},
                {"type": "clear_thinking_20251015"},
            ]
        }
    }
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert additional["context_management"]["edits"] == [{"type": "compact_20260112"}]
    assert "compact-2026-01-12" in betas


def test_filter_drops_field_when_no_compact_edit_remains():
    additional = {
        "context_management": {
            "edits": [
                {"type": "clear_tool_uses_20250919"},
                {"type": "clear_thinking_20251015"},
            ]
        }
    }
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert "context_management" not in additional
    assert betas == []


def test_filter_is_noop_when_field_absent():
    additional: dict = {}
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert additional == {}
    assert betas == []


def test_filter_drops_malformed_edits_list():
    additional = {"context_management": {"edits": "not a list"}}
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert "context_management" not in additional
    assert betas == []


def test_filter_does_not_duplicate_beta_header():
    additional = {"context_management": {"edits": [{"type": "compact_20260112"}]}}
    betas: list = ["compact-2026-01-12"]
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert betas.count("compact-2026-01-12") == 1


CLEAR_THINKING_EDIT = {
    "type": "clear_thinking_20251015",
    "keep": {"type": "thinking_turns", "value": 1},
}


@pytest.mark.parametrize(
    "thinking", [{"type": "enabled", "budget_tokens": 2048}, {"type": "adaptive"}]
)
def test_filter_keeps_clear_thinking_edit_and_adds_beta_when_thinking_is_on(thinking):
    additional = {
        "thinking": thinking,
        "context_management": {"edits": [CLEAR_THINKING_EDIT]},
    }
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert additional["context_management"] == {"edits": [CLEAR_THINKING_EDIT]}
    assert betas == ["context-management-2025-06-27"]


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_filter_drops_clear_thinking_edit_when_thinking_is_off(thinking):
    additional = {"context_management": {"edits": [CLEAR_THINKING_EDIT]}}
    if thinking is not None:
        additional["thinking"] = thinking
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert "context_management" not in additional
    assert betas == []


def test_filter_keeps_compact_and_clear_thinking_together_with_both_betas():
    additional = {
        "thinking": {"type": "adaptive"},
        "context_management": {
            "edits": [
                CLEAR_THINKING_EDIT,
                {"type": "compact_20260112"},
                {"type": "clear_tool_uses_20250919"},
            ]
        },
    }
    betas: list = []
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert additional["context_management"]["edits"] == [
        CLEAR_THINKING_EDIT,
        {"type": "compact_20260112"},
    ]
    assert sorted(betas) == ["compact-2026-01-12", "context-management-2025-06-27"]


def test_filter_does_not_duplicate_context_management_beta():
    additional = {
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": [CLEAR_THINKING_EDIT]},
    }
    betas: list = ["context-management-2025-06-27"]
    AmazonConverseConfig._filter_context_management_for_bedrock_converse(
        additional, betas
    )
    assert betas == ["context-management-2025-06-27"]

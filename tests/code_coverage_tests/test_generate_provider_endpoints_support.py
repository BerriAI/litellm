import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from scripts.generate_provider_endpoints_support import (
    generate_provider_endpoints_support,
    main,
)

ENDPOINTS: Final = {
    "a2a": {"provider_json_field": "a2a", "bridges_to_chat_completion": True},
    "messages": {"provider_json_field": "messages", "bridges_to_chat_completion": True},
    "responses": {"provider_json_field": "responses", "bridges_to_chat_completion": True},
    "generateContent": {"provider_json_field": "generateContent", "bridges_to_chat_completion": True},
    "text_completion": {"provider_json_field": "text_completion", "bridges_to_chat_completion": True},
    "count_tokens": {"provider_json_field": "count_tokens"},
}


def _current_json(providers: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "_comment": "retained",
        "_schema": {"version": 1},
        "providers": dict(providers),
        "endpoints": ENDPOINTS,
    }


def _provider(flags: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    return {
        "display_name": "Example (`example`)",
        "url": "https://docs.litellm.ai/docs/providers/example",
        "endpoints": dict(flags or {}),
    }


def _flags(generated: Mapping[str, JsonValue], slug: str) -> Mapping[str, JsonValue]:
    providers: Final = generated["providers"]
    assert isinstance(providers, dict)
    provider: Final = providers[slug]
    assert isinstance(provider, dict)
    flags: Final = provider["endpoints"]
    assert isinstance(flags, dict)
    return flags


def test_mode_derives_chat_flag() -> None:
    cost_map: Final = {"example/model": {"litellm_provider": "example", "mode": "chat"}}
    current_json: Final = _current_json({"example": _provider()})

    generated: Final = generate_provider_endpoints_support(cost_map, current_json)

    assert _flags(generated, "example")["chat_completions"] is True


def test_supported_endpoint_derives_embedding_flag() -> None:
    cost_map: Final = {
        "example/model": {
            "litellm_provider": "example",
            "mode": "completion",
            "supported_endpoints": ["/v1/embeddings"],
        }
    }
    current_json: Final = _current_json({"example": _provider()})

    generated: Final = generate_provider_endpoints_support(cost_map, current_json)

    assert _flags(generated, "example")["embeddings"] is True


def test_chat_models_enable_bridge_flags_but_other_models_do_not() -> None:
    cost_map: Final = {
        "example/chat": {"litellm_provider": "has_chat", "mode": "chat"},
        "example/endpoint-chat": {
            "litellm_provider": "endpoint_chat",
            "mode": "image_generation",
            "supported_endpoints": ["/v1/chat/completions"],
        },
        "example/image": {"litellm_provider": "no_chat", "mode": "image_generation"},
    }
    current_json: Final = _current_json(
        {
            "has_chat": _provider(),
            "endpoint_chat": _provider(),
            "no_chat": _provider(),
        }
    )

    generated: Final = generate_provider_endpoints_support(cost_map, current_json)

    assert _flags(generated, "has_chat")["a2a"] is True
    assert _flags(generated, "has_chat")["generateContent"] is True
    assert _flags(generated, "endpoint_chat")["a2a"] is True
    assert _flags(generated, "no_chat")["a2a"] is False
    assert _flags(generated, "no_chat")["generateContent"] is False


def test_unbacked_derived_flags_reset_and_handwritten_flags_remain() -> None:
    cost_map: Final = {"example/model": {"litellm_provider": "example", "mode": "image_generation"}}
    current_json: Final = _current_json(
        {
            "example": _provider({"responses": True, "count_tokens": True, "files": False}),
            "without_models": _provider({"responses": True}),
        }
    )

    generated: Final = generate_provider_endpoints_support(cost_map, current_json)
    flags: Final = _flags(generated, "example")

    assert flags["responses"] is False
    assert flags["count_tokens"] is True
    assert flags["files"] is False
    assert _flags(generated, "without_models")["chat_completions"] is False


def test_provider_aliases_share_their_mapped_slugs() -> None:
    cost_map: Final = {
        "vertex/model-a": {"litellm_provider": "vertex_ai-chat", "mode": "chat"},
        "vertex/model-b": {"litellm_provider": "vertex_ai-embedding", "mode": "embedding"},
        "bedrock/model": {"litellm_provider": "bedrock_converse", "mode": "responses"},
    }
    current_json: Final = _current_json({"vertex_ai": _provider(), "bedrock": _provider()})

    generated: Final = generate_provider_endpoints_support(cost_map, current_json)

    assert _flags(generated, "vertex_ai")["chat_completions"] is True
    assert _flags(generated, "vertex_ai")["embeddings"] is True
    assert _flags(generated, "bedrock")["responses"] is True


def test_cost_map_provider_without_file_entry_is_an_error() -> None:
    cost_map: Final = {"example/model": {"litellm_provider": "missing_provider", "mode": "chat"}}
    current_json: Final = _current_json({"example": _provider()})

    with pytest.raises(ValueError, match="missing_provider"):
        generate_provider_endpoints_support(cost_map, current_json)


def test_check_reports_stale_output_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cost_map: Final = {"example/model": {"litellm_provider": "example", "mode": "chat"}}
    current_json: Final = _current_json({"example": _provider()})
    (tmp_path / "model_prices_and_context_window.json").write_text(json.dumps(cost_map), encoding="utf-8")
    source_path: Final = tmp_path / "provider_endpoints_support.json"
    source_content: Final = json.dumps(current_json)
    source_path.write_text(source_content, encoding="utf-8")

    result: Final = main(["--check"], repo_root=tmp_path)

    assert result == 1
    assert source_path.read_text(encoding="utf-8") == source_content
    assert not (tmp_path / "litellm/provider_endpoints_support_backup.json").exists()
    error_output: Final = capsys.readouterr().err
    assert "provider endpoint support is stale" in error_output
    assert "scripts/generate_provider_endpoints_support.py" in error_output


def test_check_reports_stale_backup_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cost_map: Final = {"example/model": {"litellm_provider": "example", "mode": "chat"}}
    current_json: Final = _current_json({"example": _provider()})
    generated: Final = generate_provider_endpoints_support(cost_map, current_json)
    generated_content: Final = json.dumps(generated, indent=2, ensure_ascii=False) + "\n"
    source_path: Final = tmp_path / "provider_endpoints_support.json"
    backup_path: Final = tmp_path / "litellm/provider_endpoints_support_backup.json"
    backup_content: Final = "stale\n"
    (tmp_path / "model_prices_and_context_window.json").write_text(json.dumps(cost_map), encoding="utf-8")
    source_path.write_text(generated_content, encoding="utf-8")
    backup_path.parent.mkdir()
    backup_path.write_text(backup_content, encoding="utf-8")

    result: Final = main(["--check"], repo_root=tmp_path)

    assert result == 1
    assert source_path.read_text(encoding="utf-8") == generated_content
    assert backup_path.read_text(encoding="utf-8") == backup_content
    assert str(backup_path) in capsys.readouterr().err

from pathlib import Path

import pytest
from fastapi import HTTPException

from litellm.proxy.relay_endpoints.endpoints import _load_managed_config


def test_load_managed_config_parses_pinned_version(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text(
        "\n".join(
            [
                "claude_code:",
                "  channel: pinned",
                "  version: 1.2.3",
                "  registry: npm",
                "  model: claude-sonnet-4-5",
                "policy_version: 7",
            ]
        )
    )

    config = _load_managed_config(path)

    assert config.claude_code.version == "1.2.3"
    assert config.claude_code.channel == "pinned"
    assert config.claude_code.registry == "npm"
    assert config.claude_code.model == "claude-sonnet-4-5"
    assert config.policy_version == 7


def test_load_managed_config_missing_file_returns_empty_policy(tmp_path: Path) -> None:
    config = _load_managed_config(tmp_path / "does_not_exist.yaml")

    assert config.claude_code.version is None
    assert config.policy_version is None


def test_load_managed_config_empty_file_returns_empty_policy(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text("")

    config = _load_managed_config(path)

    assert config.claude_code.version is None


def test_load_managed_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text("- just\n- a\n- list\n")

    with pytest.raises(HTTPException) as exc_info:
        _load_managed_config(path)

    assert exc_info.value.status_code == 500


def test_load_managed_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text("claude_code: {version: '1.2.3'\n")

    with pytest.raises(HTTPException) as exc_info:
        _load_managed_config(path)

    assert exc_info.value.status_code == 500


def test_load_managed_config_preserves_nested_managed_settings(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text(
        "\n".join(
            [
                "claude_code:",
                "  version: 2.1.208",
                "  managed_settings:",
                "    env:",
                "      ANTHROPIC_MODEL: claude-sonnet-4-5",
                "    permissions:",
                "      defaultMode: acceptEdits",
            ]
        )
    )

    config = _load_managed_config(path)

    assert config.claude_code.managed_settings["env"] == {
        "ANTHROPIC_MODEL": "claude-sonnet-4-5"
    }
    assert config.claude_code.managed_settings["permissions"] == {
        "defaultMode": "acceptEdits"
    }

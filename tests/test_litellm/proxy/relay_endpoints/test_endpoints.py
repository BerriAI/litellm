from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.relay_endpoints.endpoints import (
    RELAY_SETTINGS_PATH_ENV,
    _load_managed_config,
    router,
)


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


def test_load_managed_config_parses_claude_and_codex_together(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text(
        "\n".join(
            [
                "claude_code:",
                "  version: 2.1.206",
                "  model: claude-sonnet-4-5",
                "codex:",
                "  version: 0.144.2",
                "  package: '@openai/codex'",
                "  model: gpt-5-codex",
                "policy_version: 3",
            ]
        )
    )

    config = _load_managed_config(path)

    assert config.claude_code.version == "2.1.206"
    assert config.codex.version == "0.144.2"
    assert config.codex.package == "@openai/codex"
    assert config.codex.model == "gpt-5-codex"
    assert config.policy_version == 3


def test_load_managed_config_codex_defaults_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text("claude_code:\n  version: 2.1.206\n")

    config = _load_managed_config(path)

    assert config.codex.version is None
    assert config.codex.package == "@openai/codex"


def test_load_managed_config_missing_file_returns_empty_policy(tmp_path: Path) -> None:
    config = _load_managed_config(tmp_path / "does_not_exist.yaml")

    assert config.claude_code.version is None
    assert config.codex.version is None
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


def test_endpoint_serves_policy_from_env_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "relay_settings.yaml"
    path.write_text(
        "\n".join(
            [
                "claude_code:",
                "  version: 2.1.206",
                "codex:",
                "  version: 0.144.2",
                "policy_version: 9",
            ]
        )
    )
    monkeypatch.setenv(RELAY_SETTINGS_PATH_ENV, str(path))

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: None

    with TestClient(app) as client:
        response = client.get("/relay/managed-config")

    assert response.status_code == 200
    body = response.json()
    assert body["claude_code"]["version"] == "2.1.206"
    assert body["codex"]["version"] == "0.144.2"
    assert body["policy_version"] == 9


def test_endpoint_requires_authentication() -> None:
    app = FastAPI()
    app.include_router(router)

    def _reject() -> None:
        raise HTTPException(status_code=401, detail="unauthorized")

    app.dependency_overrides[user_api_key_auth] = _reject

    with TestClient(app) as client:
        response = client.get("/relay/managed-config")

    assert response.status_code == 401

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.zai.chat.transformation import ZAIChatConfig

REPO_ROOT: Final = Path(__file__).parents[2]
MAIN_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

ZAI_API_KEY_ENV_VARS: Final = ("ZAI_API_KEY", "BIGMODEL_API_KEY", "ZHIPUAI_API_KEY")
LEGACY_API_KEY_ENV_VARS: Final = ("BIGMODEL_API_KEY", "ZHIPUAI_API_KEY")
NEW_ZAI_MODEL_KEYS: Final = ("zai/GLM-5.3", "zai/GLM-5.3-Flash", "zai/glm-5v-turbo", "zai/GLM-5V-Turbo")


def _load(path: Path) -> Mapping[str, object]:
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve get_model_info against the in-repo cost map instead of the remote one
    fetched at import time."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("legacy_env_var", LEGACY_API_KEY_ENV_VARS)
def test_provider_info_resolves_legacy_api_key(monkeypatch: pytest.MonkeyPatch, legacy_env_var: str) -> None:
    for env_var in ZAI_API_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv(legacy_env_var, "legacy-secret")

    api_base, api_key = ZAIChatConfig()._get_openai_compatible_provider_info(api_base=None, api_key=None)

    assert api_key == "legacy-secret"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_provider_info_prefers_zai_api_key_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in ZAI_API_KEY_ENV_VARS:
        monkeypatch.setenv(env_var, f"{env_var}-secret")

    _, api_key = ZAIChatConfig()._get_openai_compatible_provider_info(api_base=None, api_key=None)

    assert api_key == "ZAI_API_KEY-secret"


def test_supported_openai_params_include_new_params() -> None:
    params: Final = ZAIChatConfig().get_supported_openai_params(model="zai/glm-5.3")

    assert "response_format" in params
    assert "max_completion_tokens" in params


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("zai/GLM-5.3", "zai/glm-5.3"),
        ("zai/GLM-5.3-Flash", "zai/glm-5.3-flash"),
        ("zai/GLM-5V-Turbo", "zai/glm-5v-turbo"),
    ],
)
def test_uppercase_alias_model_info_matches_canonical(local_model_cost_map: None, alias: str, canonical: str) -> None:
    alias_info: Final = litellm.get_model_info(alias)
    canonical_info: Final = litellm.get_model_info(canonical)

    assert alias_info["key"] == alias
    assert {k: v for k, v in alias_info.items() if k != "key"} == {
        k: v for k, v in canonical_info.items() if k != "key"
    }


def test_glm_5v_turbo_model_info(local_model_cost_map: None) -> None:
    info: Final = litellm.get_model_info("zai/glm-5v-turbo")

    assert info["litellm_provider"] == "zai"
    assert info["mode"] == "chat"
    assert info["supports_vision"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True


def test_new_zai_entries_match_between_main_and_backup() -> None:
    main_cost: Final = _load(MAIN_PATH)
    backup_cost: Final = _load(BACKUP_PATH)

    for model in NEW_ZAI_MODEL_KEYS:
        assert backup_cost.get(model) == main_cost.get(model), (
            f"{model} differs between main and backup model cost maps"
        )

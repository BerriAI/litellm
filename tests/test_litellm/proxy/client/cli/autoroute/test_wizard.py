from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import click
import yaml
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.autoroute import wizard as wizard_module
from litellm.proxy.client.cli.commands.autoroute.config import DiscoveredModel
from litellm.proxy.client.cli.commands.autoroute.wizard import (
    _render_and_prompt_for_model,
    run_configure_wizard,
)

CHAT_AND_EMBEDDING_GROUPS: List[Dict[str, Any]] = [
    {"model_group": "gpt-4o-mini", "mode": "chat", "input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
    {"model_group": "gpt-4o", "mode": "chat", "input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
    {"model_group": "claude-opus", "mode": "chat"},
    {"model_group": "o1", "mode": "chat"},
    {"model_group": "text-embedding-3-small", "mode": "embedding"},
]

CHAT_ONLY_GROUPS: List[Dict[str, Any]] = [
    {"model_group": "gpt-4o-mini", "mode": "chat"},
    {"model_group": "gpt-4o", "mode": "chat"},
    {"model_group": "claude-opus", "mode": "chat"},
    {"model_group": "o1", "mode": "chat"},
]

EMBEDDING_ONLY_GROUPS: List[Dict[str, Any]] = [
    {"model_group": "text-embedding-3-small", "mode": "embedding"},
]


@click.command()
@click.pass_context
def _invoke_wizard(ctx: click.Context) -> None:
    run_configure_wizard(ctx)


def _run(tmp_path, raw_groups: List[Dict[str, Any]], input_str: str):
    config_path = tmp_path / "config.yaml"
    runner = CliRunner()
    with (
        patch.object(wizard_module, "Client") as mock_client_cls,
        patch.object(wizard_module, "CONFIG_PATH", config_path),
    ):
        mock_client_cls.return_value.model_groups.info.return_value = raw_groups
        result = runner.invoke(
            _invoke_wizard,
            obj={"base_url": "http://localhost:4000", "api_key": "sk-test"},
            input=input_str,
        )
    return result, config_path


def _router_config(config_path) -> Dict[str, Any]:
    written = yaml.safe_load(config_path.read_text())
    autorouter = next(m for m in written["model_list"] if m["model_name"] == "autorouter")
    return autorouter["litellm_params"]["complexity_router_config"]


class TestRunConfigureWizardHappyPath:
    def test_assigns_tiers_and_declines_everything(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_AND_EMBEDDING_GROUPS,
            input_str="1\n2\n3\n4\nn\nn\nn\n",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["tiers"] == {
            "SIMPLE": "gpt-4o-mini",
            "MEDIUM": "gpt-4o",
            "COMPLEX": "claude-opus",
            "REASONING": "o1",
        }
        assert router_config["default_model"] == "gpt-4o"
        assert "classifier_type" not in router_config
        assert "classifier_llm_config" not in router_config
        assert "semantic_keyword_matching" not in router_config
        assert "adaptive" not in router_config

    def test_writes_config_file_with_restricted_permissions(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_AND_EMBEDDING_GROUPS,
            input_str="1\n2\n3\n4\nn\nn\nn\n",
        )

        assert result.exit_code == 0, result.output
        assert config_path.exists()
        assert oct(config_path.stat().st_mode)[-3:] == "600"

    def test_no_embedding_pool_skips_semantic_prompt_entirely(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_ONLY_GROUPS,
            input_str="1\n2\n3\n4\nn\nn\n",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert "semantic_keyword_matching" not in router_config


class TestRunConfigureWizardLLMClassifier:
    def test_accepting_llm_classifier_records_chosen_model(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_AND_EMBEDDING_GROUPS,
            input_str="1\n2\n3\n4\ny\n2\nn\nn\n",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["classifier_type"] == "llm"
        assert router_config["classifier_llm_config"]["model"] == "gpt-4o"


class TestRunConfigureWizardSemanticMatching:
    def test_accepting_semantic_matching_records_embedding_model(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_AND_EMBEDDING_GROUPS,
            input_str="1\n2\n3\n4\nn\ny\n1\nn\n",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["semantic_keyword_matching"] is True
        assert router_config["embedding_model"] == "text-embedding-3-small"


class TestRunConfigureWizardAdaptive:
    def test_accepting_adaptive_sets_adaptive_flag(self, tmp_path):
        result, config_path = _run(
            tmp_path,
            CHAT_AND_EMBEDDING_GROUPS,
            input_str="1\n2\n3\n4\nn\nn\ny\n",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["adaptive"] is True


class TestRunConfigureWizardNoChatModels:
    def test_fails_cleanly_without_prompting_when_no_chat_models(self, tmp_path):
        result, config_path = _run(tmp_path, EMBEDDING_ONLY_GROUPS, input_str="")

        assert result.exit_code != 0
        assert "no chat-capable models" in result.output.lower()
        assert not config_path.exists()


class TestRenderAndPromptForModel:
    def _models(self) -> Tuple[DiscoveredModel, ...]:
        return (
            DiscoveredModel(name="model-a"),
            DiscoveredModel(name="model-b"),
        )

    def test_reprompts_on_non_numeric_input(self):
        with patch("click.prompt", side_effect=["not-a-number", "2"]):
            result = _render_and_prompt_for_model(self._models(), "test tier")

        assert result == "model-b"

    def test_reprompts_on_out_of_range_index(self):
        with patch("click.prompt", side_effect=["5", "1"]):
            result = _render_and_prompt_for_model(self._models(), "test tier")

        assert result == "model-a"

    def test_reprompts_on_zero_index(self):
        with patch("click.prompt", side_effect=["0", "2"]):
            result = _render_and_prompt_for_model(self._models(), "test tier")

        assert result == "model-b"

    def test_valid_first_answer_returns_immediately(self):
        with patch("click.prompt", return_value="1") as mock_prompt:
            result = _render_and_prompt_for_model(self._models(), "test tier")

        assert result == "model-a"
        mock_prompt.assert_called_once()

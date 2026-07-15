import asyncio
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import click
import pytest
import yaml
from click.testing import CliRunner
from InquirerPy.base.control import Choice
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from litellm.proxy.client.cli.commands.autoroute import wizard as wizard_module
from litellm.proxy.client.cli.commands.autoroute.config import DiscoveredModel
from litellm.proxy.client.cli.commands.autoroute.wizard import run_configure_wizard

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


def _run(
    tmp_path,
    raw_groups: List[Dict[str, Any]],
    tier_picks: Dict[str, Tuple[str, ...]],
    input_str: str,
    classifier_pick: str = "",
    embedding_pick: str = "",
):
    """Drives run_configure_wizard's orchestration logic (discovery, validation, config writing,
    classifier/semantic/adaptive branching) by mocking the fuzzy picker itself, since that widget
    is a real prompt_toolkit application tested separately in TestFuzzyPickWidget. CliRunner's
    injected input still drives the plain click.confirm() y/n prompts."""
    config_path = tmp_path / "config.yaml"
    runner = CliRunner()

    def _fake_prompt_for_models(models, prompt_label):
        return tier_picks[prompt_label]

    def _fake_prompt_for_model(models, prompt_label):
        if prompt_label == "LLM classifier":
            return classifier_pick
        if prompt_label == "semantic embeddings":
            return embedding_pick
        raise AssertionError(f"unexpected single-pick prompt_label {prompt_label!r}")

    with (
        patch.object(wizard_module, "Client") as mock_client_cls,
        patch.object(wizard_module, "CONFIG_PATH", config_path),
        patch.object(wizard_module, "_is_interactive", return_value=True),
        patch.object(wizard_module, "_render_and_prompt_for_models", side_effect=_fake_prompt_for_models),
        patch.object(wizard_module, "_render_and_prompt_for_model", side_effect=_fake_prompt_for_model),
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


_SIMPLE_TIER_PICKS: Dict[str, Tuple[str, ...]] = {
    "SIMPLE": ("gpt-4o-mini",),
    "MEDIUM": ("gpt-4o",),
    "COMPLEX": ("claude-opus",),
    "REASONING": ("o1",),
}


class TestRunConfigureWizardHappyPath:
    def test_assigns_tiers_and_declines_everything(self, tmp_path):
        result, config_path = _run(tmp_path, CHAT_AND_EMBEDDING_GROUPS, _SIMPLE_TIER_PICKS, input_str="n\nn\nn\n")

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["tiers"] == {
            "SIMPLE": ["gpt-4o-mini"],
            "MEDIUM": ["gpt-4o"],
            "COMPLEX": ["claude-opus"],
            "REASONING": ["o1"],
        }
        assert router_config["default_model"] == "gpt-4o"
        assert "classifier_type" not in router_config
        assert "classifier_llm_config" not in router_config
        assert "semantic_keyword_matching" not in router_config
        assert "adaptive" not in router_config

    def test_assigns_multiple_models_to_a_single_tier(self, tmp_path):
        tier_picks = {**_SIMPLE_TIER_PICKS, "SIMPLE": ("gpt-4o-mini", "gpt-4o")}
        result, config_path = _run(tmp_path, CHAT_AND_EMBEDDING_GROUPS, tier_picks, input_str="n\nn\nn\n")

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["tiers"]["SIMPLE"] == ["gpt-4o-mini", "gpt-4o"]
        assert router_config["default_model"] == "gpt-4o"

    def test_writes_config_file_with_restricted_permissions(self, tmp_path):
        result, config_path = _run(tmp_path, CHAT_AND_EMBEDDING_GROUPS, _SIMPLE_TIER_PICKS, input_str="n\nn\nn\n")

        assert result.exit_code == 0, result.output
        assert config_path.exists()
        assert oct(config_path.stat().st_mode)[-3:] == "600"

    def test_no_embedding_pool_skips_semantic_prompt_entirely(self, tmp_path):
        result, config_path = _run(tmp_path, CHAT_ONLY_GROUPS, _SIMPLE_TIER_PICKS, input_str="n\nn\n")

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert "semantic_keyword_matching" not in router_config


class TestRunConfigureWizardLLMClassifier:
    def test_accepting_llm_classifier_records_chosen_model(self, tmp_path):
        result, config_path = _run(
            tmp_path, CHAT_AND_EMBEDDING_GROUPS, _SIMPLE_TIER_PICKS, input_str="y\nn\nn\n", classifier_pick="gpt-4o"
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
            _SIMPLE_TIER_PICKS,
            input_str="n\ny\nn\n",
            embedding_pick="text-embedding-3-small",
        )

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["semantic_keyword_matching"] is True
        assert router_config["embedding_model"] == "text-embedding-3-small"


class TestRunConfigureWizardAdaptive:
    def test_accepting_adaptive_sets_adaptive_flag(self, tmp_path):
        result, config_path = _run(tmp_path, CHAT_AND_EMBEDDING_GROUPS, _SIMPLE_TIER_PICKS, input_str="n\nn\ny\n")

        assert result.exit_code == 0, result.output
        router_config = _router_config(config_path)
        assert router_config["adaptive"] is True


class TestRunConfigureWizardNoChatModels:
    def test_fails_cleanly_without_prompting_when_no_chat_models(self, tmp_path):
        result, config_path = _run(tmp_path, EMBEDDING_ONLY_GROUPS, {}, input_str="")

        assert result.exit_code != 0
        assert "no chat-capable models" in result.output.lower()
        assert not config_path.exists()

    def test_surfaces_clean_error_when_response_is_not_a_list(self, tmp_path):
        result, config_path = _run(tmp_path, {"data": CHAT_AND_EMBEDDING_GROUPS}, {}, input_str="")

        assert result.exit_code != 0
        assert result.exception is None or not isinstance(result.exception, AssertionError)
        assert "Unexpected response from /model_group/info" in result.output
        assert not config_path.exists()


class TestRunConfigureWizardNotInteractive:
    def test_fails_cleanly_when_not_a_tty(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        runner = CliRunner()
        with (
            patch.object(wizard_module, "Client") as mock_client_cls,
            patch.object(wizard_module, "CONFIG_PATH", config_path),
            patch.object(wizard_module, "_is_interactive", return_value=False),
        ):
            mock_client_cls.return_value.model_groups.info.return_value = CHAT_AND_EMBEDDING_GROUPS
            result = runner.invoke(_invoke_wizard, obj={"base_url": "http://localhost:4000", "api_key": "sk-test"})

        assert result.exit_code != 0
        assert "interactive terminal" in result.output
        assert not config_path.exists()


def _drive_fuzzy_pick(
    models: Tuple[DiscoveredModel, ...],
    prompt_label: str,
    multiselect: bool,
    key_events: List[Tuple[str, float]],
) -> List[str]:
    """Drives the real InquirerPy fuzzy prompt through prompt_toolkit's own test input/output,
    exercising the actual widget (filtering, tab-to-toggle, enter-to-confirm) rather than mocking
    it away. asyncio.to_thread propagates the create_app_session context into the worker thread
    running _fuzzy_pick's synchronous .execute() call."""

    async def _run() -> List[str]:
        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input, output=DummyOutput()):
                task = asyncio.ensure_future(
                    asyncio.to_thread(wizard_module._fuzzy_pick, models, prompt_label, multiselect)
                )
                await asyncio.sleep(0.05)
                for text, delay in key_events:
                    pipe_input.send_text(text)
                    await asyncio.sleep(delay)
                return await task

    return asyncio.run(_run())


class TestFuzzyPickWidget:
    def _models(self) -> Tuple[DiscoveredModel, ...]:
        return tuple(DiscoveredModel(name=f"model-{i}") for i in range(20))

    def test_single_select_filters_and_returns_highlighted_match(self):
        result = _drive_fuzzy_pick(
            self._models(), "test", multiselect=False, key_events=[("model-13", 0.3), ("\r", 0.1)]
        )
        assert result == ["model-13"]

    def test_multiselect_requires_tab_to_toggle_before_enter(self):
        result = _drive_fuzzy_pick(
            self._models(), "test", multiselect=True, key_events=[("model-7", 0.3), ("\t", 0.1), ("\r", 0.1)]
        )
        assert result == ["model-7"]

    def test_multiselect_can_pick_more_than_one_across_filters(self):
        result = _drive_fuzzy_pick(
            self._models(),
            "test",
            multiselect=True,
            key_events=[
                ("model-3", 0.3),
                ("\t", 0.1),
                *[("\x7f", 0.02) for _ in range("model-3".__len__())],
                ("model-15", 0.3),
                ("\t", 0.1),
                ("\r", 0.1),
            ],
        )
        assert set(result) == {"model-3", "model-15"}

    def test_choice_wraps_name_and_value_to_the_same_model_name(self):
        model = DiscoveredModel(name="only-model")
        choice = Choice(value=model.name, name=model.name)
        assert choice.value == choice.name == "only-model"


class TestRenderAndPromptForModelWrappers:
    def test_single_pick_wrapper_returns_bare_string(self):
        with patch.object(wizard_module, "_fuzzy_pick", return_value=["model-a"]) as mock_pick:
            result = wizard_module._render_and_prompt_for_model((), "tier")
        assert result == "model-a"
        mock_pick.assert_called_once_with((), "tier", multiselect=False)

    def test_multi_pick_wrapper_returns_tuple(self):
        with patch.object(wizard_module, "_fuzzy_pick", return_value=["model-a", "model-b"]) as mock_pick:
            result = wizard_module._render_and_prompt_for_models((), "tier")
        assert result == ("model-a", "model-b")
        mock_pick.assert_called_once_with((), "tier", multiselect=True)


@pytest.mark.parametrize("isatty_value", [True, False])
def test_is_interactive_reflects_stdin_isatty(isatty_value):
    with patch.object(wizard_module.sys.stdin, "isatty", return_value=isatty_value):
        assert wizard_module._is_interactive() is isatty_value

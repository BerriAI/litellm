import os
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.server import local_config_path


@pytest.fixture
def config_file(tmp_path: Path) -> str:
    path: Final = tmp_path / "config.yaml"
    path.write_text("model_list: []\n")
    return str(path)


def test_the_cli_config_file_is_the_one_python_loads(config_file: str) -> None:
    assert local_config_path(config_file, {}) == config_file


def test_a_relative_cli_path_is_handed_over_absolute(config_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(os.path.dirname(config_file))
    handed_over: Final = local_config_path("config.yaml", {})
    assert handed_over is not None
    assert os.path.isabs(handed_over)
    assert os.path.samefile(handed_over, config_file)


def test_config_file_path_wins_over_the_cli_flag(tmp_path: Path, config_file: str) -> None:
    from_environment: Final = tmp_path / "from_environment.yml"
    from_environment.write_text("model_list: []\n")
    assert local_config_path(config_file, {"CONFIG_FILE_PATH": str(from_environment)}) == str(from_environment)


def test_config_file_path_that_names_no_file_hides_the_cli_flag(tmp_path: Path, config_file: str) -> None:
    assert local_config_path(config_file, {"CONFIG_FILE_PATH": str(tmp_path / "missing.yaml")}) is None


def test_a_config_kept_in_a_bucket_is_not_a_local_file(config_file: str) -> None:
    assert local_config_path(config_file, {"LITELLM_CONFIG_BUCKET_NAME": "configs"}) is None


def test_a_file_that_is_not_yaml_is_not_loaded(tmp_path: Path) -> None:
    not_yaml: Final = tmp_path / "config.json"
    not_yaml.write_text("{}")
    assert local_config_path(str(not_yaml), {}) is None


def test_no_config_anywhere_hands_over_nothing() -> None:
    assert local_config_path(None, {}) is None

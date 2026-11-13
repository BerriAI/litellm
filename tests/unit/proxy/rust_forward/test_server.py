from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.rust_forward.server import local_config_path


@pytest.fixture
def files(tmp_path: Path) -> Path:
    for name in ("cli.yaml", "env.yml", "notes.txt"):
        (tmp_path / name).write_text("model_list: []\n")
    return tmp_path


@pytest.mark.parametrize(
    ("cli_config", "environ", "loaded"),
    [
        ("cli.yaml", {}, "cli.yaml"),
        (None, {"CONFIG_FILE_PATH": "env.yml"}, "env.yml"),
        ("cli.yaml", {"CONFIG_FILE_PATH": "env.yml"}, "env.yml"),
        ("cli.yaml", {"CONFIG_FILE_PATH": "missing.yaml"}, None),
        ("cli.yaml", {"LITELLM_CONFIG_BUCKET_NAME": "bucket"}, None),
        ("missing.yaml", {}, None),
        ("notes.txt", {}, None),
        (None, {}, None),
    ],
)
def test_names_the_local_yaml_file_the_proxy_loads(
    files: Path, cli_config: str | None, environ: dict[str, str], loaded: str | None
) -> None:
    in_files: Final = {
        name: str(files / value) if name == "CONFIG_FILE_PATH" else value for name, value in environ.items()
    }
    path: Final = local_config_path(None if cli_config is None else str(files / cli_config), in_files)
    assert path == (None if loaded is None else str(files / loaded))


def test_a_relative_path_is_made_absolute(files: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(files)
    assert local_config_path("cli.yaml", {}) == str(files / "cli.yaml")

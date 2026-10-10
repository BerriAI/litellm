import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands import agents as agents_module
from litellm.proxy.client.cli.commands.agents import which_ignoring_alias_dir
from litellm.proxy.client.cli.commands.alias import (
    AliasError,
    default_shim_dir,
    ensure_rc_lines,
    find_real_binary,
    install_shim,
    is_our_shim,
    rc_candidates,
    remove_shim,
    shim_content,
    shim_filename,
)
from litellm.proxy.client.cli.main import cli

ALIAS_MODULE = "litellm.proxy.client.cli.commands.alias"
AGENTS_MODULE = "litellm.proxy.client.cli.commands.agents"

IS_WINDOWS: bool = sys.platform.startswith("win")


def _fake_agent_binary(directory: Path, name: str = "claude") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary: Path = directory / (f"{name}.cmd" if IS_WINDOWS else name)
    binary.write_text("@echo off\nexit /b 0\n" if IS_WINDOWS else "#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary


class TestShimContent:
    def test_posix_shim_forwards_agent_and_args(self, tmp_path: Path):
        content: str = shim_content("claude", tmp_path, "/usr/local/bin/lite", platform="linux")
        assert content.startswith("#!/bin/sh")
        assert f'LITELLM_ALIAS_SHIM_DIR="{tmp_path}" exec /usr/local/bin/lite claude "$@"' in content
        assert "installed by `lite alias claude`" in content
        assert "lite unalias claude" in content

    def test_windows_shim_uses_cmd_extension_and_star_args(self, tmp_path: Path):
        content: str = shim_content("claude", tmp_path, r"C:\bin\lite.exe", platform="win32")
        assert 'set "LITELLM_ALIAS_SHIM_DIR=' in content
        assert r'"C:\bin\lite.exe" claude %*' in content
        assert "installed by `lite alias claude`" in content

    def test_shim_filename_matches_platform(self):
        assert shim_filename("claude", "linux") == "claude"
        assert shim_filename("claude", "win32") == "claude.cmd"


class TestInstallRemoveShim:
    def test_install_then_remove_roundtrip(self, tmp_path: Path):
        path, overwrote = install_shim("claude", tmp_path, platform="linux", lite_bin="/bin/lite")
        assert path == tmp_path / "claude"
        assert overwrote is False
        assert is_our_shim(path, "claude")
        assert remove_shim("claude", tmp_path, platform="linux") == path
        assert not path.exists()

    def test_install_refuses_foreign_file_without_force(self, tmp_path: Path):
        foreign: Path = tmp_path / "claude"
        foreign.write_text("#!/bin/sh\nsome other claude wrapper\n")
        with pytest.raises(AliasError, match="was not installed by"):
            install_shim("claude", tmp_path, platform="linux")
        assert "some other claude wrapper" in foreign.read_text()
        install_shim("claude", tmp_path, platform="linux", force=True)
        assert is_our_shim(foreign, "claude")

    def test_refresh_replaces_our_own_shim_without_force(self, tmp_path: Path):
        install_shim("claude", tmp_path, platform="linux", lite_bin="/bin/lite")
        _, overwrote = install_shim("claude", tmp_path, platform="linux", lite_bin="/bin/lite2")
        assert overwrote is True
        assert "/bin/lite2" in (tmp_path / "claude").read_text()

    def test_remove_missing_shim_raises(self, tmp_path: Path):
        with pytest.raises(AliasError, match="not aliased"):
            remove_shim("claude", tmp_path, platform="linux")

    def test_remove_refuses_foreign_file(self, tmp_path: Path):
        foreign: Path = tmp_path / "claude"
        foreign.write_text("#!/bin/sh\nsome other claude wrapper\n")
        with pytest.raises(AliasError, match="was not installed by"):
            remove_shim("claude", tmp_path, platform="linux")
        assert foreign.exists()

    @pytest.mark.skipif(IS_WINDOWS, reason="POSIX shim script")
    def test_generated_shim_runs_and_forwards_arguments(self, tmp_path: Path):
        fake_lite: Path = tmp_path / "fake-lite"
        fake_lite.write_text('#!/bin/sh\necho "agent=$1 marker=$LITELLM_ALIAS_SHIM_DIR shifted=$2 args=$#"\n')
        fake_lite.chmod(0o755)
        shim_dir: Path = tmp_path / "shims"
        install_shim("claude", shim_dir, platform="linux", lite_bin=str(fake_lite))
        result: subprocess.CompletedProcess[str] = subprocess.run(
            [str(shim_dir / "claude"), "--print", "hi there"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert f"agent=claude marker={shim_dir}" in result.stdout
        assert "shifted=--print" in result.stdout
        # the agent name plus both user args arrive: $#=3, all forwarded untouched
        assert "args=3" in result.stdout


class TestRcLines:
    def test_bash_rc_gets_path_export(self, tmp_path: Path):
        written = ensure_rc_lines(tmp_path, "/bin/bash")
        assert written == (tmp_path / ".bashrc",)
        content: str = (tmp_path / ".bashrc").read_text()
        assert 'export PATH="$HOME/.litellm/bin:$PATH"' in content

    def test_zsh_and_fish_get_their_rc_files(self, tmp_path: Path):
        zsh_rc, zsh_block = rc_candidates(tmp_path, "/usr/bin/zsh")[0]
        assert zsh_rc == tmp_path / ".zshrc"
        assert 'export PATH="$HOME/.litellm/bin:$PATH"' in zsh_block
        fish_rc, fish_block = rc_candidates(tmp_path, "/usr/bin/fish")[0]
        assert fish_rc == tmp_path / ".config" / "fish" / "config.fish"
        assert "fish_add_path -g $HOME/.litellm/bin" in fish_block

    def test_second_run_does_not_duplicate(self, tmp_path: Path):
        ensure_rc_lines(tmp_path, "/bin/bash")
        assert ensure_rc_lines(tmp_path, "/bin/bash") == ()
        assert (tmp_path / ".bashrc").read_text().count("export PATH=") == 1

    def test_fish_rc_uses_fish_add_path(self, tmp_path: Path):
        ensure_rc_lines(tmp_path, "/usr/bin/fish")
        content: str = (tmp_path / ".config" / "fish" / "config.fish").read_text()
        assert "fish_add_path -g $HOME/.litellm/bin" in content


class TestLoopGuard:
    def test_which_ignoring_alias_dir_resolves_past_shim(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        shim_dir: Path = tmp_path / "shims"
        real_dir: Path = tmp_path / "real"
        real: Path = _fake_agent_binary(real_dir)
        install_shim("claude", shim_dir, platform="linux", lite_bin="/bin/lite")
        monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")

        # The shim shadows the real binary on PATH...
        assert Path(shutil.which("claude")) == shim_dir / "claude"
        # ...but a marked launch resolves past its own directory.
        monkeypatch.setenv("LITELLM_ALIAS_SHIM_DIR", str(shim_dir))
        assert which_ignoring_alias_dir("claude") == str(real)

    def test_which_ignoring_alias_dir_without_marker_behaves_like_which(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        real_dir: Path = tmp_path / "real"
        real: Path = _fake_agent_binary(real_dir)
        monkeypatch.setenv("PATH", str(real_dir))
        monkeypatch.delenv("LITELLM_ALIAS_SHIM_DIR", raising=False)
        assert which_ignoring_alias_dir("claude") == str(real)

    def test_run_agent_resolves_past_shim_by_default(self):
        default_which = inspect.signature(agents_module.run_agent).parameters["which"].default
        assert default_which is which_ignoring_alias_dir

    def test_find_real_binary_temporarily_marks_shim_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        shim_dir: Path = tmp_path / "shims"
        real_dir: Path = tmp_path / "real"
        real: Path = _fake_agent_binary(real_dir)
        install_shim("claude", shim_dir, platform="linux", lite_bin="/bin/lite")
        monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")
        monkeypatch.delenv("LITELLM_ALIAS_SHIM_DIR", raising=False)

        assert find_real_binary("claude", shim_dir) == str(real)
        assert "LITELLM_ALIAS_SHIM_DIR" not in os.environ


class TestAliasCommands:
    def test_alias_and_unalias_help_list_public_agents(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["alias", "--help"])
        assert result.exit_code == 0
        assert "claude" in result.output
        result = runner.invoke(cli, ["unalias", "--help"])
        assert result.exit_code == 0
        assert "claude" in result.output

    def test_alias_claude_installs_shim_and_rc_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        real_dir: Path = tmp_path / "real"
        _fake_agent_binary(real_dir)
        monkeypatch.setenv("PATH", str(real_dir))
        monkeypatch.setenv("SHELL", "/bin/bash")

        runner = CliRunner()
        result = runner.invoke(cli, ["alias", "claude"])
        assert result.exit_code == 0, result.output
        shim: Path = default_shim_dir() / "claude"
        assert is_our_shim(shim, "claude")
        assert 'export PATH="$HOME/.litellm/bin:$PATH"' in (Path.home() / ".bashrc").read_text()
        assert "lite claude" in result.output

        # unalias removes the shim again
        result = runner.invoke(cli, ["unalias", "claude"])
        assert result.exit_code == 0, result.output
        assert not shim.exists()

    def test_alias_claude_without_real_binary_fails_with_install_hint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PATH", "")
        runner = CliRunner()
        result = runner.invoke(cli, ["alias", "claude"])
        assert result.exit_code != 0
        assert "nothing to alias" in result.output
        assert "docs.claude.com" in result.output

    def test_alias_claude_refuses_foreign_file_without_force(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        real_dir: Path = tmp_path / "real"
        _fake_agent_binary(real_dir)
        monkeypatch.setenv("PATH", str(real_dir))
        monkeypatch.setenv("SHELL", "/bin/bash")
        shim: Path = default_shim_dir() / "claude"
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text("#!/bin/sh\nsomeone else's claude\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["alias", "claude"])
        assert result.exit_code != 0
        assert "was not installed by" in result.output
        assert "someone else's claude" in shim.read_text()

        result = runner.invoke(cli, ["alias", "claude", "--force"])
        assert result.exit_code == 0, result.output
        assert is_our_shim(shim, "claude")

    def test_unalias_claude_without_shim_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["unalias", "claude"])
        assert result.exit_code != 0
        assert "not aliased" in result.output

    def test_unalias_refuses_foreign_file(self, tmp_path: Path):
        shim: Path = default_shim_dir() / "claude"
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text("#!/bin/sh\nsomeone else's claude\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["unalias", "claude"])
        assert result.exit_code != 0
        assert "was not installed by" in result.output
        assert shim.exists()

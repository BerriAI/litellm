"""Unit tests for litellm.proxy.guardrails.shunt_rewrite."""

import subprocess
from pathlib import Path

import pytest

from litellm.proxy.guardrails.shunt_rewrite import (
    ShuntBashRewrite,
    build_bounded_read_command,
    build_bulk_read_command,
    build_code_write_command,
    extract_bare_read_path,
    is_targeted_read,
)


class TestIsTargetedRead:
    def test_neither_set_is_untargeted(self):
        assert is_targeted_read(None, None) is False

    def test_offset_set_is_targeted(self):
        assert is_targeted_read(100, None) is True

    def test_limit_set_is_targeted(self):
        assert is_targeted_read(None, 50) is True

    def test_both_set_is_targeted(self):
        assert is_targeted_read(100, 50) is True

    def test_offset_zero_is_targeted(self):
        """Matches shunt's own documented bypass: offset:0 counts as targeted."""
        assert is_targeted_read(0, None) is True

    def test_limit_zero_is_targeted(self):
        assert is_targeted_read(None, 0) is True


class TestExtractBareReadPath:
    # Ported one-for-one from shunt's own evals/bash-hook-evals.json fixture cases.
    @pytest.mark.parametrize(
        "command,expected",
        [
            ("cat file.txt", "file.txt"),
            ("cat -n file.txt", "file.txt"),
            ("head file.txt", "file.txt"),
            ("head -100 file.txt", "file.txt"),
            ("tail file.txt", "file.txt"),
            ("less file.txt", "file.txt"),
            ("more file.txt", "file.txt"),
            ('cat "file.txt"', "file.txt"),
            ("cat file.txt | grep export", None),
            ("cat file.txt > /tmp/out.txt", None),
            ("git status", None),
            ("grep -n 'export' file.txt", None),
            ("cat /tmp/does-not-exist.txt", "/tmp/does-not-exist.txt"),
            ("", None),
        ],
    )
    def test_ported_shunt_eval_cases(self, command: str, expected: str | None):
        assert extract_bare_read_path(command) == expected

    def test_fixes_shunts_documented_parser_bug(self):
        """shunt's own parser returns "5" for `head -n 5 file` (a flag's value token mistaken
        for the path); this port skips a bare-numeric token instead."""
        assert extract_bare_read_path("head -n 5 file.txt") == "file.txt"

    def test_fixes_tails_plus_n_follow_syntax_the_same_way(self):
        assert extract_bare_read_path("tail -n +5 file.log") == "file.log"

    def test_no_non_flag_argument_returns_none(self):
        assert extract_bare_read_path("cat -n") is None


def _bounded_read(
    path: str = "litellm/router.py",
    question: str = "Summarize this file's structure.",
    min_lines: int = 350,
    bulk_read_endpoint: str = "http://localhost:4000/v1/bulk_read",
) -> ShuntBashRewrite:
    return build_bounded_read_command(
        path=path,
        question=question,
        min_lines=min_lines,
        bulk_read_endpoint=bulk_read_endpoint,
    )


def _assert_valid_bash(command: str) -> None:
    result = subprocess.run(["bash", "-n", "-c", command], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


class TestBuildBoundedReadCommand:
    def test_returns_a_shunt_bash_rewrite(self):
        assert isinstance(_bounded_read(), ShuntBashRewrite)

    def test_command_is_valid_bash(self):
        _assert_valid_bash(_bounded_read().command)

    def test_command_names_the_threshold(self):
        assert "-gt 350" in _bounded_read(min_lines=350).command

    def test_command_names_the_endpoint(self):
        endpoint = "http://localhost:4000/v1/bulk_read?router=shunt"
        assert endpoint in _bounded_read(bulk_read_endpoint=endpoint).command

    def test_command_falls_back_to_cat_for_small_files(self):
        assert "else cat litellm/router.py; fi" in _bounded_read().command

    def test_command_escapes_embedded_double_quotes_in_path(self):
        _assert_valid_bash(_bounded_read(path='weird"file.py').command)

    def test_note_mentions_the_threshold(self):
        assert "200" in _bounded_read(min_lines=200).note


# Every value in a generated command comes from the model and is run by a shell on the
# developer's own machine, so a shell metacharacter in any of them must stay inert data.
# Regression: the values were interpolated inside double quotes with only `"` escaped, so a
# path of `$(cmd)` was command-substituted and ran `cmd` locally.
_INJECTIONS = [
    "$(touch {marker})",
    "`touch {marker}`",
    "; touch {marker}",
    "&& touch {marker}",
    "| touch {marker}",
    "$(touch {marker})'; touch {marker}; '",
    'x" ; touch {marker} ; "',
    "\n touch {marker} \n",
]


def _assert_runs_without_side_effect(command: str, marker: Path) -> None:
    """Run `command` in a real bash and assert the injected marker file was never created."""
    subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False, timeout=30)
    assert not marker.exists(), f"injection executed, marker created by: {command}"


@pytest.mark.parametrize("payload", _INJECTIONS)
class TestGeneratedCommandsResistShellInjection:
    def test_bounded_read_path_is_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_path"
        rewrite = _bounded_read(path=payload.format(marker=marker))
        _assert_runs_without_side_effect(rewrite.command, marker)

    def test_bounded_read_question_is_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_question"
        rewrite = _bounded_read(question=payload.format(marker=marker))
        _assert_runs_without_side_effect(rewrite.command, marker)

    def test_bounded_read_endpoint_is_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_endpoint"
        rewrite = _bounded_read(bulk_read_endpoint=payload.format(marker=marker))
        _assert_runs_without_side_effect(rewrite.command, marker)

    def test_bulk_read_paths_are_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_bulk"
        rewrite = build_bulk_read_command(
            question="q",
            paths=["ok.py", payload.format(marker=marker)],
            bulk_read_endpoint="http://127.0.0.1:9/v1/bulk_read",
        )
        _assert_runs_without_side_effect(rewrite.command, marker)

    def test_code_write_target_is_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_target"
        rewrite = build_code_write_command(
            spec="s",
            reference="r.py",
            target=payload.format(marker=marker),
            code_write_endpoint="http://127.0.0.1:9/v1/code_write",
        )
        _assert_runs_without_side_effect(rewrite.command, marker)

    def test_code_write_spec_is_inert(self, payload: str, tmp_path: Path):
        marker = tmp_path / "pwned_spec"
        rewrite = build_code_write_command(
            spec=payload.format(marker=marker),
            reference="r.py",
            target=None,
            code_write_endpoint="http://127.0.0.1:9/v1/code_write",
        )
        _assert_runs_without_side_effect(rewrite.command, marker)


# Regression: the caller's key was interpolated straight into the generated command, so it
# landed in the model's response, the conversation history, and the next upstream turn.
class TestGeneratedCommandsNeverCarryTheCallersCredential:
    def test_bounded_read_references_the_env_var_instead_of_a_secret(self):
        command = _bounded_read().command
        assert "ANTHROPIC_AUTH_TOKEN" in command
        assert "sk-" not in command

    def test_bulk_read_references_the_env_var_instead_of_a_secret(self):
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py"], bulk_read_endpoint="http://localhost:4000/v1/bulk_read"
        )
        assert "ANTHROPIC_AUTH_TOKEN" in rewrite.command
        assert "sk-" not in rewrite.command

    def test_code_write_references_the_env_var_instead_of_a_secret(self):
        rewrite = build_code_write_command(
            spec="s", reference="r.py", target=None, code_write_endpoint="http://localhost:4000/v1/code_write"
        )
        assert "ANTHROPIC_AUTH_TOKEN" in rewrite.command
        assert "sk-" not in rewrite.command

    def test_the_env_var_expands_at_run_time(self, tmp_path: Path):
        """The header must carry the client's real token once bash evaluates the command."""
        out = tmp_path / "seen_header.txt"
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py"], bulk_read_endpoint="http://127.0.0.1:9/v1/bulk_read"
        )
        # Echo the expanded header rather than sending it, so the assertion needs no server.
        header_only = rewrite.command.split(" -H ", 1)[1].rsplit(" ", 1)[0]
        subprocess.run(
            ["bash", "-c", f"printf '%s' {header_only} > {out}"],
            capture_output=True,
            text=True,
            check=False,
            env={"ANTHROPIC_AUTH_TOKEN": "sk-real-token", "PATH": "/usr/bin:/bin"},
        )
        assert out.read_text() == "Authorization: Bearer sk-real-token"

    def test_falls_back_to_the_api_key_env_var_for_x_api_key_clients(self, tmp_path: Path):
        out = tmp_path / "seen_header.txt"
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py"], bulk_read_endpoint="http://127.0.0.1:9/v1/bulk_read"
        )
        header_only = rewrite.command.split(" -H ", 1)[1].rsplit(" ", 1)[0]
        subprocess.run(
            ["bash", "-c", f"printf '%s' {header_only} > {out}"],
            capture_output=True,
            text=True,
            check=False,
            env={"ANTHROPIC_API_KEY": "sk-from-api-key", "PATH": "/usr/bin:/bin"},
        )
        assert out.read_text() == "Authorization: Bearer sk-from-api-key"


# Regression: the commands uploaded files as `paths[]`, but the endpoint binds them under
# `paths`. FastAPI matches the form name exactly, so every delegated read 422'd.
class TestUploadFieldNameMatchesTheEndpoint:
    def test_bounded_read_uses_the_bare_paths_field_name(self):
        command = _bounded_read().command
        assert "paths=@" in command
        assert "paths[]=@" not in command

    def test_bulk_read_uses_the_bare_paths_field_name(self):
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py", "b.py"], bulk_read_endpoint="http://localhost:4000/v1/bulk_read"
        )
        assert "paths[]=@" not in rewrite.command
        for path in ("a.py", "b.py"):
            assert f"paths=@{path}" in rewrite.command


def test_bounded_read_still_reads_a_small_file_verbatim(tmp_path: Path):
    """The quoting must not break the real path: a small file is still cat'd through."""
    target = tmp_path / "small.py"
    target.write_text("line one\nline two\n")
    rewrite = _bounded_read(path=str(target), min_lines=350)
    result = subprocess.run(["bash", "-c", rewrite.command], capture_output=True, text=True, check=False)
    assert result.stdout == "line one\nline two\n"


def test_bounded_read_reports_a_path_containing_a_dollar_sign_literally(tmp_path: Path):
    target = tmp_path / "odd$name.py"
    target.write_text("x\n")
    rewrite = _bounded_read(path=str(target), min_lines=350)
    result = subprocess.run(["bash", "-c", rewrite.command], capture_output=True, text=True, check=False)
    assert result.stdout == "x\n"


class TestBuildBulkReadCommand:
    def test_unconditional_no_size_check(self):
        rewrite = build_bulk_read_command(
            question="what does this do",
            paths=["a.py", "b.py"],
            bulk_read_endpoint="http://localhost:4000/v1/bulk_read",
        )
        assert "wc -l" not in rewrite.command
        assert "if [" not in rewrite.command

    def test_includes_every_path(self):
        rewrite = build_bulk_read_command(
            question="q",
            paths=["a.py", "b.py", "c.py"],
            bulk_read_endpoint="http://localhost:4000/v1/bulk_read",
        )
        for path in ("a.py", "b.py", "c.py"):
            assert f"paths=@{path}" in rewrite.command

    def test_command_is_valid_bash(self):
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py"], bulk_read_endpoint="http://localhost:4000/v1/bulk_read"
        )
        _assert_valid_bash(rewrite.command)


class TestBuildCodeWriteCommand:
    def test_no_target_outputs_to_stdout(self):
        rewrite = build_code_write_command(
            spec="write tests",
            reference="tests/y_test.py",
            target=None,
            code_write_endpoint="http://localhost:4000/v1/code_write",
        )
        assert ">" not in rewrite.command

    def test_target_redirects_curl_output_to_the_file(self):
        rewrite = build_code_write_command(
            spec="write tests",
            reference="tests/y_test.py",
            target="tests/x_test.py",
            code_write_endpoint="http://localhost:4000/v1/code_write",
        )
        assert rewrite.command.endswith("> tests/x_test.py")

    @pytest.mark.parametrize("target", [None, "tests/x_test.py"])
    def test_command_is_valid_bash_with_and_without_target(self, target: str | None):
        rewrite = build_code_write_command(
            spec="s", reference="r.py", target=target, code_write_endpoint="http://x/v1/code_write"
        )
        _assert_valid_bash(rewrite.command)

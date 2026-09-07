"""Unit tests for litellm.proxy.guardrails.shunt_rewrite."""

import subprocess

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
    auth_header: str = "Bearer sk-1234",
) -> ShuntBashRewrite:
    return build_bounded_read_command(
        path=path,
        question=question,
        min_lines=min_lines,
        bulk_read_endpoint=bulk_read_endpoint,
        auth_header=auth_header,
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
        assert 'else cat "litellm/router.py"; fi' in _bounded_read().command

    def test_command_escapes_embedded_double_quotes_in_path(self):
        _assert_valid_bash(_bounded_read(path='weird"file.py').command)

    def test_note_mentions_the_threshold(self):
        assert "200" in _bounded_read(min_lines=200).note


class TestBuildBulkReadCommand:
    def test_unconditional_no_size_check(self):
        rewrite = build_bulk_read_command(
            question="what does this do",
            paths=["a.py", "b.py"],
            bulk_read_endpoint="http://localhost:4000/v1/bulk_read",
            auth_header="Bearer sk-1234",
        )
        assert "wc -l" not in rewrite.command
        assert "if [" not in rewrite.command

    def test_includes_every_path(self):
        rewrite = build_bulk_read_command(
            question="q",
            paths=["a.py", "b.py", "c.py"],
            bulk_read_endpoint="http://localhost:4000/v1/bulk_read",
            auth_header="Bearer sk-1234",
        )
        for path in ("a.py", "b.py", "c.py"):
            assert f"paths[]=@{path}" in rewrite.command

    def test_command_is_valid_bash(self):
        rewrite = build_bulk_read_command(
            question="q", paths=["a.py"], bulk_read_endpoint="http://localhost:4000/v1/bulk_read", auth_header="x"
        )
        _assert_valid_bash(rewrite.command)


class TestBuildCodeWriteCommand:
    def test_no_target_outputs_to_stdout(self):
        rewrite = build_code_write_command(
            spec="write tests",
            reference="tests/y_test.py",
            target=None,
            code_write_endpoint="http://localhost:4000/v1/code_write",
            auth_header="Bearer sk-1234",
        )
        assert ">" not in rewrite.command

    def test_target_redirects_curl_output_to_the_file(self):
        rewrite = build_code_write_command(
            spec="write tests",
            reference="tests/y_test.py",
            target="tests/x_test.py",
            code_write_endpoint="http://localhost:4000/v1/code_write",
            auth_header="Bearer sk-1234",
        )
        assert '> "tests/x_test.py"' in rewrite.command

    @pytest.mark.parametrize("target", [None, "tests/x_test.py"])
    def test_command_is_valid_bash_with_and_without_target(self, target: str | None):
        rewrite = build_code_write_command(
            spec="s", reference="r.py", target=target, code_write_endpoint="http://x/v1/code_write", auth_header="x"
        )
        _assert_valid_bash(rewrite.command)

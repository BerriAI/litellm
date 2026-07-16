from e2e_result_reporter import (
    E2EResult,
    file_from_nodeid,
    format_e2e_result_line,
    outcome_from_report,
    package_from_nodeid,
    result_from_pytest,
)


def test_package_from_nodeid_uses_top_level_dir() -> None:
    assert (
        package_from_nodeid(
            "logging/test_langfuse_e2e.py::TestLangfuseTeamLogging::test_success_logs_spend"
        )
        == "logging"
    )
    assert package_from_nodeid("quota_management/budgets/test_budget_e2e.py::test_x") == "quota_management"
    assert package_from_nodeid("test_transport.py::test_x") == "root"
    assert package_from_nodeid("claude_code/tool_use/test_anthropic.py::test_x") == "claude_code"


def test_file_from_nodeid() -> None:
    assert file_from_nodeid("logging/test_langfuse_e2e.py::TestX::test_y") == "test_langfuse_e2e.py"


def test_format_e2e_result_line_is_logfmt() -> None:
    line = format_e2e_result_line(
        E2EResult(
            package="logging",
            file="test_langfuse_e2e.py",
            outcome="failed",
            duration_ms=1500,
            node_id="logging/test_langfuse_e2e.py::TestX::test_y",
            covers=("logging.langfuse.team.success",),
        )
    )
    assert line.startswith("E2E_RESULT ")
    assert "package=logging" in line
    assert "file=test_langfuse_e2e.py" in line
    assert "outcome=failed" in line
    assert "duration_ms=1500" in line
    assert "node_id=logging/test_langfuse_e2e.py::TestX::test_y" in line
    assert "covers=logging.langfuse.team.success" in line


def test_format_escapes_spaces_in_covers() -> None:
    line = format_e2e_result_line(
        E2EResult(
            package="root",
            file="t.py",
            outcome="passed",
            duration_ms=1,
            node_id="t.py::test",
            covers=("a b",),
        )
    )
    assert 'covers="a b"' in line


def test_result_from_pytest_call_failed() -> None:
    result = result_from_pytest(
        nodeid="batches/test_batches_e2e.py::test_lifecycle",
        when="call",
        failed=True,
        skipped=False,
        passed=False,
        duration_seconds=2.4,
        covers=("batches.lifecycle.works",),
    )
    assert result is not None
    assert result.package == "batches"
    assert result.outcome == "failed"
    assert result.duration_ms == 2400
    assert result.covers == ("batches.lifecycle.works",)


def test_result_from_pytest_ignores_teardown() -> None:
    assert (
        result_from_pytest(
            nodeid="t.py::test",
            when="teardown",
            failed=True,
            skipped=False,
            passed=False,
            duration_seconds=0.1,
        )
        is None
    )


def test_outcome_from_report_setup_error() -> None:
    assert outcome_from_report(when="setup", failed=True, skipped=False, passed=False) == "error"
    assert outcome_from_report(when="setup", failed=False, skipped=True, passed=False) == "skipped"

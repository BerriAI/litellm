import importlib.util
import json
import sys
from pathlib import Path
from typing import Final

_SCRIPT: Final = Path(__file__).resolve().parents[2] / "ci_cd" / "spend_logs_guard.py"
_SPEC: Final = importlib.util.spec_from_file_location("spend_logs_guard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
guard: Final = importlib.util.module_from_spec(_SPEC)
sys.modules["spend_logs_guard"] = guard
_SPEC.loader.exec_module(guard)

MIGRATION: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/20260101000000_x/migration.sql"


def _file(path: str, *lines: str, start: int = 10) -> dict[str, object]:
    """An entry of `pulls/<n>/files`, with the lines as a hunk: a leading `-` removes, anything else adds."""
    hunk: Final = "\n".join(line if line.startswith("-") else f"+{line}" for line in lines)
    return {"filename": path, "changes": len(lines), "patch": f"@@ -{start},0 +{start},{len(lines)} @@\n{hunk}"}


def _files(*entries: dict[str, object]) -> list[guard.ChangedFile]:
    return guard._FILES.validate_json(json.dumps(entries))


class TestSpendLogHits:
    def test_an_added_schema_line_naming_the_table_is_a_hit(self):
        line: Final = '  @@index([api_key, startTime], map: "LiteLLM_SpendLogs_api_key_startTime_idx")'
        assert guard.spend_log_hits(_files(_file("schema.prisma", line, start=740))) == (
            f"schema.prisma: {line.strip()}",
        )

    def test_an_added_migration_line_is_a_hit_in_any_case(self):
        files: Final = _files(_file(MIGRATION, 'CREATE INDEX "i" ON "litellm_spendlogs"("api_key");', start=3))
        assert guard.spend_log_hits(files) == (f'{MIGRATION}: CREATE INDEX "i" ON "litellm_spendlogs"("api_key");',)

    def test_a_removed_line_or_another_table_or_an_unwatched_file_is_not_a_hit(self):
        files: Final = _files(
            _file(
                "schema.prisma",
                "-model LiteLLM_SpendLogs {",
                "  team_id String  // LiteLLM_TeamTable, MyLiteLLM_SpendLogs",
            ),
            _file("litellm/proxy/utils.py", 'table = "LiteLLM_SpendLogs"'),
            {"filename": "docs/spend_logs.png", "changes": 0},
        )
        assert guard.spend_log_hits(files) == ()

    def test_two_files_are_attributed_to_their_own_path(self):
        files: Final = _files(
            _file("litellm/proxy/schema.prisma", "  a LiteLLM_SpendLogs"), _file("schema.prisma", "  b")
        )
        assert guard.spend_log_hits(files) == ("litellm/proxy/schema.prisma: a LiteLLM_SpendLogs",)


class TestUnreadable:
    def test_a_watched_file_without_a_patch_is_unreadable_but_an_unwatched_or_unchanged_one_is_not(self):
        files: Final = _files(
            {"filename": MIGRATION, "changes": 900},
            {"filename": "schema.prisma", "changes": 0},
            {"filename": "ui/logo.png", "changes": 12},
        )
        assert guard.unreadable(files) == (MIGRATION,)

    def test_a_file_list_at_the_api_cap_is_unreadable(self):
        files: Final = _files(*(_file(f"litellm/f{i}.py", "x") for i in range(guard.FILES_API_CAP)))
        assert guard.unreadable(files) == ("(file list truncated by the API)",)
        assert guard.unreadable(files[:-1]) == ()


class TestApprovedBy:
    def test_the_latest_review_of_the_approver_decides(self):
        reviews: Final = tuple(
            guard.Review(state=state, user=guard.Reviewer(login=login))
            for login, state in (
                ("kerry-berri", "CHANGES_REQUESTED"),
                ("someone-else", "APPROVED"),
                ("kerry-berri", "APPROVED"),
            )
        )
        assert guard.approved_by(reviews, "kerry-berri") is True
        assert guard.approved_by(reviews[::-1], "kerry-berri") is False

    def test_no_review_from_the_approver_is_not_an_approval(self):
        other: Final = guard.Review(state="APPROVED", user=guard.Reviewer(login="someone-else"))
        assert guard.approved_by((other,), "kerry-berri") is False
        assert guard.approved_by((), "kerry-berri") is False


def _run(tmp_path: Path, files: list[dict[str, object]], reviews: list[dict[str, object]]) -> int:
    files_path: Final = tmp_path / "files.json"
    reviews_path: Final = tmp_path / "reviews.json"
    files_path.write_text(json.dumps(files))
    reviews_path.write_text(json.dumps(reviews))
    return guard.main(["--files", str(files_path), "--reviews", str(reviews_path), "--approver", "kerry-berri"])


class TestExitCode:
    def test_a_pull_request_without_spend_log_lines_passes_without_reviews(self, tmp_path, capsys):
        assert _run(tmp_path, [_file("schema.prisma", "  foo String")], []) == 0
        assert "nothing to review" in capsys.readouterr().out

    def test_a_spend_log_change_fails_until_the_approver_approves(self, tmp_path, capsys):
        files: Final = [_file("schema.prisma", "  x LiteLLM_SpendLogs", start=7)]
        assert _run(tmp_path, files, [{"user": {"login": "someone-else"}, "state": "APPROVED"}]) == 1
        assert "schema.prisma: x LiteLLM_SpendLogs" in capsys.readouterr().out
        assert _run(tmp_path, files, [{"user": {"login": "kerry-berri"}, "state": "APPROVED"}]) == 0

    def test_a_withheld_patch_on_a_watched_file_fails_even_with_approval(self, tmp_path, capsys):
        files: Final = [{"filename": MIGRATION, "changes": 2500}]
        assert _run(tmp_path, files, [{"user": {"login": "kerry-berri"}, "state": "APPROVED"}]) == 1
        assert MIGRATION in capsys.readouterr().out

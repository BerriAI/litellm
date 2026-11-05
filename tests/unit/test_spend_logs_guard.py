from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

_SCRIPT: Final = Path(__file__).resolve().parents[2] / "ci_cd" / "spend_logs_guard.py"
_SPEC: Final = importlib.util.spec_from_file_location("spend_logs_guard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
guard: Final = importlib.util.module_from_spec(_SPEC)
sys.modules["spend_logs_guard"] = guard
_SPEC.loader.exec_module(guard)

SCHEMA: Final = (
    "generator client {\n"
    '  provider = "prisma-client-py"\n'
    "}\n"
    "\n"
    "model LiteLLM_TeamTable {\n"
    "  team_id String @id\n"
    "}\n"
    "\n"
    "model LiteLLM_SpendLogs {\n"
    "  request_id String @id\n"
    "  api_key    String\n"
    '  @@index([api_key], map: "LiteLLM_SpendLogs_api_key_idx")\n'
    "}\n"
    "\n"
    "model LiteLLM_UserTable {\n"
    "  user_id String @id\n"
    "}\n"
)
_NO_BLOCKS: Final = {}


def _hunk(old_start: int, old_count: int, new_start: int, new_count: int, *lines: str) -> str:
    body: Final = "".join(f"{line}\n" for line in lines)
    return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n{body}"


def _diff(path: str, *hunks: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n" + "".join(hunks)


def _added(path: str, added: str, start: int = 10) -> str:
    return _diff(path, _hunk(start, 0, start, 1, f"+{added}"))


def _blocks(path: str, base: str = SCHEMA, head: str = SCHEMA) -> dict[str, guard.Block]:
    return {path: guard.Block(base=guard.model_block(base), head=guard.model_block(head))}


def _with_line(schema: str, after: int, line: str) -> str:
    lines: Final = schema.splitlines(keepends=True)
    return "".join(lines[:after]) + f"{line}\n" + "".join(lines[after:])


class TestModelBlock:
    def test_the_block_runs_from_the_model_line_through_its_closing_brace(self):
        assert guard.model_block(SCHEMA) == range(9, 14)

    def test_a_schema_without_the_model_has_an_empty_block(self):
        assert guard.model_block(SCHEMA.replace("LiteLLM_SpendLogs", "LiteLLM_ErrorLogs")) == range(0)


class TestSpendLogHits:
    def test_a_field_added_inside_the_block_is_a_hit_without_naming_the_table(self):
        head: Final = _with_line(SCHEMA, 9, "  gauntlet_probe String?")
        diff: Final = _added("schema.prisma", "  gauntlet_probe String?")
        assert guard.spend_log_hits(diff, _blocks("schema.prisma", head=head)) == (
            "schema.prisma: +gauntlet_probe String?",
        )

    def test_a_field_removed_from_the_block_is_a_hit(self):
        head: Final = SCHEMA.replace("  api_key    String\n", "")
        diff: Final = _diff("schema.prisma", _hunk(11, 1, 10, 0, "-  api_key    String"))
        assert guard.spend_log_hits(diff, _blocks("schema.prisma", head=head)) == ("schema.prisma: -api_key    String",)

    def test_the_whole_model_appearing_or_disappearing_is_a_hit(self):
        without: Final = SCHEMA.replace(
            SCHEMA[SCHEMA.index("model LiteLLM_SpendLogs") : SCHEMA.index("model LiteLLM_UserTable")], ""
        )
        body: Final = tuple(SCHEMA.splitlines()[8:14])
        added: Final = _diff("schema.prisma", _hunk(8, 0, 9, 6, *(f"+{line}" for line in body)))
        removed: Final = _diff("schema.prisma", _hunk(9, 6, 8, 0, *(f"-{line}" for line in body)))
        assert len(guard.spend_log_hits(added, _blocks("schema.prisma", base=without))) == 5
        assert len(guard.spend_log_hits(removed, _blocks("schema.prisma", head=without))) == 5

    def test_a_line_naming_the_table_in_another_model_is_not_a_hit(self):
        head: Final = _with_line(SCHEMA, 6, "  spend_logs LiteLLM_SpendLogs[]")
        diff: Final = _diff(
            "schema.prisma",
            _hunk(6, 0, 7, 1, "+  spend_logs LiteLLM_SpendLogs[]"),
            _hunk(16, 1, 17, 0, "-  user_id String @id"),
        )
        assert guard.spend_log_hits(diff, _blocks("schema.prisma", head=head)) == ()

    def test_each_hunk_is_placed_by_its_own_header(self):
        head: Final = _with_line(_with_line(SCHEMA, 2, '  output   = "./client"'), 10, "  new_col String?")
        diff: Final = _diff(
            "schema.prisma",
            _hunk(2, 0, 3, 1, '+  output   = "./client"'),
            _hunk(9, 0, 11, 1, "+  new_col String?"),
        )
        assert guard.spend_log_hits(diff, _blocks("schema.prisma", head=head)) == ("schema.prisma: +new_col String?",)

    def test_an_added_migration_line_is_a_hit_in_any_case_and_a_removed_one_is_not(self):
        path: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/20260101000000_x/migration.sql"
        diff: Final = _diff(
            path,
            _hunk(
                3,
                1,
                3,
                1,
                '-DROP INDEX "LiteLLM_SpendLogs_old_idx";',
                '+CREATE INDEX "i" ON "litellm_spendlogs"("api_key");',
            ),
        )
        assert guard.spend_log_hits(diff, _NO_BLOCKS) == (
            f'{path}: +CREATE INDEX "i" ON "litellm_spendlogs"("api_key");',
        )

    def test_another_table_in_a_migration_or_an_unwatched_file_is_not_a_hit(self):
        path: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/20260101000000_x/migration.sql"
        other: Final = _added(
            path, 'ALTER TABLE "LiteLLM_TeamTable" ADD COLUMN "x" TEXT; -- not MyLiteLLM_SpendLogs', 5
        )
        elsewhere: Final = _added("litellm/proxy/utils.py", 'table = "LiteLLM_SpendLogs"', 5)
        assert guard.spend_log_hits(other + elsewhere, _NO_BLOCKS) == ()

    def test_a_table_whose_name_only_starts_with_the_watched_one_is_not_a_hit(self):
        index: Final = '  @@index([team_id], map: "LiteLLM_SpendLogs_team_id_idx")'
        head: Final = _with_line(_with_line(SCHEMA, 18, "model LiteLLM_SpendLogsArchive {"), 12, index)
        diff: Final = _diff(
            "schema.prisma",
            _hunk(12, 0, 13, 1, f"+{index}"),
            _hunk(18, 0, 20, 1, "+model LiteLLM_SpendLogsArchive {"),
        )
        assert guard.spend_log_hits(diff, _blocks("schema.prisma", head=head)) == (
            'schema.prisma: +@@index([team_id], map: "LiteLLM_SpendLogs_team_id_idx")',
        )

    def test_two_files_in_one_diff_are_attributed_to_their_own_path(self):
        head: Final = _with_line(SCHEMA, 9, "  a String")
        diff: Final = _added("litellm/proxy/schema.prisma", "  a String") + _added("schema.prisma", "  b String", 2)
        blocks: Final = _blocks("litellm/proxy/schema.prisma", head=head) | _blocks("schema.prisma")
        assert guard.spend_log_hits(diff, blocks) == ("litellm/proxy/schema.prisma: +a String",)


def _review(login: str, state: str, commit_id: str | None = "head") -> guard.Review:
    return guard.Review(state=state, user=guard.Reviewer(login=login), commit_id=commit_id)


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

    @pytest.mark.parametrize(
        ("states", "approved"),
        (
            (("APPROVED", "COMMENTED"), True),
            (("APPROVED", "PENDING"), True),
            (("CHANGES_REQUESTED", "COMMENTED"), False),
            (("APPROVED", "DISMISSED"), False),
            (("COMMENTED",), False),
        ),
    )
    def test_a_comment_or_pending_review_does_not_change_the_answer(self, states, approved):
        reviews: Final = tuple(guard.Review(state=state, user=guard.Reviewer(login="kerry-berri")) for state in states)
        assert guard.approved_by(reviews, "kerry-berri") is approved

    def test_no_review_from_the_approver_is_not_an_approval(self):
        other: Final = guard.Review(state="APPROVED", user=guard.Reviewer(login="someone-else"))
        assert guard.approved_by((other,), "kerry-berri") is False
        assert guard.approved_by((), "kerry-berri") is False


class TestLatestApproval:
    def test_the_approvers_last_approved_review_is_returned_with_its_commit(self):
        reviews: Final = (
            _review("kerry-berri", "APPROVED", "old"),
            _review("someone-else", "APPROVED", "head"),
            _review("kerry-berri", "APPROVED", "new"),
            _review("kerry-berri", "COMMENTED", "head"),
        )
        assert guard.latest_approval(reviews, "kerry-berri") is reviews[2]
        assert guard.latest_approval(reviews[:2], "kerry-berri") is reviews[0]
        assert guard.latest_approval(reviews[1:2], "kerry-berri") is None
        assert guard.latest_approval((), "kerry-berri") is None


class TestApprovalCovers:
    _HITS: Final = ("schema.prisma: a LiteLLM_SpendLogs", "schema.prisma: b LiteLLM_SpendLogs")

    def test_an_approval_on_the_head_covers_it_without_reading_the_approved_commit(self):
        def unreadable(commit: str) -> None:
            raise AssertionError(f"read {commit}")

        assert guard.approval_covers(_review("kerry-berri", "APPROVED", "head"), "head", self._HITS, unreadable) is True

    def test_an_older_approval_covers_the_head_only_when_it_saw_every_spend_log_line(self):
        approval: Final = _review("kerry-berri", "APPROVED", "old")
        assert guard.approval_covers(approval, "head", self._HITS, lambda commit: self._HITS) is True
        assert guard.approval_covers(approval, "head", self._HITS[:1], lambda commit: self._HITS) is True
        assert guard.approval_covers(approval, "head", self._HITS, lambda commit: self._HITS[:1]) is False

    def test_an_approval_whose_commit_cannot_be_read_does_not_cover_the_head(self):
        gone: Final = _review("kerry-berri", "APPROVED", "gone")
        assert guard.approval_covers(gone, "head", self._HITS, lambda commit: None) is False
        without_commit: Final = _review("kerry-berri", "APPROVED", None)
        assert guard.approval_covers(without_commit, "head", self._HITS, lambda commit: self._HITS) is False


class TestExitCode:
    def test_a_pull_request_without_spend_log_lines_passes_without_reviews(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(guard, "added_diff", lambda base, head: _added("schema.prisma", "  foo String", 6))
        monkeypatch.setattr(guard, "schema_blocks", lambda base, head: _blocks("schema.prisma"))
        reviews: Final = tmp_path / "r.json"
        reviews.write_text("[]")
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 0
        assert "nothing to review" in capsys.readouterr().out

    def test_a_spend_log_change_fails_until_the_approver_approves(self, tmp_path, monkeypatch, capsys):
        head: Final = _with_line(SCHEMA, 9, "  x String?")
        monkeypatch.setattr(guard, "added_diff", lambda base_sha, head_sha: _added("schema.prisma", "  x String?"))
        monkeypatch.setattr(guard, "schema_blocks", lambda base_sha, head_sha: _blocks("schema.prisma", head=head))
        reviews: Final = tmp_path / "r.json"
        reviews.write_text('[{"user": {"login": "someone-else"}, "state": "APPROVED"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1
        assert "schema.prisma: +x String?" in capsys.readouterr().out
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "b"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 0

    def test_an_approval_before_a_new_spend_log_line_was_pushed_fails(self, tmp_path, monkeypatch, capsys):
        head: Final = _with_line(SCHEMA, 9, "  x String?")
        monkeypatch.setattr(guard, "added_diff", lambda base_sha, head_sha: _added("schema.prisma", "  x String?"))
        monkeypatch.setattr(guard, "schema_blocks", lambda base_sha, head_sha: _blocks("schema.prisma", head=head))
        monkeypatch.setattr(guard, "reviewed_hits", lambda base_sha, commit: () if commit == "old" else None)
        reviews: Final = tmp_path / "r.json"
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "old"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1
        assert "changed after @kerry-berri approved old" in capsys.readouterr().out
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "gone"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1

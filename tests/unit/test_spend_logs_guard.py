import importlib.util
import sys
from pathlib import Path
from typing import Final

_SCRIPT: Final = Path(__file__).resolve().parents[2] / "ci_cd" / "spend_logs_guard.py"
_SPEC: Final = importlib.util.spec_from_file_location("spend_logs_guard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
guard: Final = importlib.util.module_from_spec(_SPEC)
sys.modules["spend_logs_guard"] = guard
_SPEC.loader.exec_module(guard)


def _diff(path: str, added: str, start: int = 10) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -{start},0 +{start},1 @@\n+{added}\n"


class TestSpendLogHits:
    def test_an_added_schema_line_naming_the_table_is_a_hit(self):
        diff: Final = _diff(
            "schema.prisma", '  @@index([api_key, startTime], map: "LiteLLM_SpendLogs_api_key_startTime_idx")', 740
        )
        assert guard.spend_log_hits(diff) == (
            'schema.prisma: @@index([api_key, startTime], map: "LiteLLM_SpendLogs_api_key_startTime_idx")',
        )

    def test_an_added_migration_line_is_a_hit_in_any_case(self):
        path: Final = "litellm-proxy-extras/litellm_proxy_extras/migrations/20260101000000_x/migration.sql"
        hits: Final = guard.spend_log_hits(_diff(path, 'CREATE INDEX "i" ON "litellm_spendlogs"("api_key");', 3))
        assert hits == (f'{path}: CREATE INDEX "i" ON "litellm_spendlogs"("api_key");',)

    def test_a_removed_line_or_another_table_or_an_unwatched_file_is_not_a_hit(self):
        removed: Final = "--- a/schema.prisma\n+++ b/schema.prisma\n@@ -1,1 +1,0 @@\n-model LiteLLM_SpendLogs {\n"
        other: Final = _diff("schema.prisma", "  team_id String  // LiteLLM_TeamTable, not MyLiteLLM_SpendLogs", 5)
        elsewhere: Final = _diff("litellm/proxy/utils.py", 'table = "LiteLLM_SpendLogs"', 5)
        assert guard.spend_log_hits(removed + other + elsewhere) == ()

    def test_two_files_in_one_diff_are_attributed_to_their_own_path(self):
        diff: Final = _diff("litellm/proxy/schema.prisma", "  a LiteLLM_SpendLogs", 1) + _diff(
            "schema.prisma", "  b String", 1
        )
        assert guard.spend_log_hits(diff) == ("litellm/proxy/schema.prisma: a LiteLLM_SpendLogs",)


def _review(login: str, state: str, commit_id: str | None = "head") -> "guard.Review":
    return guard.Review(state=state, user=guard.Reviewer(login=login), commit_id=commit_id)


class TestLatestApproval:
    def test_the_latest_review_of_the_approver_decides(self):
        reviews: Final = (
            _review("kerry-berri", "CHANGES_REQUESTED", "old"),
            _review("someone-else", "APPROVED"),
            _review("kerry-berri", "APPROVED"),
        )
        assert guard.latest_approval(reviews, "kerry-berri") is reviews[-1]
        assert guard.latest_approval(reviews[::-1], "kerry-berri") is None

    def test_no_review_from_the_approver_is_not_an_approval(self):
        assert guard.latest_approval((_review("someone-else", "APPROVED"),), "kerry-berri") is None
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
        monkeypatch.setattr(guard, "added_diff", lambda base, head: _diff("schema.prisma", "  foo String", 1))
        reviews: Final = tmp_path / "r.json"
        reviews.write_text("[]")
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 0
        assert "nothing to review" in capsys.readouterr().out

    def test_a_spend_log_change_fails_until_the_approver_approves(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(guard, "added_diff", lambda base, head: _diff("schema.prisma", "  x LiteLLM_SpendLogs", 7))
        reviews: Final = tmp_path / "r.json"
        reviews.write_text('[{"user": {"login": "someone-else"}, "state": "APPROVED"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1
        assert "schema.prisma: x LiteLLM_SpendLogs" in capsys.readouterr().out
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "b"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 0

    def test_an_approval_before_a_new_spend_log_line_was_pushed_fails(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(guard, "added_diff", lambda base, head: _diff("schema.prisma", "  x LiteLLM_SpendLogs", 7))
        monkeypatch.setattr(guard, "reviewed_hits", lambda base, commit: () if commit == "old" else None, raising=False)
        reviews: Final = tmp_path / "r.json"
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "old"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1
        assert "changed after @kerry-berri approved old" in capsys.readouterr().out
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED", "commit_id": "gone"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 1

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
        reviews.write_text('[{"user": {"login": "kerry-berri"}, "state": "APPROVED"}]')
        assert guard.main(["--base", "a", "--head", "b", "--approver", "kerry-berri", "--reviews", str(reviews)]) == 0

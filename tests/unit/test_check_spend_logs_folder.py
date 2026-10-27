import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

_SCRIPT: Final = (
    Path(__file__).resolve().parents[2] / "tests" / "code_coverage_tests" / "check_spend_logs_folder.py"
)
_SPEC: Final = importlib.util.spec_from_file_location("check_spend_logs_folder", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
checker: Final = importlib.util.module_from_spec(_SPEC)
sys.modules["check_spend_logs_folder"] = checker
_SPEC.loader.exec_module(checker)

MODEL: Final = '''model LiteLLM_SpendLogs {
  request_id String @id
  api_key    String @default("")

  @@index([api_key])
}
'''
SCHEMA: Final = f"model LiteLLM_Other {{\n  id String @id\n}}\n\n{MODEL}\nmodel LiteLLM_Later {{\n  id String @id\n}}\n"


def _migration(root: Path, name: str, sql: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "migration.sql").write_text(sql, encoding="utf-8")


class TestMisplacedMigrations:
    def test_a_shared_migration_that_mentions_the_table_is_reported(self, tmp_path):
        _migration(tmp_path, "20260101000000_idx", 'CREATE INDEX "x" ON "LiteLLM_SpendLogs"("api_key");')
        found: Final = checker.misplaced_migrations(tmp_path)
        assert len(found) == 1
        assert "20260101000000_idx/migration.sql" in found[0]
        assert "spend_logs/migrations/" in found[0]

    def test_the_match_ignores_case_but_not_longer_identifiers(self, tmp_path):
        _migration(tmp_path, "20260101000000_lower", 'ALTER TABLE "litellm_spendlogs" ADD COLUMN "a" TEXT;')
        _migration(tmp_path, "20260101000001_other", 'ALTER TABLE "LiteLLM_SpendLogsArchive" ADD COLUMN "a" TEXT;')
        found: Final = checker.misplaced_migrations(tmp_path)
        assert [v.split("/")[0] for v in found] == ["20260101000000_lower"]

    def test_a_shared_migration_on_another_table_passes(self, tmp_path):
        _migration(tmp_path, "20260101000000_other", 'ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "a" TEXT;')
        assert checker.misplaced_migrations(tmp_path) == ()


class TestSchemaCopy:
    def test_an_exact_copy_of_the_model_block_passes(self, tmp_path):
        (tmp_path / "schema.prisma").write_text(SCHEMA, encoding="utf-8")
        (tmp_path / "spend_logs").mkdir()
        (tmp_path / "spend_logs" / "LiteLLM_SpendLogs.prisma").write_text(MODEL, encoding="utf-8")
        assert checker.schema_copy_drift(tmp_path / "schema.prisma", tmp_path / "spend_logs") == ()

    @pytest.mark.parametrize("copy", [MODEL.replace("@@index([api_key])", "@@index([api_key, request_id])"), "", None])
    def test_a_drifted_or_missing_copy_is_reported(self, tmp_path, copy):
        (tmp_path / "schema.prisma").write_text(SCHEMA, encoding="utf-8")
        (tmp_path / "spend_logs").mkdir()
        if copy is not None:
            (tmp_path / "spend_logs" / "LiteLLM_SpendLogs.prisma").write_text(copy, encoding="utf-8")
        found: Final = checker.schema_copy_drift(tmp_path / "schema.prisma", tmp_path / "spend_logs")
        assert len(found) == 1
        assert "LiteLLM_SpendLogs.prisma" in found[0]

    def test_a_schema_without_the_model_is_reported(self, tmp_path):
        (tmp_path / "schema.prisma").write_text("model LiteLLM_Other {\n  id String @id\n}\n", encoding="utf-8")
        (tmp_path / "spend_logs").mkdir()
        found: Final = checker.schema_copy_drift(tmp_path / "schema.prisma", tmp_path / "spend_logs")
        assert len(found) == 1
        assert "no `model LiteLLM_SpendLogs` block" in found[0]


class TestExitCode:
    def test_the_checked_in_tree_passes(self, capsys):
        assert checker.main() == 0
        assert "Every LiteLLM_SpendLogs change lives in" in capsys.readouterr().out

    def test_a_violation_fails_and_is_printed(self, tmp_path, monkeypatch, capsys):
        _migration(tmp_path, "20260101000000_idx", 'CREATE INDEX "x" ON "LiteLLM_SpendLogs"("api_key");')
        monkeypatch.setattr(checker, "SHARED_MIGRATIONS", tmp_path)
        assert checker.main() == 1
        assert "20260101000000_idx/migration.sql" in capsys.readouterr().out

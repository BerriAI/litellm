"""Tests for scripts/detect_new_db_queries.py.

The detector's contract is "a PR that adds DB access is flagged": raw SQL and Prisma
model calls on added lines count, removed lines and tests/UI paths do not, and a
schema edit counts on its own. The regression that motivated it is #33978, whose
`prisma_client.db.query_raw` against LiteLLM_SpendLogs shipped without a perf review.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "detect_new_db_queries.py"
_spec = importlib.util.spec_from_file_location("detect_new_db_queries", _MODULE_PATH)
detector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detector)


def _changed(filename: str, patch: str | None):
    return detector.ChangedFile(filename=filename, patch=patch)


def test_flags_added_raw_sql_query():
    findings = detector.detect(
        (
            _changed(
                "litellm/proxy/management_endpoints/tool_management_endpoints.py",
                "@@ -1,0 +1,2 @@\n+    rows = await prisma_client.db.query_raw(\n+        'SELECT 1'\n",
            ),
        )
    )
    assert [f.detail for f in findings] == ["rows = await prisma_client.db.query_raw("]


def test_flags_added_prisma_model_call():
    findings = detector.detect(
        (
            _changed(
                "litellm/proxy/spend_tracking/spend_management_endpoints.py",
                '@@ -1,0 +1,1 @@\n+    rows = await prisma_client.db.litellm_spendlogs.find_many(where={"a": 1})\n',
            ),
        )
    )
    assert len(findings) == 1


def test_ignores_removed_and_context_lines():
    findings = detector.detect(
        (
            _changed(
                "litellm/proxy/db/db_spend_update_writer.py",
                "@@ -1,3 +1,1 @@\n-    await prisma_client.db.query_raw('SELECT 1')\n     await unrelated()\n",
            ),
        )
    )
    assert findings == ()


def test_ignores_tests_and_ui_paths():
    patch = "@@ -1,0 +1,1 @@\n+    await prisma_client.db.query_raw('SELECT 1')\n"
    findings = detector.detect(
        (
            _changed("tests/test_litellm/test_something.py", patch),
            _changed("ui/litellm-dashboard/src/thing.py", patch),
        )
    )
    assert findings == ()


def test_flags_schema_change_without_a_patch():
    findings = detector.detect((_changed("litellm/proxy/schema.prisma", None),))
    assert [f.detail for f in findings] == ["prisma schema changed"]


def test_ignores_unrelated_python_change():
    findings = detector.detect(
        (
            _changed(
                "litellm/main.py",
                "@@ -1,0 +1,1 @@\n+    response = await client.chat.completions.create(**kwargs)\n",
            ),
            _changed("README.md", "@@ -1,0 +1,1 @@\n+docs\n"),
        )
    )
    assert findings == ()


def test_main_exit_codes_signal_whether_anything_was_found(monkeypatch, capsys, tmp_path):
    flagged = tmp_path / "flagged.json"
    flagged.write_text(
        '[{"filename": "litellm/proxy/db/x.py",'
        ' "patch": "@@ -1,0 +1,1 @@\\n+    await prisma_client.db.query_raw(sql)\\n"}]'
    )
    with flagged.open() as handle:
        monkeypatch.setattr(detector.sys, "stdin", handle)
        assert detector.main() == 0
    assert "litellm/proxy/db/x.py" in capsys.readouterr().out

    clean = tmp_path / "clean.json"
    clean.write_text('[{"filename": "README.md", "patch": "@@ -1,0 +1,1 @@\\n+docs\\n"}]')
    with clean.open() as handle:
        monkeypatch.setattr(detector.sys, "stdin", handle)
        assert detector.main() == 1

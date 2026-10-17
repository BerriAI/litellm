from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "risk_tier.py"
CONFIG_PATH = REPO_ROOT / ".github" / "risk-tiers.yml"
DEVIN = "devin-ai-integration[bot]"


@pytest.fixture(scope="module")
def risk_tier():
    spec = importlib.util.spec_from_file_location("risk_tier", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["risk_tier"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rules(risk_tier):
    return risk_tier.Rules.from_config(risk_tier.load_config(CONFIG_PATH))


def _file_diff(path: str, added: Sequence[str] = (), deleted: Sequence[str] = ()) -> str:
    header = (
        f"diff --git a/{path} b/{path}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,{len(deleted)} +1,{len(added)} @@\n"
    )
    body = "".join(f"-{line}\n" for line in deleted) + "".join(f"+{line}\n" for line in added)
    return header + body


def _lines(count: int, prefix: str = "x = ") -> tuple[str, ...]:
    return tuple(f"{prefix}{index}" for index in range(count))


def _rename_diff(old: str, new: str, deleted: Sequence[str] = (), added: Sequence[str] = ()) -> str:
    header = f"diff --git a/{old} b/{new}\nsimilarity index 90%\nrename from {old}\nrename to {new}\n"
    if not deleted and not added:
        return header.replace("90%", "100%")
    hunk = f"index 0000000..1111111 100644\n--- a/{old}\n+++ b/{new}\n@@ -1,{len(deleted)} +1,{len(added)} @@\n"
    body = "".join(f"-{line}\n" for line in deleted) + "".join(f"+{line}\n" for line in added)
    return header + hunk + body


NEW_TEST = ("def test_regression():", "    assert True")


def _factor(verdict, name: str):
    return next(factor for factor in verdict.factors if factor.name == name)


def _verdict(risk_tier, rules, diff: str, author: str = DEVIN, from_fork: bool = False):
    return risk_tier.classify(risk_tier.parse_diff(diff), author, from_fork, rules)


def test_always_human_path_is_high_even_when_tiny_and_tested(risk_tier, rules):
    diff = _file_diff("litellm/proxy/auth/user_api_key_auth.py", added=("x = 1",)) + _file_diff(
        "tests/test_litellm/proxy/auth/test_user_api_key_auth.py", added=NEW_TEST
    )
    verdict = _verdict(risk_tier, rules, diff)
    assert verdict.tier == "high"
    assert _factor(verdict, "paths").tier == "high"
    assert "litellm/proxy/auth/user_api_key_auth.py" in _factor(verdict, "paths").reason
    assert _factor(verdict, "size").tier == "low"
    assert _factor(verdict, "tests").tier == "low"


def test_docs_and_tests_only_change_is_low_on_every_factor(risk_tier, rules):
    diff = _file_diff("README.md", added=("hello",)) + _file_diff("tests/test_litellm/test_docs.py", added=NEW_TEST)
    verdict = _verdict(risk_tier, rules, diff)
    assert verdict.tier == "low"
    assert {factor.tier for factor in verdict.factors} == {"low"}


def test_provider_change_with_regression_test_is_medium_by_path_only(risk_tier, rules):
    diff = _file_diff("litellm/llms/anthropic/chat/transformation.py", added=_lines(20)) + _file_diff(
        "tests/test_litellm/llms/anthropic/test_chat_transformation.py", added=NEW_TEST
    )
    verdict = _verdict(risk_tier, rules, diff)
    assert verdict.tier == "medium"
    assert {factor.name: factor.tier for factor in verdict.factors} == {
        "paths": "medium",
        "modules": "low",
        "size": "low",
        "tests": "low",
        "author": "low",
    }


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (("litellm/llms/anthropic/chat/x.py",), "low"),
        (("litellm/llms/anthropic/chat/x.py", "litellm/types/llms/anthropic.py"), "medium"),
        (("litellm/llms/anthropic/chat/x.py", "litellm/llms/openai/chat/x.py", "litellm/types/llms/a.py"), "medium"),
        (
            (
                "litellm/llms/anthropic/chat/x.py",
                "litellm/llms/openai/chat/x.py",
                "litellm/types/llms/a.py",
                "ui/litellm-dashboard/src/a.tsx",
            ),
            "high",
        ),
    ],
)
def test_modules_factor_counts_distinct_production_modules(risk_tier, rules, paths, expected):
    diff = "".join(_file_diff(path, added=("x = 1",)) for path in paths)
    diff_with_tests = diff + _file_diff("tests/test_litellm/test_a.py", added=NEW_TEST)
    assert _factor(_verdict(risk_tier, rules, diff_with_tests), "modules").tier == expected


def test_same_module_touched_twice_counts_once(risk_tier, rules):
    diff = _file_diff("litellm/llms/anthropic/chat/a.py", added=("x = 1",)) + _file_diff(
        "litellm/llms/anthropic/common_utils.py", added=("x = 1",)
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "modules")
    assert factor.tier == "low"
    assert factor.reason == "1 module(s): `litellm/llms/anthropic`"


@pytest.mark.parametrize(
    ("line_counts", "expected"),
    [
        ((33, 33, 33), "low"),
        ((34, 33, 33), "medium"),
        ((1, 1, 1, 1), "medium"),
        ((100, 100, 100, 99), "medium"),
        ((100, 100, 100, 100), "high"),
        ((1,) * 11, "high"),
    ],
)
def test_size_factor_thresholds(risk_tier, rules, line_counts, expected):
    diff = "".join(
        _file_diff(f"litellm/llms/provider{index}/chat/x.py", added=_lines(count))
        for index, count in enumerate(line_counts)
    )
    assert _factor(_verdict(risk_tier, rules, diff), "size").tier == expected


def test_generated_files_do_not_count_toward_size(risk_tier, rules):
    diff = _file_diff("ui/litellm-dashboard/src/lib/http/schema.d.ts", added=_lines(5000)) + _file_diff(
        "litellm/llms/anthropic/chat/x.py", added=("x = 1",)
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "size")
    assert factor.tier == "low"
    assert factor.reason == "1 line(s) across 1 file(s) outside the docs, tests, and model map tiers"


def test_docs_and_tests_do_not_count_toward_size(risk_tier, rules):
    diff = (
        _file_diff("docs/my-website/docs/proxy/guide.md", added=_lines(5000, prefix="line "))
        + _file_diff("tests/test_litellm/test_x.py", added=_lines(500) + NEW_TEST)
        + _file_diff("litellm/llms/anthropic/chat/x.py", added=("x = 1",))
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "size")
    assert factor.tier == "low"
    assert factor.reason == "1 line(s) across 1 file(s) outside the docs, tests, and model map tiers"


def test_pure_move_counts_no_lines_toward_size(risk_tier, rules):
    diff = _rename_diff("litellm/llms/anthropic/chat/old.py", "litellm/llms/anthropic/chat/new.py")
    factor = _factor(_verdict(risk_tier, rules, diff), "size")
    assert factor.tier == "low"
    assert factor.reason == "0 line(s) across 1 file(s) outside the docs, tests, and model map tiers"


def test_move_with_edits_counts_only_the_edited_lines(risk_tier, rules):
    diff = _rename_diff(
        "litellm/llms/anthropic/chat/old.py",
        "litellm/llms/anthropic/chat/new.py",
        deleted=_lines(2),
        added=_lines(2, prefix="y = "),
    )
    assert _factor(_verdict(risk_tier, rules, diff), "size").reason.startswith("4 line(s) across 1 file(s)")


def test_move_out_of_an_always_human_path_stays_high(risk_tier, rules):
    diff = _rename_diff("litellm/proxy/auth/old.py", "litellm/proxy/common_utils/old.py")
    factor = _factor(_verdict(risk_tier, rules, diff), "paths")
    assert factor.tier == "high"
    assert factor.reason == "always-human: `litellm/proxy/auth/old.py`"


def test_move_into_an_always_human_path_is_high(risk_tier, rules):
    diff = _rename_diff("litellm/proxy/utils/new.py", "litellm/proxy/auth/new.py")
    factor = _factor(_verdict(risk_tier, rules, diff), "paths")
    assert factor.tier == "high"
    assert factor.reason == "always-human: `litellm/proxy/auth/new.py`"


def test_removed_test_function_is_high(risk_tier, rules):
    diff = _file_diff("tests/test_litellm/test_a.py", deleted=("def test_gone():", "    assert 1 == 1"))
    verdict = _verdict(risk_tier, rules, diff)
    assert verdict.tier == "high"
    assert _factor(verdict, "tests").reason == "1 test(s) removed"


@pytest.mark.parametrize(
    "marker",
    [
        '@pytest.mark.skip(reason="flaky")',
        "@pytest.mark.skip",
        'pytestmark = pytest.mark.skip("whole module is flaky")',
        '@pytest.mark.xfail(reason="broken since the refactor")',
        '@unittest.skip("flaky")',
    ],
)
def test_unconditional_skip_marker_is_high(risk_tier, rules, marker):
    diff = _file_diff("tests/test_litellm/test_a.py", added=(marker, *NEW_TEST))
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "high"
    assert factor.reason == "1 skip marker(s) added"


@pytest.mark.parametrize(
    "guard",
    [
        '@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs a real key")',
        '    if not os.getenv("OPENAI_API_KEY"):',
        '        pytest.skip("needs a real key")',
        '@unittest.skipIf(sys.platform == "win32", "posix only")',
        '@unittest.skipUnless(HAS_REDIS, "needs redis")',
        'redis = pytest.importorskip("redis")',
    ],
)
def test_conditional_skip_is_not_a_silenced_test(risk_tier, rules, guard):
    diff = _file_diff("tests/test_litellm/test_a.py", added=(guard, *NEW_TEST)) + _file_diff(
        "litellm/llms/anthropic/chat/x.py", added=("x = 1",)
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "low"
    assert factor.reason == "1 test(s) added"


@pytest.mark.parametrize(
    ("path", "line"),
    [
        ("tests/test_litellm/test_a.py", '    pytest.skip("flaky since the refactor, see LIT-0000")'),
        ("tests/test_litellm/test_a.py", 'redis = pytest.importorskip("redis")'),
        ("ui/litellm-dashboard/src/app/login/LoginPage.test.tsx", "  test.fixme(true, 'broken after the redesign');"),
    ],
)
def test_skip_call_added_to_an_existing_test_is_high_even_without_production_code(risk_tier, rules, path, line):
    verdict = _verdict(risk_tier, rules, _file_diff(path, added=(line,)))
    assert verdict.tier == "high"
    assert _factor(verdict, "tests").reason == "1 skip call(s) added to existing tests"


def test_removing_a_skip_call_is_not_silencing(risk_tier, rules):
    diff = _file_diff("tests/test_litellm/test_a.py", deleted=('    pytest.skip("flaky")',)) + _file_diff(
        "litellm/llms/anthropic/chat/x.py", added=("x = 1",)
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "medium"
    assert factor.reason == "tests edited, none added"


def test_weakened_assertions_are_high(risk_tier, rules):
    diff = _file_diff(
        "tests/test_litellm/test_a.py",
        added=("    assert result",),
        deleted=("    assert result.status == 200", "    assert result.body == expected"),
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "high"
    assert factor.reason == "1 assertion(s) removed"


def test_renamed_test_file_is_not_a_deletion(risk_tier, rules):
    body = ("def test_kept():", "    assert kept()")
    diff = (
        _file_diff("tests/test_litellm/test_old.py", deleted=body)
        + _file_diff("tests/test_litellm/test_new.py", added=body)
        + _file_diff("litellm/llms/anthropic/chat/x.py", added=("x = 1",))
    )
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "medium"
    assert factor.reason == "tests edited, none added"


def test_production_change_without_any_test_is_medium(risk_tier, rules):
    diff = _file_diff("litellm/llms/anthropic/chat/x.py", added=("x = 1",))
    factor = _factor(_verdict(risk_tier, rules, diff), "tests")
    assert factor.tier == "medium"
    assert factor.reason == "production code changed with no test touched"


@pytest.mark.parametrize(
    ("added", "expected"),
    [
        (('it("hides the notice", () => {', "  expect(screen.queryByText(notice)).toBeNull();", "});"), "low"),
        (('it.skip("hides the notice", () => {', "});"), "high"),
        (("test.skip('hides the notice', async () => {", "});"), "high"),
        (('it.only("hides the notice", () => {', "});"), "high"),
        (('describe.only("login", () => {', "});"), "high"),
        (('xit("hides the notice", () => {', "});"), "high"),
        (
            (
                'test("hides the notice", async ({ page }) => {',
                "  test.skip(!process.env.UI_BASE_URL, 'needs a UI');",
                "});",
            ),
            "low",
        ),
        (
            ('test("hides the notice", async ({ page }) => {', "  if (!process.env.UI_BASE_URL) test.skip();", "});"),
            "low",
        ),
    ],
)
def test_typescript_tests_count_like_python_ones(risk_tier, rules, added, expected):
    diff = _file_diff("ui/litellm-dashboard/src/app/login/LoginPage.test.tsx", added=added) + _file_diff(
        "ui/litellm-dashboard/src/app/login/LoginPage.tsx", added=("const x = 1;",)
    )
    assert _factor(_verdict(risk_tier, rules, diff), "tests").tier == expected


@pytest.mark.parametrize(
    ("author", "from_fork", "expected"),
    [
        (DEVIN, False, "low"),
        ("mateo-berri", False, "medium"),
        (DEVIN, True, "high"),
        ("jairandresdiazp", True, "high"),
    ],
)
def test_author_factor(risk_tier, rules, author, from_fork, expected):
    diff = _file_diff("README.md", added=("hello",))
    verdict = _verdict(risk_tier, rules, diff, author=author, from_fork=from_fork)
    assert _factor(verdict, "author").tier == expected
    assert verdict.tier == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("litellm/proxy/auth/user_api_key_auth.py", "high"),
        ("litellm/proxy/schema.prisma", "high"),
        ("litellm-proxy-extras/litellm_proxy_extras/migrations/20260901_x/migration.sql", "high"),
        ("docker/Dockerfile.database", "high"),
        ("ui/litellm-dashboard/package.json", "high"),
        ("scripts/type_check_gate.py", "high"),
        (".github/risk-tiers.yml", "high"),
        ("enterprise/pyproject.toml", "high"),
        ("litellm-proxy-extras/pyproject.toml", "high"),
        ("litellm/proxy/_experimental/mcp_server/auth/user_api_key_auth_mcp.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/outbound_credentials/x.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/discoverable_endpoints.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/oauth2_token_cache.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/byok_oauth_endpoints.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/bridge_token_flow.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/proxy_api_credentials.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/faults/render_oauth.py", "high"),
        ("litellm/proxy/_experimental/mcp_server/faults/classify.py", "medium"),
        ("litellm/proxy/_experimental/mcp_server/server.py", "medium"),
        ("enterprise/litellm_enterprise/proxy/hooks/x.py", "high"),
        ("enterprise/litellm_enterprise/proxy/auth/x.py", "high"),
        ("enterprise/litellm_enterprise/proxy/management_endpoints/x.py", "high"),
        ("litellm/proxy/pass_through_endpoints/x.py", "high"),
        ("litellm/llms/bedrock/passthrough/x.py", "medium"),
        ("ui/litellm-dashboard/src/components/networking.tsx", "medium"),
        ("litellm/types/proxy/x.py", "medium"),
        ("ui/litellm-dashboard/src/app/login/LoginPage.test.tsx", "low"),
        ("tests/e2e/x.py", "low"),
        ("litellm-proxy-extras/tests/test_x.py", "low"),
        ("helm/litellm-helm/tests/x.yaml", "medium"),
        ("helm/litellm-helm/templates/tests/test-connection.yaml", "medium"),
        ("docker/tests/nonroot.yaml", "medium"),
        ("docker/entrypoint.sh", "medium"),
        ("CLAUDE.md", "medium"),
        ("AGENTS.md", "medium"),
        ("GEMINI.md", "medium"),
        ("litellm/proxy/_experimental/mcp_server/CLAUDE.md", "medium"),
        ("tests/conftest.py", "medium"),
        ("tests/test_litellm/conftest.py", "medium"),
        ("tests/e2e/junit_properties.py", "low"),
        ("ui/litellm-dashboard/tests/x.spec.ts", "low"),
        ("litellm-rust/crates/core/tests/x.rs", "low"),
        ("enterprise/litellm_enterprise/proxy/common_utils/x.py", "medium"),
        ("cookbook/x.ipynb", "low"),
        ("model_prices_and_context_window.json", "low"),
        ("litellm/model_prices_and_context_window_backup.json", "low"),
        ("litellm/proxy/README.md", "low"),
        ("type-discipline-budget.json", "low"),
        ("ruff-strict-budget.json", "low"),
    ],
)
def test_path_tier_from_the_checked_in_config(rules, path, expected):
    assert rules.path_tier(path) == expected


def test_human_opened_docs_change_is_medium_on_the_author_factor_only(risk_tier, rules):
    verdict = _verdict(risk_tier, rules, _file_diff("README.md", added=("hello",)), author="mateo-berri")
    assert verdict.tier == "medium"
    assert {factor.name: factor.tier for factor in verdict.factors} == {
        "paths": "low",
        "modules": "low",
        "size": "low",
        "tests": "low",
        "author": "medium",
    }


MODEL_MAP = "model_prices_and_context_window.json"
BUDGET = "ruff-strict-budget.json"


def _guarded_verdict(risk_tier, rules, path: str, base: str | None, head: str | None):
    contents = {("base", path): base, ("head", path): head}
    diff = _file_diff(path, added=("changed",))
    changes = risk_tier.with_guards(risk_tier.parse_diff(diff), rules, "base", "head", lambda rev, p: contents[(rev, p)])
    return risk_tier.classify(changes, DEVIN, False, rules)


def test_additive_model_map_rows_stay_low(risk_tier, rules):
    base = json.dumps({"gpt-x": {"input_cost_per_token": 1e-6}})
    head = json.dumps({"gpt-x": {"input_cost_per_token": 1e-6}, "gpt-y": {"input_cost_per_token": 2e-6}})
    verdict = _guarded_verdict(risk_tier, rules, MODEL_MAP, base, head)
    assert verdict.tier == "low"
    assert _factor(verdict, "paths").reason == "docs, tests, cookbook, or model map only"


@pytest.mark.parametrize(
    "head",
    [
        {"gpt-x": {"input_cost_per_token": 3e-6}},
        {"gpt-x-renamed": {"input_cost_per_token": 1e-6}},
        {},
    ],
)
def test_changed_or_removed_model_map_rows_are_medium(risk_tier, rules, head):
    base = json.dumps({"gpt-x": {"input_cost_per_token": 1e-6}})
    verdict = _guarded_verdict(risk_tier, rules, MODEL_MAP, base, json.dumps(head))
    assert verdict.tier == "medium"
    assert _factor(verdict, "paths").tier == "medium"
    assert _factor(verdict, "paths").reason == f"1 existing row(s) changed or removed in `{MODEL_MAP}`"


def test_new_model_map_file_or_broken_json_is_never_low(risk_tier, rules):
    assert _guarded_verdict(risk_tier, rules, MODEL_MAP, None, "{}").tier == "low"
    assert _factor(_guarded_verdict(risk_tier, rules, MODEL_MAP, "{}", "not json"), "paths").tier == "medium"
    assert _factor(_guarded_verdict(risk_tier, rules, MODEL_MAP, "{}", None), "paths").tier == "medium"


def test_lowered_budget_limits_stay_low_and_raised_ones_are_medium(risk_tier, rules):
    base = json.dumps({"ANN001": {"limit": 10}, "B006": {"limit": 5}})
    lowered = json.dumps({"ANN001": {"limit": 9}, "B006": {"limit": 5}})
    raised = json.dumps({"ANN001": {"limit": 10}, "B006": {"limit": 6}})
    new_rule = json.dumps({"ANN001": {"limit": 10}, "B006": {"limit": 5}, "B999": {"limit": 1}})
    dropped_rule = json.dumps({"ANN001": {"limit": 10}})
    assert _guarded_verdict(risk_tier, rules, BUDGET, base, lowered).tier == "low"
    assert _guarded_verdict(risk_tier, rules, BUDGET, base, dropped_rule).tier == "low"
    for head in (raised, new_rule):
        verdict = _guarded_verdict(risk_tier, rules, BUDGET, base, head)
        assert verdict.tier == "medium"
        assert _factor(verdict, "paths").reason == f"1 limit(s) raised in `{BUDGET}`"


def test_single_star_does_not_cross_directories(risk_tier):
    assert risk_tier.glob_to_regex("Dockerfile*").fullmatch("Dockerfile.database")
    assert risk_tier.glob_to_regex("Dockerfile*").fullmatch("docker/Dockerfile.database") is None
    assert risk_tier.glob_to_regex("scripts/*gate*").fullmatch("scripts/nested/type_check_gate.py") is None
    assert risk_tier.glob_to_regex("**/migrations/**").fullmatch("migrations/x.sql")


def test_parse_diff_handles_binary_deleted_and_dash_prefixed_lines(risk_tier):
    diff = (
        "diff --git a/img.png b/img.png\n"
        "index 0000000..1111111 100644\n"
        "Binary files a/img.png and b/img.png differ\n"
        "diff --git a/gone.sql b/gone.sql\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.sql\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "--- a sql comment\n"
        "-SELECT 1;\n"
    )
    changes = risk_tier.parse_diff(diff)
    assert [change.path for change in changes] == ["img.png", "gone.sql"]
    assert changes[0].line_count == 0
    assert changes[1].deleted_lines == ("-- a sql comment", "SELECT 1;")
    assert changes[1].added_lines == ()


def test_parse_diff_of_an_empty_diff_is_empty(risk_tier):
    assert risk_tier.parse_diff("") == ()


def test_parse_diff_reads_a_rename_as_one_change_with_both_paths(risk_tier):
    diff = _rename_diff("litellm/a/very_long_old_name.py", "litellm/b/new.py", deleted=("x = 1",), added=("x = 2",))
    changes = risk_tier.parse_diff(diff)
    assert len(changes) == 1
    assert changes[0].path == "litellm/b/new.py"
    assert changes[0].previous_path == "litellm/a/very_long_old_name.py"
    assert changes[0].paths == ("litellm/a/very_long_old_name.py", "litellm/b/new.py")
    assert changes[0].line_count == 2


def test_parse_diff_unquotes_renamed_paths(risk_tier):
    diff = (
        'diff --git "a/docs/we\\"ird.md" b/docs/plain.md\n'
        "similarity index 100%\n"
        'rename from "docs/we\\"ird.md"\n'
        "rename to docs/plain.md\n"
    )
    changes = risk_tier.parse_diff(diff)
    assert changes[0].paths == ('docs/we"ird.md', "docs/plain.md")


def test_parse_diff_decodes_git_quoted_paths_instead_of_dropping_them(risk_tier):
    diff = (
        "diff --git a/docs/plain.md b/docs/plain.md\n"
        "--- a/docs/plain.md\n"
        "+++ b/docs/plain.md\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
        'diff --git "a/litellm/proxy/auth/we\\"ird\\ttab\\\\slash\\001.py" "b/litellm/proxy/auth/we\\"ird\\ttab\\\\slash\\001.py"\n'
        "new file mode 100644\n"
        '--- "a/litellm/proxy/auth/we\\"ird\\ttab\\\\slash\\001.py"\n'
        '+++ "b/litellm/proxy/auth/we\\"ird\\ttab\\\\slash\\001.py"\n'
        "@@ -0,0 +1,2 @@\n"
        "+def f():\n"
        "+    return 1\n"
    )
    changes = risk_tier.parse_diff(diff)
    assert [change.path for change in changes] == ["docs/plain.md", 'litellm/proxy/auth/we"ird\ttab\\slash\x01.py']
    assert [change.line_count for change in changes] == [1, 2]


def test_config_rejects_unknown_keys(risk_tier, tmp_path):
    bad = tmp_path / "risk-tiers.yml"
    bad.write_text(CONFIG_PATH.read_text() + "\nprompt: judge.md\n")
    with pytest.raises(ValidationError, match="Extra inputs"):
        risk_tier.load_config(bad)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@x",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@x",
        },
    ).stdout.strip()


def test_main_end_to_end_against_a_git_repo(risk_tier, tmp_path, capsys):
    repo = tmp_path / "repo"
    (repo / "litellm" / "caching").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", "repo")
    (repo / "litellm" / "caching" / "caching.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "litellm" / "caching" / "caching.py").write_text("x = 2\n")
    _git(repo, "commit", "-q", "-am", "head")
    head = _git(repo, "rev-parse", "HEAD")
    json_out = tmp_path / "risk.json"

    exit_code = risk_tier.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--base",
            base,
            "--head",
            head,
            "--author",
            DEVIN,
            "--repo",
            str(repo),
            "--json-out",
            str(json_out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_out.read_text())
    assert payload["tier"] == "high"
    assert {factor["name"]: factor["tier"] for factor in payload["factors"]} == {
        "paths": "high",
        "modules": "low",
        "size": "low",
        "tests": "medium",
        "author": "low",
    }
    printed = capsys.readouterr().out
    assert printed.startswith("risk: high (shadow mode, nothing is blocked)")
    assert payload["summary"] == printed


def test_main_reads_both_sides_of_the_model_map_from_git(risk_tier, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", "repo")
    model_map = repo / "model_prices_and_context_window.json"
    model_map.write_text(json.dumps({"gpt-x": {"input_cost_per_token": 1e-6}}, indent=1))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    model_map.write_text(json.dumps({"gpt-x": {"input_cost_per_token": 2e-6}}, indent=1))
    _git(repo, "commit", "-q", "-am", "reprice")
    head = _git(repo, "rev-parse", "HEAD")
    json_out = tmp_path / "risk.json"

    exit_code = risk_tier.main(
        ["--config", str(CONFIG_PATH), "--base", base, "--head", head, "--author", DEVIN, "--repo", str(repo), "--json-out", str(json_out)]
    )

    assert exit_code == 0
    payload = json.loads(json_out.read_text())
    assert payload["tier"] == "medium"
    paths = next(factor for factor in payload["factors"] if factor["name"] == "paths")
    assert paths["reason"] == "1 existing row(s) changed or removed in `model_prices_and_context_window.json`"


def test_git_quoted_filename_still_reaches_the_paths_factor(risk_tier, rules, tmp_path):
    repo = tmp_path / "repo"
    (repo / "litellm" / "proxy" / "auth").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", "repo")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    quoted_name = 'we"ird\ttab\\slash.py'
    (repo / "litellm" / "proxy" / "auth" / quoted_name).write_text("def f():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    head = _git(repo, "rev-parse", "HEAD")

    changes = risk_tier.parse_diff(risk_tier.git_diff(repo, base, head))

    assert [change.path for change in changes] == [f"litellm/proxy/auth/{quoted_name}"]
    assert changes[0].added_lines == ("def f():", "    return 1")
    verdict = risk_tier.classify(changes, DEVIN, False, rules)
    assert _factor(verdict, "paths").tier == "high"


def test_git_pure_move_is_one_zero_line_change(risk_tier, rules, tmp_path):
    repo = tmp_path / "repo"
    (repo / "litellm" / "proxy" / "auth").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", "repo")
    (repo / "litellm" / "proxy" / "auth" / "checks.py").write_text("".join(f"x{i} = {i}\n" for i in range(300)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "litellm" / "proxy" / "utils").mkdir()
    _git(repo, "mv", "litellm/proxy/auth/checks.py", "litellm/proxy/utils/checks.py")
    _git(repo, "commit", "-q", "-m", "move")
    head = _git(repo, "rev-parse", "HEAD")

    changes = risk_tier.parse_diff(risk_tier.git_diff(repo, base, head))

    assert [change.paths for change in changes] == [("litellm/proxy/auth/checks.py", "litellm/proxy/utils/checks.py")]
    assert changes[0].line_count == 0
    verdict = risk_tier.classify(changes, DEVIN, False, rules)
    assert _factor(verdict, "paths").tier == "high"
    assert _factor(verdict, "size").tier == "low"

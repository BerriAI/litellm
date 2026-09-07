from __future__ import annotations

import importlib.util
import json
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
    assert factor.reason == "1 line(s) across 1 file(s)"


def test_removed_test_function_is_high(risk_tier, rules):
    diff = _file_diff("tests/test_litellm/test_a.py", deleted=("def test_gone():", "    assert 1 == 1"))
    verdict = _verdict(risk_tier, rules, diff)
    assert verdict.tier == "high"
    assert _factor(verdict, "tests").reason == "1 test(s) removed"


def test_added_skip_marker_is_high(risk_tier, rules):
    diff = _file_diff("tests/test_litellm/test_a.py", added=('@pytest.mark.skip(reason="flaky")',))
    assert _factor(_verdict(risk_tier, rules, diff), "tests").tier == "high"


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
        ("enterprise/litellm_enterprise/proxy/hooks/x.py", "high"),
        ("litellm/proxy/pass_through_endpoints/x.py", "high"),
        ("litellm/llms/bedrock/passthrough/x.py", "medium"),
        ("ui/litellm-dashboard/src/components/networking.tsx", "medium"),
        ("litellm/types/proxy/x.py", "medium"),
        ("ui/litellm-dashboard/src/app/login/LoginPage.test.tsx", "low"),
        ("tests/e2e/x.py", "low"),
        ("cookbook/x.ipynb", "low"),
        ("model_prices_and_context_window.json", "low"),
        ("litellm/model_prices_and_context_window_backup.json", "low"),
        ("litellm/proxy/README.md", "low"),
    ],
)
def test_path_tier_from_the_checked_in_config(rules, path, expected):
    assert rules.path_tier(path) == expected


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

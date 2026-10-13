import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]
RULES_PATH: Final = ROOT / ".github" / "stampbot" / "risk_rules.yml"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "ci_cd" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bot: Final = _load("stampbot")
RULES: Final = bot.load_rules(RULES_PATH)
REPO: Final = "BerriAI/litellm"


def _pr(
    *,
    additions: int = 10,
    deletions: int = 2,
    draft: bool = False,
    head_repo: str = REPO,
    state: str = "open",
) -> object:
    return bot.PullRequest.model_validate(
        {
            "number": 1,
            "title": "fix(openai): map a param",
            "body": "body",
            "draft": draft,
            "state": state,
            "user": {"login": "someone", "type": "User"},
            "head": {"sha": "a" * 40, "ref": "litellm_x", "repo": {"full_name": head_repo}},
            "base": {"sha": "b" * 40, "ref": "main", "repo": {"full_name": REPO}},
            "additions": additions,
            "deletions": deletions,
            "changed_files": 1,
        }
    )


def _file(path: str, status: str = "modified", patch: str | None = None) -> object:
    return bot.ChangedFile(filename=path, status=status, additions=1, deletions=1, patch=patch)


def _check(name: str, conclusion: str | None, status: str = "completed") -> object:
    return bot.CheckRun(name=name, status=status, conclusion=conclusion)


def _review(login: str, state: str) -> object:
    return bot.Review(id=1, user=bot.User(login=login), state=state)


def _findings(
    pr: object,
    files: tuple[object, ...],
    checks: tuple[object, ...] = (),
    statuses: tuple[object, ...] = (),
    reviews: tuple[object, ...] = (),
    unresolved: int = 0,
) -> tuple[object, ...]:
    return bot.compute_findings(pr, REPO, files, checks, statuses, reviews, unresolved, RULES)


def _floor(findings: tuple[object, ...]) -> object:
    return max((f.tier for f in findings), default=bot.Tier.low)


CODE_WITH_TEST: Final = (_file("litellm/llms/openai/chat/transformation.py"), _file("tests/unit/llms/openai/test_x.py"))


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("litellm/proxy/auth/**", "litellm/proxy/auth/user_api_key_auth.py", True),
        ("litellm/proxy/auth/**", "litellm/proxy/auth/deep/er/file.py", True),
        ("litellm/proxy/auth/**", "litellm/proxy/authx/file.py", False),
        ("**/migrations/**", "litellm/proxy/db/migrations/0001/migration.sql", True),
        ("**/migrations/**", "migrations/0001/migration.sql", True),
        ("**/*.md", "README.md", True),
        ("**/*.md", "docs/my_website/docs/x.md", True),
        ("**/*.md", "docs/x.mdx", False),
        ("litellm/router.py", "litellm/router.py", True),
        ("litellm/router.py", "litellm/router_utils/x.py", False),
        ("**/Dockerfile*", "docker/Dockerfile.database", True),
        ("ui/litellm-dashboard/**/*.test.tsx", "ui/litellm-dashboard/src/components/a/b.test.tsx", True),
    ],
)
def test_glob_matching(pattern: str, path: str, expected: bool) -> None:
    assert bot.matches_any(path, (pattern,)) is expected


def test_rules_file_covers_the_sensitive_surfaces() -> None:
    for path in (
        "litellm/proxy/auth/user_api_key_auth.py",
        "litellm/proxy/management_endpoints/team_endpoints.py",
        "litellm/proxy/spend_tracking/spend_tracking_utils.py",
        "litellm/proxy/db/prisma_client.py",
        "litellm/proxy/proxy_server.py",
        "litellm/router.py",
        "litellm/caching/redis_cache.py",
        "schema.prisma",
        ".github/workflows/stampbot.yml",
        ".github/stampbot/risk_rules.yml",
        "ci_cd/stampbot.py",
        "pyproject.toml",
        "uv.lock",
        "enterprise/litellm_enterprise/x.py",
    ):
        assert bot.matches_any(path, RULES.always_human), path
    for path in ("litellm/llms/openai/chat/transformation.py", "docs/my_website/docs/x.md", "tests/unit/x.py"):
        assert not bot.matches_any(path, RULES.always_human), path


def test_clean_small_code_change_with_a_test_is_a_low_floor() -> None:
    findings = _findings(_pr(), CODE_WITH_TEST, checks=(_check("unit passed", "success"),))
    assert findings == ()


def test_always_human_path_is_high() -> None:
    findings = _findings(_pr(), (_file("litellm/proxy/auth/user_api_key_auth.py"), _file("tests/unit/t.py")))
    assert _floor(findings) == bot.Tier.high
    assert any("always-human" in f.reason for f in findings)


def test_fork_draft_and_closed_are_high() -> None:
    assert _floor(_findings(_pr(head_repo="someone/litellm"), CODE_WITH_TEST)) == bot.Tier.high
    assert _floor(_findings(_pr(draft=True), CODE_WITH_TEST)) == bot.Tier.high
    assert _floor(_findings(_pr(state="closed"), CODE_WITH_TEST)) == bot.Tier.high


def test_over_the_size_cap_is_high() -> None:
    findings = _findings(_pr(additions=RULES.size.judge_max.lines, deletions=1), CODE_WITH_TEST)
    assert _floor(findings) == bot.Tier.high
    many_files = tuple(_file(f"litellm/llms/openai/f{i}.py") for i in range(RULES.size.judge_max.files)) + (
        _file("tests/unit/t.py"),
    )
    assert _floor(_findings(_pr(), many_files)) == bot.Tier.high


def test_code_change_without_a_test_is_medium() -> None:
    findings = _findings(_pr(), (_file("litellm/llms/openai/chat/transformation.py"),))
    assert _floor(findings) == bot.Tier.medium


def test_red_check_or_status_is_high_but_cancelled_or_pending_is_not() -> None:
    assert _floor(_findings(_pr(), CODE_WITH_TEST, checks=(_check("unit passed", "failure"),))) == bot.Tier.high
    assert _floor(_findings(_pr(), CODE_WITH_TEST, checks=(_check("x", None, status="in_progress"),))) == bot.Tier.low
    assert _floor(_findings(_pr(), CODE_WITH_TEST, checks=(_check("Validate PR title", "cancelled"),))) == bot.Tier.low
    red_status = (bot.CommitStatus(context="ci/circleci: integration", state="failure"),)
    assert _floor(_findings(_pr(), CODE_WITH_TEST, statuses=red_status)) == bot.Tier.high


def test_changes_requested_and_unresolved_threads_are_high() -> None:
    assert _floor(_findings(_pr(), CODE_WITH_TEST, reviews=(_review("rev", "CHANGES_REQUESTED"),))) == bot.Tier.high
    assert _floor(_findings(_pr(), CODE_WITH_TEST, unresolved=1)) == bot.Tier.high


def test_changes_requested_then_approved_by_the_same_reviewer_clears() -> None:
    reviews = (_review("rev", "CHANGES_REQUESTED"), _review("rev", "APPROVED"))
    assert _floor(_findings(_pr(), CODE_WITH_TEST, reviews=reviews)) == bot.Tier.low


def test_weakened_tests_are_high() -> None:
    deleted = (_file("litellm/llms/openai/x.py"), _file("tests/unit/t.py", status="removed"))
    assert _floor(_findings(_pr(), deleted)) == bot.Tier.high
    skipped = (
        _file("litellm/llms/openai/x.py"),
        _file(
            "tests/unit/t.py",
            patch="@@ -1,2 +1,3 @@\n+@pytest.mark.skip(reason='flaky')\n def test_x():\n     assert 1",
        ),
    )
    assert _floor(_findings(_pr(), skipped)) == bot.Tier.high
    fewer_asserts = (
        _file("litellm/llms/openai/x.py"),
        _file(
            "tests/unit/t.py",
            patch="@@ -1,3 +1,2 @@\n def test_x():\n-    assert a == 1\n-    assert b == 2\n+    assert a == 1",
        ),
    )
    assert _floor(_findings(_pr(), fewer_asserts)) == bot.Tier.high
    moved_assert = (
        _file("litellm/llms/openai/x.py"),
        _file(
            "tests/unit/t.py",
            patch="@@ -1,3 +1,3 @@\n def test_x():\n-    assert a == 1\n+    assert a == 2\n     pass",
        ),
    )
    assert _floor(_findings(_pr(), moved_assert)) == bot.Tier.low


def test_low_by_rule_is_docs_and_tests_only() -> None:
    assert bot.low_by_rule((_file("docs/my_website/docs/x.md"), _file("tests/unit/t.py")), RULES)
    assert not bot.low_by_rule((_file("docs/x.md"), _file("litellm/llms/openai/x.py")), RULES)
    assert not bot.low_by_rule((), RULES)


def _judgement(tier: str) -> object:
    verdict = bot.Verdict(
        tier=tier, change_kind="additive", blast_radius="one_provider_or_page", reasons="r", look_at=""
    )
    return bot.Judgement("judge-model", verdict)


def test_judge_raises_but_never_lowers() -> None:
    low_floor = bot.Evaluation(_pr(), (), _judgement("high"))
    assert low_floor.final == bot.Tier.high
    medium_floor = bot.Evaluation(_pr(), (bot.Finding(bot.Tier.medium, "no test"),), _judgement("low"))
    assert medium_floor.final == bot.Tier.medium
    skipped = bot.Evaluation(_pr(), (), None)
    assert skipped.final == bot.Tier.low


def test_unusable_judge_answer_is_high() -> None:
    broken = bot.Evaluation(_pr(), (), bot.Judgement("judge-model", "judge call failed with HTTP 500"))
    assert broken.final == bot.Tier.high


def _chat(content: str) -> str:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_parse_verdict() -> None:
    good = _chat(
        json.dumps(
            {
                "tier": "low",
                "change_kind": "additive",
                "blast_radius": "one_provider_or_page",
                "reasons": "adds one param mapping",
                "look_at": "",
            }
        )
    )
    verdict = bot.parse_verdict(good)
    assert isinstance(verdict, bot.Verdict) and verdict.tier == "low"
    assert isinstance(bot.parse_verdict(_chat(json.dumps({"tier": "approve"}))), str)
    assert isinstance(bot.parse_verdict(_chat("not json")), str)
    assert isinstance(bot.parse_verdict("{}"), str)
    assert isinstance(bot.parse_verdict(json.dumps({"choices": [{"message": {"content": None}}]})), str)


def test_decide_only_approves_a_low_tier_with_the_app_and_the_flag() -> None:
    low = bot.Evaluation(_pr(), (), None)
    assert bot.decide(low, approvals_enabled=True, app_token=True).kind == "approved"
    assert bot.decide(low, approvals_enabled=False, app_token=True).kind == "would_approve"
    assert bot.decide(low, approvals_enabled=True, app_token=False).kind == "would_approve"
    medium = bot.Evaluation(_pr(), (bot.Finding(bot.Tier.medium, "no test"),), None)
    assert bot.decide(medium, approvals_enabled=True, app_token=True).kind == "needs_human"


def test_comment_carries_the_marker_tier_sha_and_findings() -> None:
    evaluation = bot.Evaluation(_pr(), (bot.Finding(bot.Tier.high, "`x` is on the always-human list"),), None)
    comment = bot.render_comment(evaluation, bot.decide(evaluation, True, True), "rules.yml")
    assert comment.startswith(bot.MARKER)
    assert "risk high" in comment and "aaaaaaa" in comment and "always-human" in comment


def test_verdict_lines_split_on_newlines_and_strip_bullets() -> None:
    verdict = bot.Verdict(
        tier="medium",
        change_kind="scoped_behavior_change",
        blast_radius="shared_helper",
        reasons="- first\n- second\n\n",
        look_at="litellm/a.py\nlitellm/b.py",
    )
    assert verdict.reason_lines() == ("first", "second")
    assert verdict.look_at_lines() == ("litellm/a.py", "litellm/b.py")
    rendered = bot.render_comment(
        bot.Evaluation(_pr(), (), bot.Judgement("m", verdict)),
        bot.decide(bot.Evaluation(_pr(), (), bot.Judgement("m", verdict)), True, True),
        "r",
    )
    assert "Look at first:" in rendered and "`litellm/a.py`" in rendered and "- second" in rendered


def test_judge_schema_matches_the_verdict_model() -> None:
    schema = json.loads((ROOT / ".github" / "stampbot" / "judge.schema.json").read_text())
    assert set(schema["required"]) == set(bot.Verdict.model_fields)
    assert schema["properties"]["tier"]["enum"] == ["low", "medium", "high"]


def test_judge_user_message_is_data_not_instructions() -> None:
    files = (_file("litellm/llms/openai/x.py", patch="@@ -1 +1 @@\n-a\n+b"),)
    message = bot.judge_user_message(_pr(), files, RULES)
    assert "## Diff" in message and "--- litellm/llms/openai/x.py" in message and "(small by rule)" in message

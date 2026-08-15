"""Tests for the vacuous-test audit tooling.

Both directions are pinned: the classifier flags what it claims to flag, and it
leaves healthy tests alone. A false positive here sends the daily automation to
rewrite a working test
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from datetime import date
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flake_gate
import guardrails
import inventory
import mutation_probe


def findings_for(source: str, monkeypatch, tmp_path) -> List[str]:
    path = tmp_path / "test_target.py"
    path.write_text(textwrap.dedent(source))
    monkeypatch.setattr(flake_gate, "REPO_ROOT", str(tmp_path))
    return [finding.problem for finding in flake_gate.static_findings("test_target.py::test_thing")]


def bucket_of(source: str, name: str = "test_thing") -> Optional[str]:
    candidates = inventory.classify_file("/repo/tests/test_sample.py", textwrap.dedent(source))
    for candidate in candidates:
        if candidate.name.endswith(name):
            return candidate.bucket
    return None


def test_flags_trivial_assert() -> None:
    assert bucket_of("def test_thing():\n    do_work()\n    assert True\n") == "trivial_assert"


def test_flags_self_comparison() -> None:
    source = """
    def test_thing():
        value = compute()
        assert value == value
    """
    assert bucket_of(source) == "trivial_assert"


def test_flags_missing_assertion() -> None:
    assert bucket_of("def test_thing():\n    result = compute()\n    print(result)\n") == "no_assert"


def test_flags_swallowed_assertion() -> None:
    source = """
    def test_thing():
        try:
            assert compute() == 3
        except Exception:
            pass
    """
    assert bucket_of(source) == "swallowed_failure"


def test_flags_unconditional_skip() -> None:
    source = """
    @pytest.mark.skip(reason="broken")
    def test_thing():
        assert compute() == 3
    """
    assert bucket_of(source) == "dead_skip"


def test_flags_mock_only_comparison() -> None:
    source = """
    def test_thing():
        client = MagicMock()
        other = MagicMock()
        assert client.send.return_value == other.send.return_value
    """
    assert bucket_of(source) == "mock_tautology"


def test_ignores_real_assertion() -> None:
    source = """
    def test_thing():
        assert compute() == 3
    """
    assert bucket_of(source) is None


def test_ignores_pytest_raises_only_test() -> None:
    source = """
    def test_thing():
        with pytest.raises(ValueError):
            compute()
    """
    assert bucket_of(source) is None


def test_ignores_assertion_in_shared_helper() -> None:
    source = """
    def test_thing():
        assert_response_matches(compute(), expected)
    """
    assert bucket_of(source) is None


def test_ignores_mock_passed_into_real_code() -> None:
    source = """
    def test_thing():
        client = MagicMock()
        result = handler(client)
        assert result == client.send.return_value
    """
    assert bucket_of(source) is None


def test_ignores_conditional_skip() -> None:
    source = """
    @pytest.mark.skipif(sys.platform == "win32", reason="posix only")
    def test_thing():
        assert compute() == 3
    """
    assert bucket_of(source) is None


def test_classifies_methods_of_test_classes() -> None:
    source = """
    class TestThings:
        def test_thing(self):
            compute()
    """
    candidates = inventory.classify_file("/repo/tests/test_sample.py", textwrap.dedent(source))
    assert [c.name for c in candidates] == ["TestThings.test_thing"]


def test_ratchet_rejects_new_candidates() -> None:
    failures = inventory.regressions(
        {"tests/test_sample.py": {"no_assert": 2}},
        {"tests/test_sample.py": {"no_assert": 1}},
    )
    assert failures == ["tests/test_sample.py: no_assert went from 1 to 2"]


def test_ratchet_allows_fewer_candidates_and_untouched_files() -> None:
    baseline = {"tests/test_sample.py": {"no_assert": 2}, "tests/test_other.py": {"dead_skip": 1}}
    assert inventory.regressions({"tests/test_sample.py": {"no_assert": 1}}, baseline) == []


def test_ratchet_rejects_candidates_in_a_new_file() -> None:
    failures = inventory.regressions({"tests/test_new.py": {"trivial_assert": 1}}, {})
    assert failures == ["tests/test_new.py: trivial_assert went from 0 to 1"]


def _mutant_descriptions(source: str, lines: List[int], tmp_path: str) -> List[str]:
    path = os.path.join(tmp_path, "module.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(source))
    relative = os.path.relpath(path, mutation_probe.REPO_ROOT)
    return [mutant.description for mutant in mutation_probe.generate_mutants(relative, lines)]


def test_mutates_only_covered_lines(tmp_path) -> None:
    source = """
    def covered(value):
        return value > 3

    def uncovered(value):
        return value < 9
    """
    descriptions = _mutant_descriptions(source, [3], str(tmp_path))
    assert any("value > 3" in description for description in descriptions)
    assert not any("value < 9" in description for description in descriptions)


def test_mutant_source_is_valid_python(tmp_path) -> None:
    source = """
    def covered(value):
        if value == 3 and value is not None:
            return "three"
        return None
    """
    path = os.path.join(str(tmp_path), "module.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(source))
    relative = os.path.relpath(path, mutation_probe.REPO_ROOT)
    mutants = mutation_probe.generate_mutants(relative, [3, 4, 5])
    assert mutants
    for mutant in mutants:
        ast.parse(mutant.source)
    assert any("flip `Eq`" in mutant.description for mutant in mutants)


def test_import_time_lines_are_not_behavioural(tmp_path) -> None:
    source = """
    DEFAULT = True

    class Config:
        enabled = False

        def check(self):
            return self.enabled is True
    """
    path = os.path.join(str(tmp_path), "module.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(source))
    relative = os.path.relpath(path, mutation_probe.REPO_ROOT)
    assert mutation_probe.behavioural_lines(relative, range(1, 9)) == [8]


def test_overlay_isolates_the_mutant_from_the_real_tree(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "tests").mkdir()
    target = root / "pkg" / "sub" / "mod.py"
    target.write_text("X = 1\n")
    (root / "pkg" / "other.py").write_text("Y = 1\n")
    monkeypatch.setattr(mutation_probe, "REPO_ROOT", str(root))
    mutant = mutation_probe.Mutant(path="pkg/sub/mod.py", lineno=1, description="d", source="X = 2\n")

    with mutation_probe.mutant_overlay(mutant) as overlay:
        assert open(os.path.join(overlay, "pkg", "sub", "mod.py"), encoding="utf-8").read() == "X = 2\n"
        assert not os.path.islink(os.path.join(overlay, "pkg", "sub", "mod.py"))
        assert os.path.islink(os.path.join(overlay, "tests"))
        assert os.path.islink(os.path.join(overlay, "pkg", "other.py"))
        assert target.read_text() == "X = 1\n"

    assert not os.path.exists(overlay)
    assert target.read_text() == "X = 1\n"


def test_module_under_test_is_recognised_from_imports() -> None:
    imports = {"litellm.llms.bedrock.base_aws_llm", "litellm.llms.bedrock.base_aws_llm.BaseAWSLLM"}
    assert mutation_probe._is_under_test("litellm/llms/bedrock/base_aws_llm.py", imports)
    assert not mutation_probe._is_under_test("litellm/caching/dual_cache.py", imports)


def test_swallowed_exception_becomes_a_mutant_that_a_no_assert_test_can_notice(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    (root / "litellm").mkdir(parents=True)
    target = root / "litellm" / "hooks.py"
    target.write_text(
        textwrap.dedent(
            """
            def record(value):
                try:
                    return int(value)
                except ValueError:
                    return 0
            """
        ).lstrip()
    )
    monkeypatch.setattr(mutation_probe, "REPO_ROOT", str(root))
    mutants = mutation_probe.generate_mutants("litellm/hooks.py", range(1, 6))

    swallows = [m for m in mutants if m.swallow]
    assert len(swallows) == 1
    assert "stop swallowing" in swallows[0].description
    assert "raise" in swallows[0].source
    assert "return 0" not in swallows[0].source
    # A test that only claims "this does not raise" dies to that mutant and to
    # nothing else, so it has to be tried first.
    assert mutation_probe.select_mutants(mutants, 2)[0].swallow


def test_already_reraising_handlers_produce_no_mutant(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    (root / "litellm").mkdir(parents=True)
    (root / "litellm" / "hooks.py").write_text(
        textwrap.dedent(
            """
            def record(value):
                try:
                    return int(value)
                except ValueError:
                    raise
            """
        ).lstrip()
    )
    monkeypatch.setattr(mutation_probe, "REPO_ROOT", str(root))

    assert not [m for m in mutation_probe.generate_mutants("litellm/hooks.py", range(1, 6)) if m.swallow]


def test_area_rotation_moves_on_each_day_and_is_stable_within_one(monkeypatch) -> None:
    monkeypatch.setattr(inventory, "cleared_ids", lambda: frozenset())
    candidates = [
        inventory.Candidate(path=path, lineno=index, name=f"test_{index}", bucket="no_assert", evidence="e")
        for index, path in enumerate(
            ["tests/a/one.py"] * 3 + ["tests/b/two.py"] * 2 + ["tests/c/three.py"],
        )
    ]
    assert inventory.areas(candidates) == (("tests/a", 3), ("tests/b", 2), ("tests/c", 1))
    picks = [inventory.rotated_area(candidates, date(2026, 8, day)) for day in (15, 16, 17, 18)]
    assert len(set(picks[:3])) == 3
    assert picks[3] == picks[0]
    assert inventory.rotated_area(candidates, date(2026, 8, 15)) == picks[0]
    assert inventory.rotated_area([], date(2026, 8, 15)) is None


def test_budget_reaches_the_module_under_test() -> None:
    def mutant(path: str, lineno: int) -> mutation_probe.Mutant:
        return mutation_probe.Mutant(path=path, lineno=lineno, description=f"{path}:{lineno}", source="")

    under_test = [mutant("litellm/llms/bedrock/base_aws_llm.py", line) for line in (10, 11)]
    shared = [mutant("litellm/caching/dual_cache.py", line) for line in range(100, 120)]
    chosen = mutation_probe.interleave([under_test, shared], 8)
    assert [m.path for m in chosen].count("litellm/llms/bedrock/base_aws_llm.py") == 2
    assert len(chosen) == 8


def test_flake_gate_flags_sleep_and_wall_clock(monkeypatch, tmp_path) -> None:
    source = """
    def test_thing():
        time.sleep(0.5)
        started = datetime.datetime.now()
        assert started
    """
    problems = findings_for(source, monkeypatch, tmp_path)
    assert any("time.sleep" in problem for problem in problems)
    assert any("datetime.now" in problem for problem in problems)


def test_flake_gate_flags_live_call_outside_the_patch(monkeypatch, tmp_path) -> None:
    source = """
    def test_thing():
        with patch("litellm.main.completion") as mocked:
            litellm.completion(model="gpt-5", messages=[])
        litellm.acompletion(model="gpt-5", messages=[])
    """
    problems = findings_for(source, monkeypatch, tmp_path)
    assert problems == ["uses `litellm.acompletion`: hits a live provider unless mocked or replayed"]


def test_flake_gate_accepts_patched_network_call(monkeypatch, tmp_path) -> None:
    source = """
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_thing(mocked_post):
        response = litellm.completion(model="gpt-5", messages=[])
        assert response.choices
    """
    assert findings_for(source, monkeypatch, tmp_path) == []


def test_flake_gate_does_not_excuse_sleep_in_a_patched_test(monkeypatch, tmp_path) -> None:
    source = """
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_thing(mocked_post):
        time.sleep(1)
        assert mocked_post.called
    """
    assert findings_for(source, monkeypatch, tmp_path) == [
        "uses `time.sleep`: wall-clock sleep: slow and racy under load"
    ]


def test_removals_need_a_citation_each() -> None:
    removed = frozenset({"test_one", "test_two"})
    citations = {"tests/test_sample.py::test_one": "tests/test_sample.py::test_covers_one"}
    problems = guardrails.uncited_removals("tests/test_sample.py", removed, citations)
    assert len(problems) == 1
    assert "test_two was removed without a citation" in problems[0]


def test_blank_citation_does_not_count() -> None:
    problems = guardrails.uncited_removals(
        "tests/test_sample.py", frozenset({"test_one"}), {"tests/test_sample.py::test_one": "  "}
    )
    assert len(problems) == 1

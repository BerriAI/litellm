import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_any_discipline.py"
)
_spec = importlib.util.spec_from_file_location("check_any_discipline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

Violation = mod.Violation


def _v(path="litellm/x.py", line=10, code="LIT009"):
    return Violation(Path(path), line, 0, code, "Any-typed value")


# --------------------------------------------------------------------------- #
# cap_for: ~25% headroom over the grandfathered baseline
# --------------------------------------------------------------------------- #


def test_zero_baseline_gets_no_headroom():
    # A file with no recorded debt (a brand new file) must stay Any-free.
    assert mod.cap_for(0) == 0


def test_headroom_is_a_quarter_of_the_baseline():
    assert mod.cap_for(100) == 125
    assert mod.cap_for(40) == 50


def test_headroom_rounds_up_so_small_baselines_get_at_least_one_slot():
    # ceil(1 * 0.25) == 1, ceil(4 * 0.25) == 1, ceil(8 * 0.25) == 2.
    assert mod.cap_for(1) == 2
    assert mod.cap_for(4) == 5
    assert mod.cap_for(8) == 10


# --------------------------------------------------------------------------- #
# lit009_counts: only Any findings are counted, grouped per file
# --------------------------------------------------------------------------- #


def test_counts_only_lit009_per_file():
    counts = mod.lit009_counts(
        [
            _v(path="litellm/a.py", line=1),
            _v(path="litellm/a.py", line=2),
            _v(path="litellm/b.py", line=9),
            _v(path="litellm/a.py", line=3, code="LIT005"),
            _v(path="litellm/a.py", line=0, code="LIT000"),
        ]
    )
    assert counts == {"litellm/a.py": 2, "litellm/b.py": 1}


# --------------------------------------------------------------------------- #
# budget_breaches: a file fails only above baseline + ~25%
# --------------------------------------------------------------------------- #


def test_file_at_its_ceiling_does_not_breach():
    assert mod.budget_breaches({"litellm/a.py": 125}, {"litellm/a.py": 100}) == []


def test_file_one_over_its_ceiling_breaches():
    assert mod.budget_breaches({"litellm/a.py": 126}, {"litellm/a.py": 100}) == [
        ("litellm/a.py", 126, 125)
    ]


def test_file_below_its_baseline_does_not_breach():
    assert mod.budget_breaches({"litellm/a.py": 70}, {"litellm/a.py": 100}) == []


def test_new_file_with_any_breaches_against_a_zero_budget():
    # A file absent from the budget is grandfathered at zero -> ceiling 0.
    assert mod.budget_breaches({"litellm/new.py": 1}, {}) == [("litellm/new.py", 1, 0)]


def test_clean_file_never_breaches():
    assert mod.budget_breaches({}, {"litellm/a.py": 100}) == []

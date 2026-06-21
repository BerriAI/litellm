"""Tests for scripts/check_type_discipline.py.

Each rule is exercised on a snippet that violates it and on one that does not, so a
mutation that drops a rule, inverts a suppression, or breaks the comment scanner makes
a test fail. The comment-scanner cases are the regression for the readline path: if
`scan_comments` ever stops tokenizing comments, the LIT003/LIT005 assertions go red.
"""

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "check_type_discipline.py"
_spec = importlib.util.spec_from_file_location("check_type_discipline", _MODULE_PATH)
checker = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = checker  # let the frozen dataclass resolve its own module
_spec.loader.exec_module(checker)


def _codes(tmp_path, source):
    f = tmp_path / "snippet.py"
    f.write_text(source, encoding="utf-8")
    return [v.code for v in checker.check_file(f)]


# --------------------------------------------------------------------------- #
# Comment scanning (the readline path) — LIT003 / LIT004 / LIT005
# --------------------------------------------------------------------------- #


def test_scan_comments_tokenizes_every_comment():
    # Direct regression for scan_comments: a bare noqa (LIT003) only surfaces if the comment
    # was tokenized, and the valid cast-ok suppression line must be captured. A crash in the
    # readline path would leave both empty.
    source = "x = 1  # noqa\ny = 2  # cast-ok: validated upstream by the caller\n"
    comments, violations = checker.scan_comments(Path("snippet.py"), source)
    assert [v.code for v in violations] == ["LIT003"]
    assert comments.cast_ok_lines == frozenset({2})


def test_scan_comments_does_not_crash_on_malformed_source():
    # A dedent mismatch makes tokenize raise IndentationError (a SyntaxError subclass);
    # scan_comments must swallow it, not propagate and crash the whole run.
    comments, violations = checker.scan_comments(Path("x.py"), "if True:\n    a = 1\n  b = 2\n")
    assert violations == ()
    assert comments.cast_ok_lines == frozenset()


def test_malformed_source_degrades_to_lit000(tmp_path):
    # The checker's contract is "bad file -> LIT000, never crash". An untokenizable file
    # falls through scan_comments to ast.parse, which is reported as a single LIT000.
    assert _codes(tmp_path, "if True:\n    a = 1\n  b = 2\n") == ["LIT000"]


def test_noqa_without_codes_is_flagged(tmp_path):
    assert "LIT003" in _codes(tmp_path, "x = 1  # noqa\n")


def test_noqa_with_codes_and_reason_is_clean(tmp_path):
    assert "LIT003" not in _codes(tmp_path, "x = 1  # noqa: TID251  # legacy import, removed in #123\n")


def test_ignore_without_reason_is_flagged(tmp_path):
    assert "LIT004" in _codes(tmp_path, "x = 1  # type: ignore[arg-type]\n")


def test_ignore_with_codes_and_reason_is_clean(tmp_path):
    assert "LIT004" not in _codes(tmp_path, "x = 1  # pyright: ignore[reportArgumentType]  # upstream stub is wrong\n")


def test_ok_suppression_without_reason_is_flagged(tmp_path):
    codes = _codes(tmp_path, "y = []  # mutable-ok\n")
    assert "LIT005" in codes  # reasonless suppression
    assert "LIT002" in codes  # and it does not suppress, so the construction still trips


# --------------------------------------------------------------------------- #
# Mutable annotations (LIT001) and construction (LIT002)
# --------------------------------------------------------------------------- #


def test_mutable_annotation_is_flagged(tmp_path):
    assert "LIT001" in _codes(tmp_path, "x: dict[str, int]\n")


def test_typing_alias_and_forward_ref_annotations_are_flagged(tmp_path):
    assert "LIT001" in _codes(tmp_path, "from typing import List\nx: List[int]\n")
    assert "LIT001" in _codes(tmp_path, 'x: "dict[str, int]"\n')


def test_readonly_annotations_are_clean(tmp_path):
    for ann in ("Mapping[str, int]", "Sequence[int]", "tuple[int, ...]", "frozenset[int]"):
        assert "LIT001" not in _codes(tmp_path, f"from typing import Mapping, Sequence\nx: {ann}\n")


def test_mutable_construction_is_flagged(tmp_path):
    assert "LIT002" in _codes(tmp_path, "y = []\n")
    assert "LIT002" in _codes(tmp_path, "z = dict(a=1)\n")


def test_construction_inside_annotation_is_exempt(tmp_path):
    # `Callable[[int], str]` carries a list display that is type syntax, not construction.
    assert "LIT002" not in _codes(
        tmp_path, "from typing import Callable\ndef f(cb: Callable[[int], str]) -> None:\n    return None\n"
    )


def test_generator_and_tuple_are_not_construction(tmp_path):
    assert "LIT002" not in _codes(tmp_path, "g = tuple(i for i in range(3))\n")
    assert "LIT002" not in _codes(tmp_path, "t = (1, 2, 3)\n")


def test_dict_list_set_method_calls_are_not_construction(tmp_path):
    # `.dict()` / `.list()` / `.set()` are common method names (e.g. pydantic model.dict()),
    # not collection construction; only the unqualified builtins count.
    assert "LIT002" not in _codes(tmp_path, "d = model.dict()\n")
    assert "LIT002" not in _codes(tmp_path, "s = obj.set()\n")
    assert "LIT002" in _codes(tmp_path, "d = dict(a=1)\n")  # unqualified still counts


def test_qualified_collections_constructors_still_count(tmp_path):
    # collections concretes are rarely method names, so a qualified call still flags.
    assert "LIT002" in _codes(tmp_path, "import collections\nq = collections.deque()\n")
    assert "LIT002" in _codes(tmp_path, "import collections\nm = collections.defaultdict(list)\n")


def test_mutable_ok_with_reason_suppresses_both_rules(tmp_path):
    codes = _codes(tmp_path, "x: dict[str, int] = {}  # mutable-ok: in-place buffer mutated hot path\n")
    assert "LIT001" not in codes
    assert "LIT002" not in codes


# --------------------------------------------------------------------------- #
# Casts (LIT006)
# --------------------------------------------------------------------------- #


def test_cast_call_is_flagged(tmp_path):
    assert "LIT006" in _codes(tmp_path, "from typing import cast\ny = cast(int, object())\n")


def test_cast_ok_with_reason_suppresses(tmp_path):
    assert "LIT006" not in _codes(
        tmp_path, "from typing import cast\ny = cast(int, object())  # cast-ok: validated by schema above\n"
    )


# --------------------------------------------------------------------------- #
# Narrowing predicates (LIT007) — must fire only in return annotations
# --------------------------------------------------------------------------- #


def test_guard_in_return_annotation_is_flagged(tmp_path):
    src = "from typing import TypeGuard\ndef is_int(v: object) -> TypeGuard[int]:\n    return isinstance(v, int)\n"
    assert "LIT007" in _codes(tmp_path, src)


def test_guard_name_outside_annotation_is_not_flagged(tmp_path):
    # A runtime name or attribute that merely reads `TypeGuard`/`TypeIs` is not a predicate.
    assert "LIT007" not in _codes(tmp_path, "TypeGuard = 1\nx = TypeGuard + 1\n")
    assert "LIT007" not in _codes(tmp_path, "import obj\n_ = obj.TypeIs\n")


def test_guard_ok_with_reason_suppresses(tmp_path):
    src = (
        "from typing import TypeGuard\n"
        "def is_int(v: object) -> TypeGuard[int]:  # guard-ok: predicate proven by the assert below\n"
        "    assert isinstance(v, int)\n"
        "    return True\n"
    )
    assert "LIT007" not in _codes(tmp_path, src)


# --------------------------------------------------------------------------- #
# **kwargs (LIT008) — typed *args stays clean
# --------------------------------------------------------------------------- #


def test_kwargs_parameter_is_flagged(tmp_path):
    assert "LIT008" in _codes(tmp_path, "def f(**kwargs) -> None:\n    return None\n")


def test_typed_args_is_clean_but_kwargs_ok_suppresses(tmp_path):
    assert "LIT008" not in _codes(tmp_path, "def f(*args: int) -> None:\n    return None\n")
    assert "LIT008" not in _codes(
        tmp_path, "def f(**kwargs: int) -> None:  # kwargs-ok: passthrough to a third-party sink\n    return None\n"
    )


# --------------------------------------------------------------------------- #
# Budget integrity: every emittable LIT rule (bar the LIT000 read/parse error) is gated
# --------------------------------------------------------------------------- #


def test_budget_covers_exactly_the_checker_rules():
    budget = json.loads((_REPO_ROOT / "type-discipline-budget.json").read_text())
    assert set(budget) == {f"LIT00{n}" for n in range(1, 9)}

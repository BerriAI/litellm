"""Tests for scripts/check_type_discipline.py.

Each rule is exercised on a snippet that violates it and on one that does not, so a
mutation that drops a rule, inverts a suppression, or breaks the comment scanner makes
a test fail. The comment-scanner cases are the regression for the readline path: if
`scan_comments` ever stops tokenizing comments, the LIT003/LIT005 assertions go red.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_pyright_ignore_without_codes_is_flagged(tmp_path):
    assert "LIT004" in _codes(tmp_path, "x = 1  # pyright: ignore\n")


def test_pyright_ignore_without_reason_is_flagged(tmp_path):
    assert "LIT004" in _codes(tmp_path, "x = 1  # pyright: ignore[reportArgumentType]\n")


def test_ignore_with_codes_and_reason_is_clean(tmp_path):
    codes = _codes(tmp_path, "x = 1  # pyright: ignore[reportArgumentType]  # upstream stub is wrong\n")
    assert "LIT004" not in codes
    assert "LIT009" not in codes


def test_bare_type_ignore_is_flagged(tmp_path):
    assert "LIT009" in _codes(tmp_path, "x = 1  # type: ignore\n")


def test_type_ignore_with_codes_and_reason_is_still_flagged(tmp_path):
    codes = _codes(tmp_path, "x = 1  # type: ignore[arg-type]  # inertness is the point\n")
    assert "LIT009" in codes
    assert "LIT004" not in codes


def test_prose_mentioning_type_ignored_is_not_flagged(tmp_path):
    assert "LIT009" not in _codes(tmp_path, "x = 1  # type: ignored by the stub refresh, revisit\n")


def test_mypy_ignore_shape_is_lit004_not_lit009(tmp_path):
    codes = _codes(tmp_path, "x = 1  # mypy: ignore[assignment]\n")
    assert "LIT004" in codes
    assert "LIT009" not in codes


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


def test_literal_string_args_are_values_not_forward_refs(tmp_path):
    assert "LIT001" not in _codes(tmp_path, 'from typing import Literal\nx: Literal["list"] = "list"\n')
    assert "LIT001" not in _codes(
        tmp_path,
        'from typing import Literal\ndef f(op: Literal["create", "list"] = "create") -> None:\n    return None\n',
    )
    assert "LIT001" not in _codes(tmp_path, 'import typing\nx: typing.Literal["dict"] = "dict"\n')
    assert "LIT001" in _codes(tmp_path, 'from typing import Literal\nx: dict[str, Literal["a"]]\n')
    assert "LIT001" in _codes(tmp_path, "x: \"Literal['x'] | list[int]\"\n")


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


def test_value_frozen_by_wrapper_is_exempt(tmp_path):
    assert "LIT002" not in _codes(tmp_path, "from types import MappingProxyType\nm = MappingProxyType({'a': 1})\n")
    assert "LIT002" not in _codes(tmp_path, "import types\nm = types.MappingProxyType({'a': 1})\n")
    assert "LIT002" not in _codes(tmp_path, "from types import MappingProxyType\nm = MappingProxyType(dict(a=1))\n")
    assert "LIT002" not in _codes(tmp_path, "f = frozenset({1, 2})\n")
    assert "LIT002" not in _codes(tmp_path, "t = tuple([1, 2])\n")


def test_same_named_method_does_not_exempt_its_argument(tmp_path):
    assert "LIT002" in _codes(tmp_path, "t = obj.tuple([1, 2])\n")
    assert "LIT002" in _codes(tmp_path, "f = obj.frozenset({1, 2})\n")
    assert "LIT002" in _codes(tmp_path, "m = obj.MappingProxyType({'a': 1})\n")


def test_mutable_nested_inside_frozen_wrapper_still_counts(tmp_path):
    assert "LIT002" in _codes(tmp_path, "from types import MappingProxyType\nm = MappingProxyType({'a': []})\n")


def test_unfrozen_literal_still_counts(tmp_path):
    assert "LIT002" in _codes(tmp_path, "from types import MappingProxyType\nd = {'a': 1}\nm = MappingProxyType(d)\n")


def test_lit002_fix_message_names_mappingproxytype(tmp_path):
    f = tmp_path / "snippet.py"
    f.write_text("x = {'a': 1}\n", encoding="utf-8")
    messages = [v.message for v in checker.check_file(f) if v.code == "LIT002"]
    assert "MappingProxyType" in messages[0]


def test_mutable_ok_with_reason_suppresses_both_rules(tmp_path):
    codes = _codes(tmp_path, "x: dict[str, int] = {}  # mutable-ok: in-place buffer mutated hot path\n")
    assert "LIT001" not in codes
    assert "LIT002" not in codes


def test_typeddict_annotated_dict_literal_is_exempt(tmp_path):
    assert "LIT002" not in _codes(
        tmp_path, "from typing import Final\nfrom foo import MyTD\nx: Final[MyTD] = {'a': 1}\n"
    )
    assert "LIT002" not in _codes(tmp_path, "from foo import MyTD\nx: MyTD = {'a': 1}\n")
    assert "LIT002" not in _codes(tmp_path, "from typing import Final\nx: Final['MyTD'] = {'a': 1}\n")
    assert "LIT002" not in _codes(tmp_path, "import foo\nfrom typing import Final\nx: Final[foo.MyTD] = {'a': 1}\n")


def test_wrapped_typeddict_annotations_share_the_exemption(tmp_path):
    assert "LIT002" not in _codes(
        tmp_path, "from typing import Final, Optional\nx: Final[Optional[MyTD]] = {'a': 1}\n"
    )
    assert "LIT002" not in _codes(
        tmp_path, "from typing import Annotated, Final\nx: Final[Annotated[MyTD, 'meta']] = {'a': 1}\n"
    )
    assert "LIT002" not in _codes(
        tmp_path, "from typing import ClassVar\nclass C:\n    x: ClassVar[MyTD] = {'a': 1}\n"
    )
    assert "LIT002" not in _codes(tmp_path, "from typing import Final\nx: Final[MyTD | None] = {'a': 1}\n")
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[dict[str, int] | None] = {'a': 1}\n")


def test_bare_final_dict_literal_still_counts(tmp_path):
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final = {'a': 1}\n")
    assert "LIT002" in _codes(tmp_path, "from typing import ClassVar\nclass C:\n    x: ClassVar = {'a': 1}\n")


def test_non_typeddict_annotations_do_not_exempt(tmp_path):
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[dict[str, int]] = {'a': 1}\n")
    assert "LIT002" in _codes(
        tmp_path, "from collections.abc import Mapping\nfrom typing import Final\nx: Final[Mapping[str, int]] = {'a': 1}\n"
    )
    assert "LIT002" in _codes(tmp_path, "from typing import Any, Final\nx: Final[Any] = {'a': 1}\n")
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[object] = {'a': 1}\n")


def test_typeddict_exemption_covers_only_dict_literals(tmp_path):
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[MyTD] = dict(a=1)\n")
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[MyTD] = {k: 1 for k in ('a',)}\n")


def test_nested_dict_literals_share_the_typeddict_exemption(tmp_path):
    assert "LIT002" not in _codes(
        tmp_path, "from typing import Final\nx: Final[Outer] = {'inner': {'a': 1}, 'steps': ({'b': 2},)}\n"
    )
    assert "LIT002" in _codes(tmp_path, "from typing import Final\nx: Final[Outer] = {'tags': ['a']}\n")


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
# Final declarations (LIT010)
# --------------------------------------------------------------------------- #


def test_unannotated_assignment_is_flagged(tmp_path):
    assert "LIT010" in _codes(tmp_path, "x = 1\n")


def test_non_final_annotation_is_still_flagged(tmp_path):
    assert "LIT010" in _codes(tmp_path, "x: int = 1\n")


def test_final_annotated_assignments_are_clean(tmp_path):
    codes = _codes(tmp_path, "from typing import Final\nx: Final = 1\ny: Final[int] = 2\n")
    assert "LIT010" not in codes


def test_each_rebinding_site_is_flagged(tmp_path):
    assert _codes(tmp_path, "x = 1\nx = 2\n").count("LIT010") == 2


def test_bare_final_declaration_allows_deferred_branch_assignment(tmp_path):
    src = (
        "from typing import Final\n"
        "def f(flag: bool) -> int:\n"
        "    result: Final[int]\n"
        "    if flag:\n"
        "        result = 1\n"
        "    else:\n"
        "        result = 2\n"
        "    return result\n"
    )
    assert "LIT010" not in _codes(tmp_path, src)


def test_unpacking_targets_are_implicitly_final(tmp_path):
    assert "LIT010" not in _codes(tmp_path, "a, b = (1, 2)\n")


def test_reassignment_after_unpack_is_flagged(tmp_path):
    assert _codes(tmp_path, "a, b = (1, 2)\na = 3\n").count("LIT010") == 1


def test_augmented_assignment_is_flagged(tmp_path):
    assert _codes(tmp_path, "x = 1  # rebind-ok: seeded accumulator\nx += 1\n").count("LIT010") == 1


def test_valueless_annotation_is_clean(tmp_path):
    assert "LIT010" not in _codes(tmp_path, "x: int\n")


def test_repeated_unpack_flags_each_rebound_name(tmp_path):
    assert _codes(tmp_path, "a, b = (1, 2)\na, b = (b, a)\n").count("LIT010") == 2


def test_walrus_rebinding_is_flagged(tmp_path):
    src = "if (m := 1):\n    pass\nif (m := 2):\n    pass\n"
    assert _codes(tmp_path, src).count("LIT010") == 1


def test_unpack_after_global_declaration_is_flagged(tmp_path):
    src = (
        "count = 0  # rebind-ok: seeded module counter\n"
        "def f() -> None:\n"
        "    global count\n"
        "    count, other = (1, 2)\n"
    )
    assert _codes(tmp_path, src).count("LIT010") == 1


def test_single_leading_underscore_is_not_exempt(tmp_path):
    assert "LIT010" in _codes(tmp_path, "_x = 1\n")


def test_rebind_ok_suppresses_a_repeated_unpack(tmp_path):
    src = "a, b = (1, 2)\na, b = (b, a)  # rebind-ok: swap in place\n"
    assert "LIT010" not in _codes(tmp_path, src)


def test_loop_body_assignment_is_exempt(tmp_path):
    assert "LIT010" not in _codes(tmp_path, "for i in range(3):\n    x = i\n")


def test_non_assignment_binding_forms_are_exempt(tmp_path):
    src = (
        "import os\n"
        "for i in range(3):\n"
        "    pass\n"
        "with open('f') as fh:\n"
        "    pass\n"
        "try:\n"
        "    pass\n"
        "except Exception as exc:\n"
        "    pass\n"
        "if (n := 1):\n"
        "    pass\n"
    )
    assert "LIT010" not in _codes(tmp_path, src)


def test_dunder_underscore_class_body_and_type_alias_are_exempt(tmp_path):
    src = (
        "from typing import TypeAlias\n"
        "__all__ = ['C']\n"
        "_ = 1\n"
        "Alias: TypeAlias = str\n"
        "class C:\n"
        "    field = 1\n"
    )
    assert "LIT010" not in _codes(tmp_path, src)


def test_comprehension_targets_are_exempt(tmp_path):
    src = "from typing import Final\nsquares: Final = tuple(i * i for i in range(3))\n"
    assert "LIT010" not in _codes(tmp_path, src)


def test_global_reassignment_inside_function_is_flagged(tmp_path):
    src = (
        "count = 0  # rebind-ok: seeded module counter\n"
        "def bump() -> None:\n"
        "    global count\n"
        "    count = 1\n"
    )
    assert _codes(tmp_path, src).count("LIT010") == 1


def test_inner_function_scope_is_checked(tmp_path):
    src = "def outer() -> None:\n    def inner() -> None:\n        x = 1\n"
    assert "LIT010" in _codes(tmp_path, src)


def test_rebind_ok_with_reason_suppresses_lit010(tmp_path):
    src = "x = 1  # rebind-ok: accumulator seeded here\nx = 2  # rebind-ok: accumulator grows\n"
    assert "LIT010" not in _codes(tmp_path, src)


def test_rebind_ok_without_reason_is_flagged_and_does_not_suppress(tmp_path):
    codes = _codes(tmp_path, "x = 1  # rebind-ok\n")
    assert "LIT005" in codes
    assert "LIT010" in codes


def _codes_at(tmp_path, parts, source):
    f = tmp_path.joinpath(*parts)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source, encoding="utf-8")
    return [v.code for v in checker.check_file(f)]


def test_config_surface_module_scope_is_exempt(tmp_path):
    src = "api_key = None\napi_key = 'sk'\n"
    assert "LIT010" not in _codes_at(tmp_path, ("litellm", "__init__.py"), src)


def test_config_surface_function_scopes_are_still_checked(tmp_path):
    src = "def f() -> None:\n    x = 1\n"
    assert "LIT010" in _codes_at(tmp_path, ("litellm", "__init__.py"), src)


def test_nested_init_modules_are_not_config_surface(tmp_path):
    assert "LIT010" in _codes_at(tmp_path, ("litellm", "types", "__init__.py"), "x = 1\n")


# --------------------------------------------------------------------------- #
# Parameter rebinding and in-place mutation (LIT011)
# --------------------------------------------------------------------------- #


def test_parameter_rebinding_is_flagged(tmp_path):
    assert "LIT011" in _codes(tmp_path, "def f(a: int) -> int:\n    a = 2\n    return a\n")


def test_parameter_augmented_assignment_is_flagged(tmp_path):
    assert "LIT011" in _codes(tmp_path, "def f(a: int) -> int:\n    a += 1\n    return a\n")


def test_parameter_in_place_mutation_sites_are_each_flagged(tmp_path):
    src = (
        "def f(payload: dict, options: object) -> None:\n"
        "    payload['k'] = 1\n"
        "    options.attr = 2\n"
        "    del payload['k']\n"
    )
    assert _codes(tmp_path, src).count("LIT011") == 3


def test_self_and_cls_attribute_stores_are_exempt(tmp_path):
    src = (
        "class C:\n"
        "    def m(self, v: int) -> None:\n"
        "        self.v = v\n"
        "    @classmethod\n"
        "    def cm(cls, v: int) -> None:\n"
        "        cls.v = v\n"
    )
    assert "LIT011" not in _codes(tmp_path, src)


def test_local_variable_mutation_is_not_lit011(tmp_path):
    src = "from typing import Final\ndef f() -> None:\n    box: Final = {}\n    box['k'] = 1\n"
    assert "LIT011" not in _codes(tmp_path, src)


def test_untouched_parameter_is_clean(tmp_path):
    assert "LIT011" not in _codes(tmp_path, "def f(a: int) -> int:\n    return a + 1\n")


def test_rebind_ok_with_reason_suppresses_lit011(tmp_path):
    src = "def f(a: int) -> int:\n    a = 2  # rebind-ok: normalized in place\n    return a\n"
    assert "LIT011" not in _codes(tmp_path, src)


def test_for_target_rebinding_a_parameter_is_flagged(tmp_path):
    src = "def f(items: tuple, x: int) -> None:\n    for x in items:\n        pass\n"
    assert "LIT011" in _codes(tmp_path, src)


def test_unpacking_onto_parameters_flags_each(tmp_path):
    src = "def f(a: int, b: int) -> None:\n    a, b = (b, a)\n"
    assert _codes(tmp_path, src).count("LIT011") == 2


def test_rebinding_self_is_flagged(tmp_path):
    src = "class C:\n    def m(self, other: object) -> None:\n        self = other\n"
    assert "LIT011" in _codes(tmp_path, src)


def test_rebind_ok_with_reason_suppresses_mutation_sites(tmp_path):
    src = (
        "def f(payload: dict) -> None:\n"
        "    payload['k'] = 1  # rebind-ok: caller expects in-place update\n"
        "    del payload['j']  # rebind-ok: caller expects in-place update\n"
    )
    assert "LIT011" not in _codes(tmp_path, src)


def test_for_and_with_targets_on_parameter_attributes_are_mutation(tmp_path):
    src = (
        "def f(cfg: object) -> None:\n"
        "    for cfg.attr in (1, 2):\n"
        "        pass\n"
        "    with open('f') as cfg.handle:\n"
        "        pass\n"
    )
    assert _codes(tmp_path, src).count("LIT011") == 2


def test_nonlocal_rebinding_of_enclosing_parameter_names_the_owner(tmp_path):
    src = (
        "def outer(p: int) -> int:\n"
        "    def inner() -> None:\n"
        "        nonlocal p\n"
        "        p = 2\n"
        "    inner()\n"
        "    return p\n"
    )
    f = tmp_path / "snippet.py"
    f.write_text(src, encoding="utf-8")
    messages = [v.message for v in checker.check_file(f) if v.code == "LIT011"]
    assert len(messages) == 1
    assert "outer" in messages[0]


def test_lambda_parameter_walrus_rebinding_is_flagged(tmp_path):
    src = "from typing import Final\nf: Final = lambda x: (x := 2)\n"
    assert "LIT011" in _codes(tmp_path, src)


def test_walrus_in_own_defaults_binds_in_enclosing_scope_not_the_parameter(tmp_path):
    src = "def f(x: int, y: int = (x := 41)) -> None:\n    return None\n"
    assert "LIT011" not in _codes(tmp_path, src)


def test_walrus_in_nested_defaults_rebinds_the_enclosing_parameter(tmp_path):
    src = (
        "def g(p: int) -> None:\n"
        "    def inner(q: int = (p := 2)) -> None:\n"
        "        return None\n"
    )
    assert "LIT011" in _codes(tmp_path, src)


# --------------------------------------------------------------------------- #
# Writable TypedDict fields (LIT012)
# --------------------------------------------------------------------------- #


def test_typeddict_writable_field_is_flagged(tmp_path):
    src = "from typing import TypedDict\nclass P(TypedDict):\n    a: int\n"
    assert "LIT012" in _codes(tmp_path, src)


def test_typeddict_readonly_field_is_clean(tmp_path):
    src = (
        "from typing_extensions import ReadOnly, TypedDict\n"
        "class P(TypedDict):\n"
        "    a: ReadOnly[int]\n"
    )
    assert "LIT012" not in _codes(tmp_path, src)


def test_readonly_nests_with_qualifiers_annotated_and_forward_refs(tmp_path):
    src = (
        "import typing_extensions\n"
        "from typing import Annotated, TypedDict\n"
        "from typing_extensions import NotRequired, ReadOnly, Required\n"
        "class P(TypedDict):\n"
        "    a: Required[ReadOnly[int]]\n"
        "    b: NotRequired[typing_extensions.ReadOnly[int]]\n"
        "    c: ReadOnly[Required[int]]\n"
        "    d: Annotated[ReadOnly[int], 'meta']\n"
        "    e: 'Required[ReadOnly[int]]'\n"
    )
    assert "LIT012" not in _codes(tmp_path, src)


def test_readonly_in_annotated_metadata_position_does_not_qualify(tmp_path):
    src = (
        "from typing import Annotated, TypedDict\n"
        "from typing_extensions import ReadOnly, Required\n"
        "class P(TypedDict):\n"
        "    a: Annotated[int, ReadOnly]\n"
        "    b: Required[int]\n"
    )
    assert _codes(tmp_path, src).count("LIT012") == 2


def test_typeddict_subclass_in_same_module_is_flagged(tmp_path):
    src = (
        "from typing import TypedDict\n"
        "class Base(TypedDict):\n"
        "    pass\n"
        "class Child(Base, total=False):\n"
        "    a: int\n"
    )
    assert "LIT012" in _codes(tmp_path, src)


def test_plain_class_annotations_are_exempt(tmp_path):
    src = "class C:\n    a: int\nclass D(C):\n    b: int\n"
    assert "LIT012" not in _codes(tmp_path, src)


def test_functional_typeddict_fields_are_checked(tmp_path):
    src = (
        "from typing import Final, TypedDict\n"
        "from typing_extensions import ReadOnly\n"
        "P: Final = TypedDict('P', {'a': int, 'b': ReadOnly[int]})\n"
    )
    f = tmp_path / "snippet.py"
    f.write_text(src, encoding="utf-8")
    flagged = [v for v in checker.check_file(f) if v.code == "LIT012"]
    assert len(flagged) == 1
    assert "`a` of `P`" in flagged[0].message


def test_writable_ok_with_reason_suppresses_lit012(tmp_path):
    src = (
        "from typing import TypedDict\n"
        "class P(TypedDict):\n"
        "    a: int  # writable-ok: accumulated in place across stream chunks\n"
    )
    assert "LIT012" not in _codes(tmp_path, src)


def test_writable_ok_without_reason_is_lit005_and_does_not_suppress(tmp_path):
    src = (
        "from typing import TypedDict\n"
        "class P(TypedDict):\n"
        "    a: int  # writable-ok\n"
    )
    codes = _codes(tmp_path, src)
    assert "LIT005" in codes
    assert "LIT012" in codes


# --------------------------------------------------------------------------- #
# Budget integrity: every emittable LIT rule (bar the LIT000 read/parse error) is gated
# --------------------------------------------------------------------------- #


def test_budget_covers_exactly_the_checker_rules():
    budget = json.loads((_REPO_ROOT / "type-discipline-budget.json").read_text())
    emitted = set(re.findall(r"LIT\d{3}", _MODULE_PATH.read_text(encoding="utf-8"))) - {"LIT000"}
    assert set(budget) == emitted
    for spec in budget.values():
        assert isinstance(spec["limit"], int)
        assert spec["limit"] >= 0


_FANS_OUT = checker._worker_count(checker.PARALLEL_MIN_PATHS) > 1
_SERIAL_ONLY = "one usable core, so scan_paths stays serial and there is no fan-out to compare"


def _corpus(tmp_path: Path, count: int) -> tuple[Path, ...]:
    for index in range(count):
        (tmp_path / f"mod_{index}.py").write_text(
            f"def build_{index}(items: list[int]) -> None:\n    return None\n",
            encoding="utf-8",
        )
    return tuple(sorted(tmp_path.rglob("*.py")))


def _run_checker(target: Path) -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH), str(target)],
        capture_output=True, text=True, timeout=300,
    )
    return completed.stdout.splitlines()


def test_worker_count_stays_serial_below_the_threshold():
    assert checker._worker_count(checker.PARALLEL_MIN_PATHS - 1) == 1


def test_worker_count_fans_out_at_the_threshold():
    assert checker._worker_count(checker.PARALLEL_MIN_PATHS) == max(
        1, min(os.cpu_count() or 1, checker.MAX_WORKERS)
    )


def test_worker_count_never_exceeds_the_cap():
    assert checker._worker_count(100_000) <= checker.MAX_WORKERS


def test_scan_paths_below_the_threshold_returns_every_violation(tmp_path):
    paths = _corpus(tmp_path, 3)
    assert checker._worker_count(len(paths)) == 1
    found = checker.scan_paths(paths)
    assert found and len({v.path for v in found}) == 3


@pytest.mark.skipif(not _FANS_OUT, reason=_SERIAL_ONLY)
def test_a_fanned_out_run_reports_exactly_what_a_serial_run_reports(tmp_path):
    paths = _corpus(tmp_path, checker.PARALLEL_MIN_PATHS + 5)
    serial = [v.render() for v in sorted(v for path in paths for v in checker.check_file(path))]
    assert serial, "corpus must produce violations or the comparison proves nothing"
    assert _run_checker(tmp_path) == serial


@pytest.mark.skipif(not _FANS_OUT, reason=_SERIAL_ONLY)
def test_a_fanned_out_run_reports_each_generated_file_exactly_once(tmp_path):
    paths = _corpus(tmp_path, checker.PARALLEL_MIN_PATHS + 5)
    reported = _run_checker(tmp_path)
    assert reported
    assert len({line.split(":")[0] for line in reported}) == len(paths)

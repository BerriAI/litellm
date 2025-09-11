import pytest
from pytest import WarningsRecorder

from shapely.decorators import deprecate_positional


@deprecate_positional(["b", "c"])
def func_two(a, b=2, c=3):
    return a, b, c


@deprecate_positional(["b", "c", "d"])
def func_three(a, b=1, c=2, d=3):
    return a, b, c, d


@deprecate_positional(["b", "d"])
def func_noncontig(a, b=1, c=2, d=3):
    return a, b, c, d


@deprecate_positional(["b"], category=UserWarning)
def func_custom_category(a, b=1):
    return a, b


@deprecate_positional(["b"])
def func_varargs(a, b=1, *args):
    return a, b, args


@deprecate_positional([])
def func_no_deprecations(a, b=1):
    return a, b


def test_all_kwargs_no_warning(recwarn: WarningsRecorder) -> None:
    assert func_two(a=10, b=20, c=30) == (10, 20, 30)
    assert not recwarn.list


def test_only_required_arg_no_warning(recwarn: WarningsRecorder) -> None:
    assert func_two(1) == (1, 2, 3)
    assert not recwarn.list


def test_single_positional_warning() -> None:
    with pytest.warns(
        DeprecationWarning, match="positional argument `b` for `func_two` is deprecated"
    ):
        out = func_two(1, 4)
        assert out == (1, 4, 3)


def test_multiple_positional_warning() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="positional arguments `b` and `c` for `func_two` are deprecated",
    ):
        out = func_two(1, 4, 5)
        assert out == (1, 4, 5)


def test_three_positional_warning_oxford_comma() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="positional arguments `b`, `c`, and `d` for `func_three` are deprecated",
    ):
        out = func_three(1, 2, 3, 4)
        assert out == (1, 2, 3, 4)


def test_noncontiguous_partial_warning() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="positional argument `b` for `func_noncontig` is deprecated",
    ):
        out = func_noncontig(1, 2, 3)
        assert out == (1, 2, 3, 3)


def test_noncontiguous_full_warning() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="positional arguments `b` and `d` for `func_noncontig` are deprecated",
    ):
        out = func_noncontig(1, 2, 3, 4)
        assert out == (1, 2, 3, 4)


def test_custom_warning_category() -> None:
    with pytest.warns(
        UserWarning,
        match="positional argument `b` for `func_custom_category` is deprecated",
    ):
        out = func_custom_category(1, 2)
        assert out == (1, 2)


def test_func_no_deprecations_never_warns(recwarn: WarningsRecorder) -> None:
    out = func_no_deprecations(7, 8)
    assert out == (7, 8)
    assert not recwarn.list


def test_missing_required_arg_no_warning(recwarn: WarningsRecorder) -> None:
    with pytest.raises(TypeError):
        func_two()  # missing required 'a'  # type: ignore
    assert not recwarn.list


def test_unknown_keyword_no_warning(recwarn: WarningsRecorder) -> None:
    with pytest.raises(TypeError):
        func_two(1, 4, d=5)  # unknown keyword 'd'  # type: ignore
    assert not recwarn.list


def test_varargs_behavior_and_deprecation() -> None:
    with pytest.warns(
        DeprecationWarning,
        match="positional argument `b` for `func_varargs` is deprecated",
    ):
        out = func_varargs(1, 2, 3, 4)
        assert out == (1, 2, (3, 4))


def test_varargs_no_warning(recwarn: WarningsRecorder) -> None:
    out = func_varargs(1)
    assert out == (1, 1, ())
    assert not recwarn.list


def test_repeated_warnings() -> None:
    with pytest.warns(DeprecationWarning) as record:
        func_two(1, 4, 5)
        func_two(1, 4, 5)
        assert len(record) == 2
        assert str(record[0].message) == str(record[1].message)

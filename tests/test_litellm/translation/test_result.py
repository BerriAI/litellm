"""The contract that makes ``result.Result`` worth having over
``expression.Result``: variants are distinct classes, so the payload of the
other arm does not exist at runtime (and is a pyright error statically)."""

import dataclasses

import pytest

from litellm.translation.result import Error, Ok, Result


def test_ok_has_no_error_attribute():
    with pytest.raises(AttributeError):
        Ok(42).error  # type: ignore[attr-defined]


def test_error_has_no_ok_attribute():
    with pytest.raises(AttributeError):
        Error("boom").ok  # type: ignore[attr-defined]


def test_match_discriminates_by_class():
    def unwrap(result: Result[int, str]) -> int:
        match result:
            case Ok(ok=value):
                return value
            case Error(error=err):
                return len(err)

    assert unwrap(Ok(7)) == 7
    assert unwrap(Error("boom")) == 4


def test_isinstance_discriminates():
    result: Result[int, str] = Error("boom")
    assert isinstance(result, Error)
    assert not isinstance(result, Ok)


def test_predicates():
    assert Ok(1).is_ok() and not Ok(1).is_error()
    assert Error("x").is_error() and not Error("x").is_ok()


def test_map_applies_only_to_ok():
    assert Ok(2).map(lambda v: v * 10) == Ok(20)
    assert Error("boom").map(lambda v: v * 10) == Error("boom")


def test_bind_chains_only_through_ok():
    def halve(value: int) -> Result[int, str]:
        return Ok(value // 2) if value % 2 == 0 else Error("odd")

    assert Ok(8).bind(halve) == Ok(4)
    assert Ok(3).bind(halve) == Error("odd")
    assert Error("upstream").bind(halve) == Error("upstream")


def test_variants_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Ok(1).ok = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Error("a").error = "b"  # type: ignore[misc]

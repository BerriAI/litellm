import dataclasses
from collections.abc import Mapping, MutableMapping
from typing import Final, get_args

import pytest

from litellm.harness.options import HarnessOptions


def is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_defaults_pass_python_311_unhashable_default_check(options_type: type) -> None:
    unhashable: Final = tuple(
        options_field.name
        for options_field in dataclasses.fields(options_type)
        if options_field.default is not dataclasses.MISSING and not is_hashable(options_field.default)
    )
    assert unhashable == ()


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_mapping_defaults_are_read_only(options_type: type) -> None:
    options: Final = options_type()
    values: Final = tuple(getattr(options, options_field.name) for options_field in dataclasses.fields(options_type))
    writable: Final = tuple(value for value in values if isinstance(value, MutableMapping))
    assert writable == ()
    assert all(value == {} for value in values if isinstance(value, Mapping))

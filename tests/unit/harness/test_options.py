import dataclasses
from typing import get_args

import pytest

from litellm.harness.options import HarnessOptions


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_defaults_pass_python_311_unhashable_default_check(options_type: type) -> None:
    for options_field in dataclasses.fields(options_type):
        if options_field.default is not dataclasses.MISSING:
            hash(options_field.default)


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_mapping_defaults_are_read_only(options_type: type) -> None:
    options = options_type()
    for options_field in dataclasses.fields(options_type):
        value = getattr(options, options_field.name)
        if hasattr(value, "keys"):
            with pytest.raises(TypeError):
                value["key"] = "value"

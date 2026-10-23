import copy
import dataclasses
from collections.abc import Mapping
from typing import Final, get_args

import pytest

from litellm.harness.options import HarnessOptions


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_default_options_survive_deepcopy_and_asdict(options_type: type) -> None:
    options: Final = options_type()
    assert copy.deepcopy(options) == options
    assert dataclasses.asdict(options) == vars(options)


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_each_options_instance_gets_its_own_empty_mappings(options_type: type) -> None:
    first: Final = vars(options_type())
    second: Final = vars(options_type())
    mapping_names: Final = tuple(name for name, value in first.items() if isinstance(value, Mapping))
    assert {name: first[name] for name in mapping_names} == dict.fromkeys(mapping_names, {})
    assert [name for name in mapping_names if first[name] is second[name]] == [], "defaults shared between instances"

from collections.abc import Mapping, MutableMapping
from typing import Final, get_args

import pytest

from litellm.harness.options import HarnessOptions


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_each_options_instance_gets_its_own_empty_read_only_mappings(options_type: type) -> None:
    first: Final = vars(options_type())
    second: Final = vars(options_type())
    mapping_names: Final = tuple(name for name, value in first.items() if isinstance(value, Mapping))
    assert {name: first[name] for name in mapping_names} == dict.fromkeys(mapping_names, {})
    assert [name for name in mapping_names if isinstance(first[name], MutableMapping)] == [], "writable defaults"
    assert [name for name in mapping_names if first[name] is second[name]] == [], "defaults shared between instances"

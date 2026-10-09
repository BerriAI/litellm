from collections.abc import Mapping, MutableMapping
from typing import Final, get_args

import pytest

from litellm.harness.options import HarnessOptions


@pytest.mark.parametrize("options_type", get_args(HarnessOptions), ids=lambda t: t.__name__)
def test_each_options_instance_gets_its_own_empty_read_only_mappings(options_type: type) -> None:
    first: Final = vars(options_type())
    second: Final = vars(options_type())
    mapping_names: Final = tuple(name for name, value in first.items() if isinstance(value, Mapping))
    assert all(first[name] == {} for name in mapping_names)
    assert not any(isinstance(first[name], MutableMapping) for name in mapping_names)
    assert all(first[name] is not second[name] for name in mapping_names)

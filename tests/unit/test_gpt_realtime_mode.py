from typing import get_args

from typing_extensions import get_type_hints

from litellm.types.utils import ModelInfoBase


def test_realtime_is_a_valid_mode_literal():
    hints = get_type_hints(ModelInfoBase, include_extras=False)
    assert "realtime" in get_args(hints["mode"])

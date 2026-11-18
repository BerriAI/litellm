from typing import Final

import pytest

from litellm.responses.mcp.parked_stream_dispatch import parked_success
from litellm.types.utils import ModelResponse

_ASSEMBLED: Final = ModelResponse(id="chatcmpl-parked", model="gpt-5.6")


@pytest.mark.parametrize(
    "parked",
    [(), (_ASSEMBLED,), (_ASSEMBLED, False, "not a logger"), (_ASSEMBLED, False, None, None)],
    ids=("empty", "assembled-alone", "third-part-is-not-a-logger", "four-parts"),
)
def test_a_parked_dispatch_of_an_unknown_shape_is_never_dispatched(parked: tuple[object, ...]):
    assert parked_success(parked) is None

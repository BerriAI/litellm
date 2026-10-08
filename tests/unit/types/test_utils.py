import pytest

from litellm.types.utils import CaptureMessageContent


@pytest.mark.parametrize("mode", list(CaptureMessageContent))
def test_capture_message_content_formats_as_its_bare_value(mode: CaptureMessageContent) -> None:
    assert (str(mode), f"{mode}") == (mode.value, mode.value)

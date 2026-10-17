from typing import Final

import pytest

import litellm
from litellm import utils
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.initialize_dynamic_callback_params import validate_excluded_services_value
from litellm.types.utils import (
    CAPTURE_MESSAGE_CONTENT_VALUES,
    CallTypes,
    CaptureMessageContent,
    captures_span_content,
)


@pytest.mark.parametrize("value", ["", "  ", "auth", "none,redis", "redis,", ",postgres"])
def test_invalid_excluded_services_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid excluded_services.*none"):
        validate_excluded_services_value(value)


@pytest.mark.parametrize(
    "mode",
    [
        CaptureMessageContent.NO_CONTENT,
        CaptureMessageContent.SPAN_ONLY,
        CaptureMessageContent.EVENT_ONLY,
        CaptureMessageContent.SPAN_AND_EVENT,
    ],
)
def test_capture_message_content_members_are_plain_strings(mode: str) -> None:
    assert type(mode) is str
    assert (str(mode), f"{mode}", repr(mode)) == (mode, mode, repr(str(mode)))


def test_a_destination_may_only_set_the_span_capture_modes() -> None:
    assert CAPTURE_MESSAGE_CONTENT_VALUES == {"no_content", "span_only"}


def test_capture_message_content_constructor_accepts_any_string() -> None:
    assert CaptureMessageContent("custom") == "custom"


@pytest.mark.parametrize(
    ("mode", "captures"),
    [
        ("no_content", False),
        ("span_only", True),
        ("event_only", False),
        ("span_and_event", True),
        ("bogus", False),
        (None, False),
    ],
)
def test_captures_span_content(mode: str | None, captures: bool) -> None:
    assert captures_span_content(mode) is captures


@pytest.mark.parametrize(
    "call_type",
    [
        "get_responses",
        "aget_responses",
        "delete_responses",
        "adelete_responses",
        "cancel_responses",
        "acancel_responses",
        "compact_responses",
        "acompact_responses",
    ],
)
@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_passes_responses_call_type(
    call_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingLogger(CustomLogger):
        received_call_type: CallTypes | None = None

        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object] | None:
            self.received_call_type = call_type
            return None

    callback: Final = RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [callback])

    await utils.async_pre_call_deployment_hook({}, call_type)

    assert callback.received_call_type == call_type

import subprocess
import sys
from typing import Final

import pytest

from litellm.litellm_core_utils.initialize_dynamic_callback_params import validate_excluded_services_value
from litellm.types.utils import (
    CAPTURE_MESSAGE_CONTENT_VALUES,
    CallTypes,
    CaptureMessageContent,
    captures_span_content,
)


def test_parallel_first_usage_construction_does_not_race_schema_build() -> None:
    probe: Final = r'''
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from litellm.types.utils import Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper

source, first_line = inspect.getsourcelines(BaseModel.model_rebuild)
delete_line = next((first_line + i for i, line in enumerate(source) if "delattr(cls, attr)" in line), None)
sys.setswitchinterval(1e-6)

for model in (Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper):
    barrier = threading.Barrier(8, timeout=5)
    start_barrier = threading.Barrier(8, timeout=5)
    def trace(frame, event, arg):
        if (event == "line" and frame.f_code is BaseModel.model_rebuild.__func__.__code__
            and delete_line is not None and frame.f_lineno == delete_line and frame.f_locals.get("cls") is model
            and frame.f_locals.get("attr") == "__pydantic_core_schema__"):
            barrier.wait()
        return trace

    def construct(index):
        sys.settrace(trace)
        try:
            start_barrier.wait()
            return model().model_dump()
        finally:
            sys.settrace(None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(construct, range(8)))
    assert len(results) == 8
    assert all(result == results[0] for result in results)
    print(model.__name__ + ": 8 parallel constructions passed")
'''
    result: Final = subprocess.run(
        [sys.executable, "-P", "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize("call_type", ["compact_responses", "acompact_responses"])
def test_responses_compact_call_types_round_trip_their_public_names(call_type: str) -> None:
    assert CallTypes(call_type).value == call_type

import pytest
from typing import List, cast
from unittest.mock import patch

import litellm
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.types.llms.openai import AllMessageValues


def test_missing_file_id_and_file_data_raises_bad_request_error():
    """When file element has neither file_id nor file_data, a BadRequestError is raised."""
    messages = cast(List[AllMessageValues], [{"role": "user", "content": [{"type": "file", "file": {}}]}])
    with pytest.raises(litellm.BadRequestError) as exc_info:
        _gemini_convert_messages_with_history(messages, model="gemini-1.5-pro")
    assert "Unknown file type" in str(exc_info.value)


def test_image_fetch_error_raises_bad_request_error():
    """ImageFetchError from _process_gemini_media is re-raised as BadRequestError."""
    messages = cast(
        List[AllMessageValues],
        [{"role": "user", "content": [{"type": "file", "file": {"file_id": "some_id"}}]}],
    )
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._process_gemini_media",
        side_effect=litellm.ImageFetchError(
            message="403 Forbidden",
            model="gemini-1.5-pro",
            llm_provider="vertex_ai",
        ),
    ):
        with pytest.raises(litellm.BadRequestError) as exc_info:
            _gemini_convert_messages_with_history(messages, model="gemini-1.5-pro")
    assert "403 Forbidden" in str(exc_info.value)


def test_generic_exception_raises_bad_request_error_with_mime_message():
    """Generic exception from _process_gemini_media is wrapped as BadRequestError with MIME message."""
    messages = cast(
        List[AllMessageValues],
        [{"role": "user", "content": [{"type": "file", "file": {"file_id": "some_id"}}]}],
    )
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._process_gemini_media",
        side_effect=ValueError("cannot determine mime type"),
    ):
        with pytest.raises(litellm.BadRequestError) as exc_info:
            _gemini_convert_messages_with_history(messages, model="gemini-1.5-pro")
    assert "Unable to determine mime type" in str(exc_info.value)
    assert "cannot determine mime type" in str(exc_info.value)


def test_bad_request_error_from_process_gemini_media_is_re_raised_as_is():
    """BadRequestError from _process_gemini_media is re-raised verbatim (not replaced by generic message)."""
    original_message = "Invalid image received - https://example.com/img.png. Supported formats are..."
    messages = cast(
        List[AllMessageValues],
        [{"role": "user", "content": [{"type": "file", "file": {"file_id": "some_id"}}]}],
    )
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._process_gemini_media",
        side_effect=litellm.BadRequestError(
            message=original_message,
            model="gemini-1.5-pro",
            llm_provider="vertex_ai",
        ),
    ):
        with pytest.raises(litellm.BadRequestError) as exc_info:
            _gemini_convert_messages_with_history(messages, model="gemini-1.5-pro")
    assert original_message in str(exc_info.value)
    assert "Unable to determine mime type" not in str(exc_info.value)
